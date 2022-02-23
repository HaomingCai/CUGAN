import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
import time

logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network


        if self.is_train:
            self.netG.train()
            # G pixel loss
            self.using_pre_MSE     = self.opt['train']['pre_G_MSE']
            self.fix_G_cond        = self.opt['network_G']['fix_G_cond']
            self.fix_G             = self.opt['network_G']['fix_G']
            self.reweighting = self.opt['train']['loss_reweighting']
            print("self.reweighting is :{} ...... type is {}".format(self.reweighting, type(self.reweighting)))
            print("self.using_pre_MSE is :{} ...... type is {}".format(self.using_pre_MSE, type(self.using_pre_MSE)))
            print("self.fix_G_cond is :{} ...... type is {}".format(self.fix_G_cond, type(self.fix_G_cond)))
            print("self.fix_G is :{} ...... type is {}".format(self.fix_G, type(self.fix_G)))
            print('=================================================================================================')
            time.sleep(6)
            if self.using_pre_MSE:
                print('You Are Using Pre-Trained MSE to Pre-process the input image firstly !!!')
                print('=================================================================================================')
                time.sleep(2)
                self.pre_G = networks.define_pre_G(opt).to(self.device)
                if opt['dist']:
                    self.pre_G = DistributedDataParallel(self.pre_G, device_ids=[torch.cuda.current_device()])
                else:
                    self.pre_G = DataParallel(self.pre_G)
                for k, v in self.pre_G.named_parameters():
                    v.requires_grad = False  # 固定参数

            # Fix Condition Network
            if self.fix_G_cond:
                print('In Fix G cond codes')
                for k, v in self.netG.named_parameters():
                    if 'scale' in k or 'cond' in k:
                        v.requires_grad = False  # 固定参数
            else:
                print('NOT in Fix G cond codes')
            time.sleep(5)

            # Fix Generative Network
            if self.fix_G:
                print('In Fix G codes')
                for k, v in self.netG.named_parameters():
                    if ('scale' not in k) and ('cond' not in k):
                        v.requires_grad = False  # 固定参数
            else:
                print('NOT in Fix G cond codes')
            time.sleep(5)

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                if train_opt['loss_reweighting']:
                    self.cri_pix = nn.L1Loss(reduce=False).to(self.device)
                else:
                    self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            if train_opt['finetune_adafm']:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    v.requires_grad = False
                    if k.find('adafm') >= 0:
                        v.requires_grad = True
                        optim_params.append(v)
                        logger.info('Params [{:s}] will optimize.'.format(k))
            else:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()
        self.load()

    def feed_data(self, data, need_GT=True, need_cond=False):
        if self.opt['network_G']['which_model_G'] == 'CUGAN':
            B, C, H, W = data['LQ'].shape
            H_num, W_num = H % 8, W % 8
            if H_num != 0:
                data['LQ'] = data['LQ'][:, :, :-H_num, :]
            if W_num != 0:
                data['LQ'] = data['LQ'][:, :, :, :-W_num]
            self.var_L = data['LQ'].to(self.device)
        else:
            self.var_L = data['LQ'].to(self.device)  # LQ

        if need_GT:
            if self.opt['network_G']['which_model_G'] == 'CUGAN':
                B, C, H, W = data['GT'].shape
                H_num, W_num = H % 8, W % 8
                if H_num != 0:
                    data['GT'] = data['GT'][:, :, :-H_num, :]
                if W_num != 0:
                    data['GT'] = data['GT'][:, :, :, :-W_num]
                self.real_H = data['GT'].to(self.device)
            else:
                self.real_H = data['GT'].to(self.device)  # GT

        if need_cond:
            self.cond = data['cond'].to(self.device) # cond
            if self.using_pre_MSE:
                self.var_L = self.pre_G([self.var_L, self.cond])
            self.input = [self.var_L, self.cond]
        else:
            self.input = self.var_L


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.input)

        if self.reweighting:
            weight = 1.1 - self.cond
            re_weight = torch.mean(weight, dim=1)
            l_pix = re_weight.view(-1, 1, 1, 1) * self.cri_pix(self.fake_H, self.real_H)
            l_pix = l_pix.mean()
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)

        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.input)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

        load_path_pre_G = self.opt['path']['pretrain_model_pre_G']
        if self.using_pre_MSE and load_path_pre_G is not None:
            logger.info('Loading model for pre_G [{:s}] ...'.format(load_path_pre_G))
            self.load_network(load_path_pre_G, self.pre_G, self.opt['path']['strict_load'])

        if self.opt['train']['AdaFM_Finetune']:
            for param in self.netG.parameters():
                if param == 'AdaFM':
                    param.requires_grad = True
                else:
                    print(param)
                    param.requires_grad = False
            time.sleep(5)

    def update(self, new_model_dict):
        if isinstance(self.netG, nn.DataParallel):
            network = self.netG.module
            network.load_state_dict(new_model_dict)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
