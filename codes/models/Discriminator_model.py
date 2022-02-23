import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import GANLoss
import time

logger = logging.getLogger('base')


class DModel(BaseModel):
    def __init__(self, opt):
        super(DModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netD = networks.define_D(opt).to(self.device)
        if opt['dist']:
            self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
        else:
            self.netD = DataParallel(self.netD)



        self.gan_D_reweighting = self.opt['train']['ganD_loss_reweighting']
        self.gan_G_reweighting = self.opt['train']['ganG_loss_reweighting']
        self.fix_G_cond        = self.opt['network_G']['fix_G_cond']
        print("self.gan_D_reweighting is :{} ...... type is {}".format(self.gan_D_reweighting, type(self.gan_D_reweighting)))
        print("self.gan_G_reweighting is :{} ...... type is {}".format(self.gan_G_reweighting, type(self.gan_G_reweighting)))
        print("self.fix_G_cond is :{} ...... type is {}".format(self.fix_G_cond, type(self.fix_G_cond)))
        print('=================================================================================================')
        self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
        self.l_gan_w = train_opt['gan_weight']

        self.log_dict = OrderedDict()

        # self.print_network()  # print network
        self.load()  # load G and D if needed



    def feed_data(self, data, need_GT=True, need_cond=False):
        B, C, H, W = data['LQ'].shape
        self.var_L = data['LQ'][:,:, int(H/2)-32:int(H/2)+32, int(W/2)-32:int(W/2)+32].to(self.device)  # LQ
        self.real_H = data['GT'][:,:, int(H/2)-32:int(H/2)+32, int(W/2)-32:int(W/2)+32].to(self.device)  # GT
        if need_cond:
            self.cond = data['cond'].to(self.device) # cond
            self.input = [self.var_L, self.cond]
        else:
            self.input = self.var_L



    def optimize_parameters(self, step):
        pass

    def test(self):
        print(self.var_L.shape)
        self.netD.eval()
        with torch.no_grad():
            self.pred_fake = self.netD(self.var_L, self.cond)
            self.pred_real = self.netD(self.real_H, self.cond)


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['pred_fake'] = self.pred_fake.detach()[0].float().cpu()
        out_dict['pred_real'] = self.pred_real.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):

        load_path_D = self.opt['path']['pretrain_model_D']
        logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
        self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])


    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
