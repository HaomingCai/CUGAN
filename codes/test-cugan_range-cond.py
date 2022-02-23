import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch
import numpy as np

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

import lpips as lpips_metric
from DISTS_pytorch import DISTS as dists_metric

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['lpips'] = []
    test_results['dists'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    D_lpips = lpips_metric.LPIPS(net='alex')
    D_dists = dists_metric()

    for data in test_loader:

        img_path = data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        dataset_img_dir = osp.join(opt['path']['results_root'], test_set_name, img_name)
        util.mkdir(dataset_img_dir)

        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        cond_init = test_loader.dataset.opt['cond_init']   # [0.0, 0.1]
        range_mode = test_loader.dataset.opt['range_mode'] # 1 / 2 / 3
        range_stride = test_loader.dataset.opt['range_stride'] # 0.1

        # Prepare all cond set by the user | obtain a list of conds
        if range_mode == 2:
            conds_list_deblur  = np.arange(cond_init[0], 1.000001, range_stride).tolist()
            conds_list_denoise = np.arange(cond_init[1], 1.000001, range_stride).tolist()
        elif range_mode == 0:
            conds_list_deblur = np.arange(cond_init[0], 1.000001, range_stride).tolist()
            conds_list_denoise = [cond_init[1]]
        elif range_mode == 1:
            conds_list_deblur = [cond_init[0]]
            conds_list_denoise = np.arange(cond_init[1], 1.000001, range_stride).tolist()
        else:
            raise NotImplementedError(
                "In test-cugan_range-cond.py, you must provide right cond_mode settings in your yml file.")

        conds_list = []
        for cond_deblur in conds_list_deblur:
            for cond_denoise in conds_list_denoise:
                conds_list.append([cond_deblur, cond_denoise])

        for idx, cond in enumerate(conds_list):

            if cond is not None:
                data['cond'] = torch.Tensor(cond).view(1, 2)
                need_cond = True
            elif data['cond'] is not None:
                need_cond = True
            else:
                raise NotImplementedError(
                    "In test-cugan_range-cond.py, you must provide cond settings in your yml file.")

            model.feed_data(data, need_GT=need_GT, need_cond=need_cond)

            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)

            sr_img = util.tensor2img(visuals['rlt'])  # uint8

            # save images
            save_img_path = osp.join(dataset_dir, img_name, img_name + '_{}_deblur{:.3f}_denoise{:.3f}'.format(idx, cond[0], cond[1]) + '.png')
            util.save_img(sr_img, save_img_path)

            # calculate PSNR / SSIM / LPIPS / DISTS
            metric_list = opt['evaluate_metric']
            if need_GT:
                for metric_name in metric_list:
                    gt_img = util.tensor2img(visuals['GT'])
                    if metric_name == 'psnr':
                        psnr = util.calculate_psnr(sr_img, gt_img)
                        test_results['psnr'].append(psnr)
                    elif metric_name == 'ssim':
                        ssim = util.calculate_ssim(sr_img, gt_img)
                        test_results['ssim'].append(ssim)
                    elif metric_name == 'dists':
                        GT_valid, SR_valid = torch.unsqueeze(visuals['GT'], 0), torch.unsqueeze(visuals['rlt'], 0)
                        dists = D_dists(GT_valid, SR_valid, require_grad=False).item()
                        test_results['dists'].append(dists)
                    elif metric_name == 'lpips':
                        GT_valid, SR_valid = torch.unsqueeze(visuals['GT'], 0), torch.unsqueeze(visuals['rlt'], 0)
                        lpips = D_lpips(GT_valid, SR_valid).item()
                        test_results['lpips'].append(lpips)
                    elif metric_name == 'ssim_y' and gt_img.shape[2] == 3:
                        sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
                        gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)
                        ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                        test_results['ssim_y'].append(ssim_y)
                    elif metric_name == 'psnr_y' and gt_img.shape[2] == 3:
                        sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
                        gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)
                        psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                        test_results['psnr_y'].append(psnr_y)
                    else:
                        raise NotImplementedError("You are asking a metric evaluation that we do not support yet")

                evaluation_info = '{:20s} Conds: deblur {:.3f} denoise {:.3f} -'.format(img_name, cond[0], cond[1])
                for metric_name in metric_list:
                    evaluation_info += '{}: {:.6f}'.format(metric_name, test_results[metric_name][-1])
                logger.info(evaluation_info)
            else:
                logger.info('{}, Conds: deblur {:.3f} denoise {:.3f}'.format(img_name, cond[0], cond[1]))

