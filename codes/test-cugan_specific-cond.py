import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch

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

    cond = test_loader.dataset.opt['cond']

    D_lpips = lpips_metric.LPIPS(net='alex')
    D_dists = dists_metric()

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True

        if cond is not None:
            data['cond'] = torch.Tensor(cond).view(1, 2)
            need_cond = True
        elif data['cond'] is not None:
            need_cond = True
        else:
            raise NotImplementedError("In test-cugan_specific-cond.py, you must provide cond (such as [0.3, 0.2]) in your yml file.")
            
        model.feed_data(data, need_GT=need_GT, need_cond=need_cond)

        img_path = data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals(need_GT=need_GT)

        sr_img = util.tensor2img(visuals['rlt'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
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

    metric_list = opt['evaluate_metric']
    if need_GT:
        # Average PSNR/SSIM results
        ave_evaluation_info = '----Average PSNR/SSIM results for {}----\n\t'.format(test_set_name)
        for metric_name in metric_list:
            ave_metric = sum(test_results[metric_name]) / len(test_results[metric_name])
            ave_evaluation_info += '{}: {}'.format(metric_name, ave_metric)
        logger.info(ave_evaluation_info)

