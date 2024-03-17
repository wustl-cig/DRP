import os.path
import logging
from collections import OrderedDict
import hdf5storage
from scipy import ndimage
import sys
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))

from utils import utils_conj as conj

from utils import utils_logger
from utils import utils_image as util
from utils.util_sr import *
from utils.utils_DRP import *




""" 
The code is built based on code of DPIR
github: https://github.com/cszn/DPIR 
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
"""


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 2.55 / 255.0  # default: 0, noise level for LR image
    noise_level_model = noise_level_img  # noise level of model, default 0
    model_name = 'DRP'
    testset_name = 'set3c'  # test set,  'set3c'| 'set5' | 'cbsd68'| 'McMaster'

    iter_num = 35  # number of iterations

    show_img = True  # default: False
    save_L = True  # save LR image
    save_E = True  # save estimated image
    border = 0

    # --------------------------------
    # load kernel
    # --------------------------------
    kernel = hdf5storage.loadmat(os.path.join('./kernels', 'kernel_1.6.mat'))[
        'kernel']

    task_current = 'deblur'  # 'deblur' for deblurring
    n_channels = 3  # fixed
    model_zoo = './model_zoo'  # fixed
    testsets = './testsets'  # fixed
    results = './pub_results'  # fixed
    result_name = testset_name + '_' + task_current + '_' + model_name
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # --------------------------------
    # SR prior path
    # --------------------------------
    model_path_3x = './model_zoo/swinir/SwinIR_3x.pth'
    model_path_2x = './model_zoo/swinir/SwinIR_2x.pth'

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name)  # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------


    logger.info(
        'model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []  # record average PSNR for each kernel

    logger.info('-------k:std = 1.6 ---------')
    test_results = OrderedDict()
    test_results['psnr'] = []
    k = kernel.astype(np.float64)
    util.imshow(k) if show_img else None

    for idx, img in enumerate(L_paths):

        # --------------------------------
        # (1) get img_L
        # --------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_H = util.modcrop(img_H, 12)  # modcrop

        img_L = ndimage.filters.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
        util.imshow(img_L) if show_img else None
        img_L = util.uint2single(img_L)

        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img, img_L.shape)  # add AWGN

        # --------------------------------
        # (3) initialize x, and pre-calculation
        # --------------------------------

        x = util.single2tensor4(img_L).to(device)

        aty = ndimage.filters.convolve(img_L, np.expand_dims(
                k[::-1, ::-1], axis=2), mode='wrap')

        aty = util.single2tensor4(aty)

        img_L_tensor, k_tensor = util.single2tensor4(img_L), util.single2tensor4(np.expand_dims(k, 2))
        [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
        h, w = x.shape[-2], x.shape[-1]

        # --------------------------------
        # (4) main iterations
        # --------------------------------
        for i in range(iter_num):

            x_pre = x

            if i < 30:
                sr_rate = 3
                sup_model = define_model(model_path_3x, scale=3, training_patch_size=64)
                sup_model.eval()
                sup_model = sup_model.to(device)

                z_lq = H_function(x, 3).to(device)

                with torch.no_grad():
                    x = test(z_lq, sup_model)
                    x = x[..., :h, :w]

            else:
                sr_rate = 2
                sup_model = define_model(model_path_2x, scale=2, training_patch_size=64)
                sup_model.eval()
                sup_model = sup_model.to(device)

                z_lq = H_function(x, 2).to(device)

                with torch.no_grad():
                    x = test(z_lq, sup_model)
                    x = x[..., :h, :w]

            out = x

            if i < 30:
                z = 0.1 * x + (1-0.1) * x_pre
            else:
                z = 0.05 * x + (1-0.05) * x_pre

            function = conj_function_deblur(kernel_forward = k_tensor, sr_prior=sr_rate, lam=0.04)
            b = 0.04 * HT_function(H_function(z, sr_rate), sr_rate) + aty
            conj_cal = conj.ConjGrad(b, function, max_iter=3, l2lam=0, verbose=False)
            x = conj_cal(z.cpu()).to(device)


        # --------------------------------
        # (3) img_E
        # --------------------------------

        img_E = util.tensor2uint(out)
        util.imshow(img_E) if show_img else None

        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name + '_std=1.6' + '_' + "Recon" + '.png'))

        if save_L:
            util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name + '_std=1.6' + '_LR.png'))

        psnr = util.calculate_psnr(img_E, img_H, border=border)  # change with your own border

        test_results['psnr'].append(psnr)
        logger.info('{:->4d}--> {:>10s} --k:std=1.6 PSNR: {:.5f}dB'.format(idx + 1, img_name + ext, psnr))

    # --------------------------------
    # Average PSNR
    # --------------------------------

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    logger.info(
        '------> Average PSNR of ({}), kernel: (std = 1.6) sigma: ({:.2f}): {:.2f} dB'.format(testset_name,
                                                                                           noise_level_model, ave_psnr))
    test_results_ave['psnr'].append(ave_psnr)



if __name__ == '__main__':
    main()