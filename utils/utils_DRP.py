from utils import utils_image as util
import torch
from utils.util_sr import *
from models.network_swinir import SwinIR as net
import torch.nn as nn
from scipy import ndimage


class conj_function_deblur(nn.Module):
    """
    performs DC step
    """

    def __init__(self, kernel_forward, sr_prior, lam):
        super(conj_function_deblur, self).__init__()
        self.kernel_forward = kernel_forward
        self.sr_prior = sr_prior
        self.lam = lam


    def forward(self, im):  # step for batch image
        """
        :im: input image (B x nrow x nrol)
        """
        ax = ndimage.filters.convolve(im.squeeze().permute(1, 2, 0).cpu(), np.expand_dims(self.kernel_forward.cpu().squeeze().cpu().numpy(), axis=2), mode='wrap')
        atax = ndimage.filters.convolve(ax, np.expand_dims(self.kernel_forward.cpu().squeeze().cpu().numpy()[::-1, ::-1], axis=2), mode='wrap')
        atax = torch.tensor(atax).permute(2, 0, 1).unsqueeze(0)

        hx =imresize(im.squeeze(), 1 / self.sr_prior, True)
        hthx = imresize(hx, self.sr_prior, True).unsqueeze(0)
        return atax + self.lam * hthx

class conj_function_sisr(nn.Module):
    """
    performs DC step
    """

    def __init__(self, kernel_forward, sr_forward, sr_prior, lam):
        super(conj_function_sisr, self).__init__()
        self.kernel_forward = kernel_forward
        self.sr_forward = sr_forward
        self.sr_prior = sr_prior
        self.lam = lam

    def forward(self, im):  # step for batch image
        """
        :im: input image (B x nrow x nrol)
        """
        ax = ndimage.filters.convolve(im.squeeze().permute(1, 2, 0).cpu(),
                                      np.expand_dims(self.kernel_forward.squeeze().cpu().numpy(), axis=2),
                                      mode='wrap')
        ax = imresize(torch.tensor(ax).permute(2, 0, 1), 1 / self.sr_forward, False)
        atax = imresize(ax, self.sr_forward, False).permute(1, 2, 0).numpy()

        atax = ndimage.filters.convolve(atax,
                                        np.expand_dims(self.kernel_forward.squeeze().numpy()[::-1, ::-1], axis=2),
                                        mode='wrap')
        atax = shift_pixel(atax, self.sr_forward)
        atax = torch.tensor(atax).permute(2, 0, 1).unsqueeze(0)

        hx = imresize(im.squeeze(), 1 / self.sr_prior, True)
        hthx = imresize(hx, self.sr_prior, True).unsqueeze(0)

        out = atax + self.lam * hthx

        return out

class conj_function_denoise(nn.Module):
    """
    performs DC step
    """

    def __init__(self, sr_prior, lam):
        super(conj_function_denoise, self).__init__()
        self.sr_prior = sr_prior
        self.lam = lam


    def forward(self, im):  # step for batch image
        """
        :im: input image (B x nrow x nrol)
        """
        atax = im

        hx =imresize(im.squeeze(), 1 / self.sr_prior, True)
        hthx = imresize(hx, self.sr_prior, True).unsqueeze(0)
        return atax + self.lam * hthx



def H_function(img, scale=3):
    if img.is_cuda:
        img = img.cpu()
    img = img.squeeze()
    img_lq = imresize(img, 1 / scale, True)
    img_lq = img_lq.unsqueeze(0)
    return img_lq
def HT_function(img, scale=3):
    if img.is_cuda:
        img = img.cpu()
    img = img.squeeze()
    img_lq = imresize(img, scale, True)
    img_lq = img_lq.unsqueeze(0)
    return img_lq

def define_model(model_path, scale=3, training_patch_size=48):
    # 001 classical image sr
    model = net(upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(
        pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
        strict=True)

    return model



def test(img_lq, model):
    output = model(img_lq)
    return output

