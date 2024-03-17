# -*- coding: utf-8 -*-
import torch.fft
import torch
import numpy as np
from scipy import fftpack
import torch
from scipy import ndimage
from scipy.interpolate import interp2d


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2,-1))
    #n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    #otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]



def data_solution(x, FB, FBC, F2B, FBFy, alpha, sf):
    FR = FBFy + torch.fft.fftn(alpha*x, dim=(-2,-1))
    x1 = FB * FR
    # x1 = FB.mul(FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW + alpha)
    FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
    FX = (FR-FCBinvWBR)/alpha
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

    return Xest


def data_solution_weighted(x, FB, FB_forward, FBC, F2B, F2B_forward, FBFy, alpha, sf, sf_forward):
    # FR = FBFy + torch.fft.fftn(alpha*x, dim=(-2,-1))
    # x = shift_pixel(x, 3)
    # x = downsample(x, 3)
    # x = upsample(x, 3)

    FBC_forward = torch.conj(FB_forward)
    # FR = FBFy + FBC_forward * torch.fft.fftn(alpha * x, dim=(-2, -1))
    FR = FBFy + torch.fft.fftn(alpha * x, dim=(-2, -1))

    # FR = FBFy + torch.fft.fftn(alpha*x, dim=(-2,-1))


    FR = FB_forward.mul(FR)

    x1 = FB.mul(FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)


    # invW = torch.mean(splits(F2B_forward, sf))
    # a = torch.mean(splits(FB, sf), dim=-1, keepdim=False)
    # b = torch.mean(splits(FBC, sf), dim=-1, keepdim=False)
    # invW_forward = a * invW * b

    # F2B_weighted = torch.abs(FB * F2B_forward * FBC)
    F2B_weighted = torch.abs(FBC * F2B_forward * FB)

    # invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invW_forward = torch.mean(splits(F2B_weighted, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW_forward + alpha)
    FCBinvWBR = F2B_forward * FBC * invWBR.repeat(1, 1, sf, sf)
    FX = (FR-FCBinvWBR)/alpha
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

    return Xest



def pre_calculate(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy


def pre_calculate_weighted(x, k, k_forward, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        k_forward: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FB_forward = p2o(k_forward, (w*sf, h*sf))

    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    F2B_forward = torch.pow(torch.abs(FB_forward), 2)

    STy = upsample(x, sf=sf)
    FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy, FB_forward, F2B_forward


def classical_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double, positive
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    #x = ndimage.filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def upsample_np(x, sf=3, center=False):
    st = (sf-1)//2 if center else 0
    z = np.zeros((x.shape[0]*sf, x.shape[1]*sf, x.shape[2]))
    z[st::sf, st::sf, ...] = x
    return z

def upsample_torch(x, sf=3, center=False):
    st = (sf-1)//2 if center else 0
    z = torch.zeros((x.shape[0]*sf, x.shape[1]*sf, x.shape[2]))
    z[st::sf, st::sf, ...] = x
    return z


def downsample_np(x, sf=3, center=False):
    st = (sf-1)//2 if center else 0
    return x[st::sf, st::sf, ...]










def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x

def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x







