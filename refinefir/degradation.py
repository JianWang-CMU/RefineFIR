import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import degradations
from DiffJPEG.DiffJPEG import DiffJPEG


def get_degrade_params(h, w, noise_range=(0, 50)):
    kernel_list = ["iso", "aniso"]
    kernel_prob = [0.5, 0.5]
    if h == 256:
        blur_kernel_size = 21
        blur_sigma = [0.1, 5]
        downsample_range = [0.8, 8]
        jpeg_range = [60, 100]
    else:
        blur_kernel_size = 41
        blur_sigma = [0.1, 10]
        downsample_range = [0.8, 32]
        jpeg_range = [60, 100]

    kernel = degradations.random_mixed_kernels(
        kernel_list,
        kernel_prob,
        blur_kernel_size,
        blur_sigma,
        blur_sigma,
        [-math.pi, math.pi],
        noise_range=None,
    )
    kernel = torch.FloatTensor(kernel).view(1, 1, blur_kernel_size, blur_kernel_size)
    kernel = kernel.repeat(3, 1, 1, 1)

    sigma = np.random.uniform(noise_range[0], noise_range[1])
    noise = torch.randn(1, 3, h, w) * sigma / 255
    scale = h / int(h / np.random.uniform(downsample_range[0], downsample_range[1]))
    compression = int(np.random.uniform(jpeg_range[0], jpeg_range[1]))
    return kernel, sigma, noise, scale, compression


def degrade_tensor(x, kernel, sigma, noise, scale, compression):
    device = x.device
    squeezed = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeezed = True

    h, w = x.size()[-2:]
    x = (x + 1) / 2

    kernel = kernel.to(device)
    blur_kernel_size = kernel.size(-1)
    pad_size = (-1 + blur_kernel_size) // 2
    x = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
    x = F.conv2d(x, weight=kernel, groups=3)

    noise = noise.to(device)
    x = (x + noise).clamp(0, 1)
    x = F.interpolate(x, scale_factor=1 / scale, mode="area")
    x = F.interpolate(x, size=(h, w), mode="bilinear")

    jpeg = DiffJPEG(h, w, differentiable=True, quality=compression).to(device)
    x = jpeg(x)

    if squeezed:
        x = x.squeeze(0)
    return 2 * x - 1


def degrade_bgr(image_bgr, size=256, seed=None, noise_range=(0, 50)):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb / 255.0).float().permute(2, 0, 1)
    tensor = tensor * 2 - 1
    params = get_degrade_params(size, size, noise_range=noise_range)
    degraded = degrade_tensor(tensor.unsqueeze(0), *params)[0]
    degraded = ((degraded.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).detach().numpy()
    degraded = (degraded * 255).round().astype(np.uint8)
    return cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)

