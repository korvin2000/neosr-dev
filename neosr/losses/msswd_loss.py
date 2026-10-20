import torch
from torch import nn
from torch.nn import functional as F

from neosr.utils.registry import LOSS_REGISTRY


class GaussianPyramid(nn.Module):
    def __init__(self, num_scale=5, channels=3):
        super().__init__()
        self.num_scale = num_scale
        self.register_buffer("kernel", self.gauss_kernel(channels), persistent=False)

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([
            [1.0, 4.0, 6.0, 4.0, 1],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ])
        kernel /= 256.0
        # Change kernel to PyTorch: (channels, 1, size, size)
        return kernel.repeat(channels, 1, 1, 1)

    def downsample(self, x):
        # Downsample along image (H,W)
        # Takes every 2 pixels: output (H, W) = input (H/2, W/2)
        return x[:, :, ::2, ::2]

    def conv_gauss(self, img, kernel):
        # Assume input of img is of shape: (N, C, H, W)
        if len(img.shape) != 4:
            raise IndexError(
                "Expected input tensor to be of shape: (N, C, H, W) but got: "
                + str(img.shape)
            )
        img = F.pad(img, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(img, kernel, groups=img.shape[1])

    def forward(self, img):
        kernel = self.kernel.to(img.dtype)
        img_pyramid = [img]
        for _n in range(self.num_scale - 1):
            img = self.conv_gauss(img, kernel)
            img = self.downsample(img)
            img_pyramid.append(img)
        return img_pyramid


# linear-rgb to xyz, assumes d65
# https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
RGB2XYZ = (
    (10135552 / 24577794, 8788810 / 24577794, 4435075 / 24577794),
    (2613072 / 12288897, 8788810 / 12288897, 887015 / 12288897),
    (1425312 / 73733382, 8788810 / 73733382, 70074185 / 73733382),
)
# inverse D65 reference illuminant
INV_ILLUMINANT = (1.052156925, 1.000000000, 0.918357670)


def color_space_transform(input_color, rgb2xyz, inv_illuminant):
    """
    Transforms inputs between different color spaces
    :param input_color: tensor of colors to transform (with NxCxHxW layout)
    :param rgb2xyz: (3, 3) RGB2XYZ tensor
    :param inv_illuminant: (3,) INV_ILLUMINANT tensor
    :return: transformed tensor (with NxCxHxW layout)
    """
    # srgb to linear-rgb
    limit = 0.04045
    transformed_color = torch.where(
        input_color > limit,
        torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
        input_color / 12.92,
    )  # clamp to stabilize training

    # linear-rgb to xyz, then xyz to cie lab
    transformed_color = torch.einsum(
        "ij,njhw->nihw", rgb2xyz.to(transformed_color.dtype), transformed_color
    )
    transformed_color = transformed_color * inv_illuminant.view(1, 3, 1, 1)
    delta = 6 / 29
    delta_square = delta * delta
    delta_cube = delta * delta_square
    factor = 1 / (3 * delta_square)

    clamped_term = torch.pow(
        torch.clamp(transformed_color, min=delta_cube), 1.0 / 3.0
    ).to(dtype=transformed_color.dtype)
    div = (factor * transformed_color + (4 / 29)).to(dtype=transformed_color.dtype)
    transformed_color = torch.where(
        transformed_color > delta_cube, clamped_term, div
    )  # clamp to stabilize training

    L = 116 * transformed_color[:, 1:2, :, :] - 16
    a = 500 * (transformed_color[:, 0:1, :, :] - transformed_color[:, 1:2, :, :])
    b = 200 * (transformed_color[:, 1:2, :, :] - transformed_color[:, 2:3, :, :])

    return torch.cat((L, a, b), 1)


@LOSS_REGISTRY.register()
class msswd_loss(nn.Module):
    """
    Adapted from: https://github.com/real-hjq/MS-SWD
    """

    def __init__(
        self, num_scale=3, num_proj=24, patch_size=11, stride=1, c=3, loss_weight=1.0
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_scale = num_scale
        self.num_proj = num_proj
        self.patch_size = patch_size
        self.stride = stride
        self.c = c
        self.gaussian_pyramid = GaussianPyramid(num_scale, c)
        self.register_buffer("rgb2xyz", torch.tensor(RGB2XYZ), persistent=False)
        self.register_buffer(
            "inv_illuminant", torch.tensor(INV_ILLUMINANT), persistent=False
        )

    def sample_projections(self, x):
        # Sample random normalized projections, on x's device and dtype
        rand = torch.randn(
            self.num_proj, self.c * self.patch_size**2, device=x.device, dtype=x.dtype
        )
        rand = rand / torch.linalg.vector_norm(
            rand, dim=1, keepdim=True
        )  # normalize to unit directions
        return rand.reshape(self.num_proj, self.c, self.patch_size, self.patch_size)

    def forward_once(self, x, y):
        rand = self.sample_projections(x)
        # Project patches
        pad_num = self.patch_size // 2
        x = F.pad(x, pad=(pad_num, pad_num, pad_num, pad_num), mode="reflect")
        y = F.pad(y, pad=(pad_num, pad_num, pad_num, pad_num), mode="reflect")
        projx = F.conv2d(x, rand, stride=self.stride).reshape(
            x.shape[0], self.num_proj, -1
        )
        projy = F.conv2d(y, rand, stride=self.stride).reshape(
            y.shape[0], self.num_proj, -1
        )
        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=2)
        projy, _ = torch.sort(projy, dim=2)

        swd = torch.abs(projx - projy)
        return torch.mean(swd, dim=[1, 2])

    def forward(self, x, y):
        ms_swd = 0.0
        # Build Gaussian pyramids
        x_pyramid = self.gaussian_pyramid(x)
        y_pyramid = self.gaussian_pyramid(y)
        for n in range(self.num_scale):
            # Image preprocessing
            x_single = color_space_transform(
                x_pyramid[n], self.rgb2xyz, self.inv_illuminant
            )
            y_single = color_space_transform(
                y_pyramid[n], self.rgb2xyz, self.inv_illuminant
            )
            ms_swd = ms_swd + self.forward_once(x_single, y_single)

        ms_swd = ms_swd / self.num_scale
        # decrease magnitude to balance with other losses
        ms_swd = ms_swd.mean() * 0.1

        return ms_swd * self.loss_weight
