import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils import data

from neosr.config import ConfigError
from neosr.data.data_util import lmdb_keys, read_img
from neosr.data.degradations import (  # type: ignore[attr-defined]
    circular_lowpass_kernel,
    random_mixed_kernels,
)
from neosr.data.file_client import FileClient
from neosr.data.transforms import basic_augment
from neosr.utils import get_root_logger, img2tensor, scandir, tc
from neosr.utils.rng import rng


def otf_crop_size(patch_size: int, scale: int) -> int:
    """GT crop/pad size: at least 512 and the GT patch, a multiple of scale so the
    degraded LQ (GT // scale) keeps the exact scale ratio.
    """
    if not (isinstance(patch_size, int) and isinstance(scale, int)) or min(patch_size, scale) < 1:
        msg = f"otf needs positive integer patch_size and scale, got {patch_size!r}, {scale!r}."
        raise ValueError(msg)
    return math.ceil(max(512, patch_size * scale) / scale) * scale


class otf(data.Dataset):
    """OTF degradation dataset. Originally from Real-ESRGAN.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
    ----
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.

    """

    def __init__(self, opt: dict[str, Any]) -> None:
        super().__init__()
        logger = get_root_logger()
        self.opt = opt
        self.file_client: FileClient | None = None
        self.color = self.opt.get("color", None) != "y"
        self.gt_folder = opt["dataroot_gt"]

        if opt.get("dataroot_lq") is not None:
            msg = f"[datasets.{opt.get('phase')}] 'dataroot_lq' is not supported by otf; use paired or remove it"
            raise ConfigError(msg)
        self.crop_pad_size = otf_crop_size(opt.get("patch_size"), opt.get("scale"))  # type: ignore[arg-type]

        # sets flag for file_client.py
        if self.gt_folder.endswith("lmdb"):
            self.io_backend_opt: dict[str, str] = {"type": "lmdb"}
            lmdb = True
        else:
            self.io_backend_opt: dict[str, str] = {"type": "disk"}
            lmdb = False

        # file client (lmdb io backend)
        if lmdb:
            self.io_backend_opt["db_paths"] = [self.gt_folder]  # type: ignore[assignment]
            self.io_backend_opt["client_keys"] = ["gt"]  # type: ignore[assignment]
            if not self.gt_folder.endswith(".lmdb"):
                msg = f"{tc.red}'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}{tc.end}"
                logger.error(msg)
            self.paths = lmdb_keys(self.gt_folder)
        elif (meta_info := opt.get("meta_info", opt.get("meta_info_file"))) is not None:
            # disk backend with meta_info ('meta_info_file' is the legacy key)
            # Each line in the meta_info describes the relative path to an image
            with Path(meta_info).open(encoding="utf-8") as fin:
                paths = [line.strip().split(" ")[0] for line in fin]
                self.paths = [str(Path(self.gt_folder) / v) for v in paths]
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = sorted(scandir(self.gt_folder, full_path=True))

        # blur settings for the first degradation
        self.kernel_list = opt.get("kernel_list")
        # a list for each kernel probability
        self.kernel_prob = opt.get("kernel_prob")
        self.blur_sigma = opt.get("blur_sigma")
        self.blur_sigmay = opt.get("blur_sigmay", self.blur_sigma)
        # betag used in generalized Gaussian blur kernels
        self.betag_range = opt.get("betag_range")
        # betap used in plateau blur kernels
        self.betap_range = opt.get("betap_range")
        self.sinc_prob = opt.get("sinc_prob")  # the probability for sinc filters

        self.rotation = opt.get('rotation', [-math.pi, math.pi])
        # blur settings for the second degradation
        self.kernel_list2 = opt.get("kernel_list2")
        self.kernel_prob2 = opt.get("kernel_prob2")
        self.blur_sigma2 = opt.get("blur_sigma2")
        self.blur_sigmay2 = opt.get('blur_sigmay2', self.blur_sigma2)
        self.betag_range2 = opt.get("betag_range2")
        self.betap_range2 = opt.get("betap_range2")
        self.sinc_prob2 = opt.get("sinc_prob2")

        # a final sinc filter
        self.final_sinc_prob = opt.get("final_sinc_prob")

        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect

        # Note: this operation must run on cpu, otherwise CUDAPrefetcher will fail
        with torch.device("cpu"):
            self.pulse_tensor = torch.zeros(21, 21, dtype=torch.float32)
        self.pulse_tensor[10, 10] = 1

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"),  # type: ignore[union-attr]
                **self.io_backend_opt,
            )

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_gt = read_img(self.file_client, gt_path, "gt")

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = basic_augment(  # type: ignore[assignment]
            img_gt, self.opt.get("use_hflip", True), self.opt.get("use_rot", True)
        )

        # crop or pad to crop_pad_size
        h, w = img_gt.shape[0:2]  # type: ignore[union-attr]
        crop_pad_size = self.crop_pad_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(
                img_gt,  # type: ignore[reportArgumentType]
                0,
                pad_h,
                0,
                pad_w,
                cv2.BORDER_REFLECT_101,  # type: ignore[call-overload,reportArgumentType]
            )
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:  # type: ignore[union-attr]
            h, w = img_gt.shape[0:2]  # type: ignore[union-attr]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top : top + crop_pad_size, left : left + crop_pad_size, ...]  # type: ignore[index]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if rng.uniform() < self.opt.get("sinc_prob", None):
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = rng.uniform(np.pi / 3, np.pi)
            else:
                omega_c = rng.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigmay, self.rotation,
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if rng.uniform() < self.opt.get("sinc_prob2", None):
            if kernel_size < 13:
                omega_c = rng.uniform(np.pi / 3, np.pi)
            else:
                omega_c = rng.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigmay2, self.rotation,
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if rng.uniform() < self.opt.get("final_sinc_prob", None):
            kernel_size = random.choice(self.kernel_range)
            omega_c = rng.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.from_numpy(sinc_kernel).float()
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True, color=self.color)[0]  # type: ignore[reportArgumentType,arg-type]
        # from_numpy stays on the CPU regardless of the default device
        kernel = torch.from_numpy(kernel).float()
        kernel2 = torch.from_numpy(kernel2).float()

        return {
            "gt": img_gt,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "gt_path": gt_path,
        }
