"""LPIPS v0.1 (Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual
Metric", CVPR 2018), rewritten from the reference (richzhang/PerceptualSimilarity): ImageNet
AlexNet/VGG16 features at five ReLUs, unit-normalised over channels, squared differences weighted by
the official v0.1 linear layers (loaded strictly), spatial mean, summed. Lower is better."""

import torch
from torch import Tensor, nn
from torchvision.models import AlexNet_Weights, VGG16_Weights, alexnet, vgg16

# net -> (torchvision builder, ImageNet weights, feature index after each tapped ReLU, its channels)
NETS = {
    "alex": (
        alexnet,
        AlexNet_Weights.IMAGENET1K_V1,
        (2, 5, 8, 10, 12),
        (64, 192, 384, 256, 256),
    ),
    "vgg": (
        vgg16,
        VGG16_Weights.IMAGENET1K_V1,
        (4, 9, 16, 23, 30),
        (64, 128, 256, 512, 512),
    ),
}
# the v0.1 linear layers, mirrored by pyiqa (as the TOPIQ weights)
URL = "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/LPIPS_v0.1_{}.pth"
FILES = {"alex": "alex-df73285e", "vgg": "vgg-a78928a0"}


class lpips(nn.Module):
    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        build, weights, ends, chns = NETS[net]
        features = build(weights=weights).features
        self.slices = nn.ModuleList(
            features[a:b] for a, b in zip((0, *ends[:-1]), ends, strict=True)
        )
        self.lins = nn.ModuleList(nn.Conv2d(c, 1, 1, bias=False) for c in chns)
        sd = torch.hub.load_state_dict_from_url(
            URL.format(FILES[net]), map_location="cpu", weights_only=True
        )
        # official keys lin<k>.model.1.weight (model.0 is the training-time dropout)
        self.lins.load_state_dict(
            {k.removeprefix("lin").replace(".model.1.", "."): v for k, v in sd.items()},
            strict=True,
        )
        self.register_buffer(
            "shift", torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "scale", torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1), False
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """x, y: RGB in [0, 1], one batch through the backbone; the distance per image pair."""
        h = (torch.cat([x, y]) * 2 - 1 - self.shift) / self.scale
        dist = 0
        for features, lin in zip(self.slices, self.lins, strict=True):
            h = features(h)
            dist = dist + _layer(h, lin)  # temporaries freed before the next slice
        return dist.squeeze()  # type: ignore[union-attr]


def _layer(h: Tensor, lin: nn.Module) -> Tensor:
    f = h / (h.square().sum(1, keepdim=True).sqrt() + 1e-10)
    fx, fy = f.chunk(2)
    return lin((fx - fy).square()).mean([2, 3])
