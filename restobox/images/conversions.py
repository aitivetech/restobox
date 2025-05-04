import kornia
import torch


def rgb_to_l_ab(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lab = kornia.color.rgb_to_lab(input)

    luminance = lab[0:1, :, :]
    ab = lab[1:3, :, :]

    luminance_norm = luminance / 100
    ab_norm = (ab + 128.0) / 255.0

    return luminance_norm, ab_norm


def l_ab_to_rgb(luminance_norm: torch.Tensor, ab_norm: torch.Tensor) -> torch.Tensor:
    luminance = luminance_norm * 100
    ab = (ab_norm * 255.0) - 128.0
    lab = torch.cat([luminance, ab], dim=1)

    return kornia.color.lab_to_rgb(lab, clip=True)