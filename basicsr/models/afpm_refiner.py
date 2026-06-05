from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.arch_util import LayerNorm2d


class _SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class AnisotropicPatchModulator(nn.Module):
    """Patch-wise frequency modulation for AFPM-style refinement.

    The descriptor is built from the current frequency position and local patch
    statistics, then converted to bounded residual gates. It keeps the AFPM
    patch idea but avoids fixed-size buffers and external module structure.
    """

    def __init__(
        self,
        channels: int,
        patch_size: int = 8,
        descriptor_hidden: int | None = None,
        min_hidden: int = 16,
        patch_gate_limit: float = 0.5,
        channel_gate_limit: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if min_hidden <= 0:
            raise ValueError("min_hidden must be positive.")
        if patch_gate_limit <= 0:
            raise ValueError("patch_gate_limit must be positive.")
        if channel_gate_limit <= 0:
            raise ValueError("channel_gate_limit must be positive.")

        hidden = descriptor_hidden
        if hidden is None:
            hidden = max(channels // 4, min_hidden)
        if hidden <= 0:
            raise ValueError("descriptor_hidden must be positive.")

        self.channels = channels
        self.patch_size = int(patch_size)
        self.patch_gate_limit = float(patch_gate_limit)
        self.channel_gate_limit = float(channel_gate_limit)
        self.eps = eps

        self.patch_gate = nn.Sequential(
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.patch_size * self.patch_size),
        )
        self.channel_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self._last_aux: Dict[str, torch.Tensor] = {}

        nn.init.zeros_(self.patch_gate[-1].weight)
        nn.init.zeros_(self.patch_gate[-1].bias)
        nn.init.zeros_(self.channel_gate.weight)
        nn.init.zeros_(self.channel_gate.bias)

    def _pad_to_patch_grid(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        patch_size = self.patch_size
        pad_h = (patch_size - height % patch_size) % patch_size
        pad_w = (patch_size - width % patch_size) % patch_size
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0), height, width

    def _build_position_descriptor(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        patch_size = self.patch_size
        num_h = height // patch_size
        num_w = width // patch_size

        ys = (torch.arange(num_h, device=device, dtype=dtype) + 0.5) * patch_size
        xs = (torch.arange(num_w, device=device, dtype=dtype) + 0.5) * patch_size
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        center_y = torch.as_tensor(height * 0.5, device=device, dtype=dtype)
        center_x = torch.as_tensor(width * 0.5, device=device, dtype=dtype)
        dy = (yy - center_y) / center_y.clamp_min(1.0)
        dx = (xx - center_x) / center_x.clamp_min(1.0)
        radius = torch.sqrt(dx.pow(2) + dy.pow(2) + self.eps)
        radius = radius / radius.max().clamp_min(self.eps)

        return torch.stack([radius, dx, dy], dim=-1).view(-1, 3)

    def _normalize_stat(self, stat: torch.Tensor) -> torch.Tensor:
        mean = stat.mean(dim=1, keepdim=True)
        std = stat.std(dim=1, keepdim=True, unbiased=False)
        return ((stat - mean) / (std + self.eps)).clamp(-3.0, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}.")

        x, orig_h, orig_w = self._pad_to_patch_grid(x)
        _, _, height, width = x.shape
        patch_size = self.patch_size
        patch_area = patch_size * patch_size

        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        patches = patches.view(batch, channels, patch_area, -1)
        patch_count = patches.shape[-1]

        patch_mean = patches.mean(dim=2)
        patch_energy = patches.abs().mean(dim=(1, 2))
        patch_deviation = (patches - patch_mean.unsqueeze(2)).abs().mean(dim=(1, 2))
        patch_energy = self._normalize_stat(patch_energy)
        patch_deviation = self._normalize_stat(patch_deviation)

        pos_desc = self._build_position_descriptor(height, width, x.device, x.dtype)
        pos_desc = pos_desc.unsqueeze(0).expand(batch, -1, -1)
        stat_desc = torch.stack([patch_energy, patch_deviation], dim=-1)
        descriptor = torch.cat([pos_desc, stat_desc], dim=-1)

        point_gate = self.patch_gate(descriptor)
        point_gate = 1.0 + self.patch_gate_limit * torch.tanh(point_gate)
        point_gate = point_gate.transpose(1, 2).view(batch, 1, patch_area, patch_count)

        channel_gate = self.channel_gate(patch_mean)
        channel_gate = 1.0 + self.channel_gate_limit * torch.tanh(channel_gate)
        channel_gate = channel_gate.unsqueeze(2)

        modulated = patches * point_gate * channel_gate
        modulated = modulated.view(batch, channels * patch_area, patch_count)
        out = F.fold(modulated, output_size=(height, width), kernel_size=patch_size, stride=patch_size)
        out = out[:, :, :orig_h, :orig_w]

        self._last_aux = {
            "patch_gate_abs_mean": (point_gate - 1.0).abs().mean().detach(),
            "channel_gate_abs_mean": (channel_gate - 1.0).abs().mean().detach(),
            "patch_energy_abs_mean": patch_energy.abs().mean().detach(),
        }
        return out

    def get_last_aux(self) -> Dict[str, torch.Tensor]:
        return self._last_aux


class AFPMFrequencyRefiner(nn.Module):
    """NAFNet decoder refinement block built around an original AFPM variant."""

    def __init__(
        self,
        channels: int,
        patch_size: int = 8,
        expansion: float = 1.0,
        descriptor_hidden: int | None = None,
        min_hidden: int = 16,
        patch_gate_limit: float = 0.5,
        channel_gate_limit: float = 0.25,
        scale_limit: float = 0.2,
        fft_norm: str | None = "ortho",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if expansion <= 0:
            raise ValueError("expansion must be positive.")
        if scale_limit <= 0:
            raise ValueError("scale_limit must be positive.")
        if fft_norm not in ("backward", "ortho", "forward", None):
            raise ValueError("fft_norm must be one of 'backward', 'ortho', 'forward', or None.")

        hidden_channels = max(int(math.ceil(channels * expansion)), 1)
        self.channels = channels
        self.scale_limit = float(scale_limit)
        self.fft_norm = fft_norm

        self.norm = LayerNorm2d(channels)
        self.freq_norm = LayerNorm2d(channels * 2)
        self.project_in = nn.Conv2d(channels * 2, hidden_channels * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=True,
        )
        self.sg = _SimpleGate()
        self.modulator = AnisotropicPatchModulator(
            hidden_channels,
            patch_size=patch_size,
            descriptor_hidden=descriptor_hidden,
            min_hidden=min_hidden,
            patch_gate_limit=patch_gate_limit,
            channel_gate_limit=channel_gate_limit,
        )
        self.freq_mix = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        self.project_out = nn.Conv2d(hidden_channels, channels * 2, kernel_size=1, bias=True)
        self.spatial_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.residual_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        identity = x
        freq_input = self.norm(x)
        freq = torch.fft.fft2(freq_input, norm=self.fft_norm)
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        freq = torch.cat([freq.real, freq.imag], dim=1)

        freq = self.freq_norm(freq)
        freq = self.project_in(freq)
        freq = self.dwconv(freq)
        freq = self.sg(freq)
        freq = self.modulator(freq)
        freq = self.freq_mix(freq)
        freq = self.project_out(freq)

        real, imag = freq.chunk(2, dim=1)
        freq = torch.complex(real, imag)
        freq = torch.fft.ifftshift(freq, dim=(-2, -1))
        residual = torch.fft.ifft2(freq, norm=self.fft_norm).real
        residual = self.spatial_proj(residual)

        alpha = self.scale_limit * torch.tanh(self.residual_scale)
        out = identity + alpha * residual

        self._last_aux = {
            "alpha_abs_mean": alpha.abs().mean().detach(),
            "residual_abs_mean": residual.abs().mean().detach(),
        }
        self._last_aux.update(self.modulator.get_last_aux())
        if return_aux:
            return out, self._last_aux
        return out

    def get_last_aux(self) -> Dict[str, torch.Tensor]:
        return self._last_aux
