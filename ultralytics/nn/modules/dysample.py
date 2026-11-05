import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Content-aware LP dynamic upsampling."""

    def __init__(self, channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()

        if style.lower() != "lp":
            raise ValueError("This implementation supports only style='lp'.")
        if channels % groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={groups}.")

        self.scale = scale
        self.groups = groups
        offset_channels = 2 * groups * scale * scale

        self.offset = nn.Conv2d(channels, offset_channels, kernel_size=1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.offset.bias)

        self.scope = None
        if dyscope:
            self.scope = nn.Conv2d(channels, offset_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.scope.weight)

        self.register_buffer("initial_offsets", self._initial_offsets(), persistent=False)

    def _initial_offsets(self):
        pos = (torch.arange(self.scale, dtype=torch.float32) + 0.5) / self.scale - 0.5
        dy, dx = torch.meshgrid(pos, pos, indexing="ij")

        offset_x = dx.reshape(-1).repeat(self.groups)
        offset_y = dy.reshape(-1).repeat(self.groups)
        return torch.cat((offset_x, offset_y)).view(1, -1, 1, 1)

    def _make_grid(self, x, offsets):
        b, _, h, w = offsets.shape
        offsets = offsets.reshape(b, 2, -1, h, w)

        xs = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        ys = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        base_grid = torch.stack((grid_x, grid_y), dim=0).permute(0, 2, 1)
        base_grid = base_grid.unsqueeze(0).unsqueeze(2)  # [1, 2, 1, H, W]

        normalizer = torch.tensor([w, h], device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1)
        coords = 2.0 * (base_grid + offsets) / normalizer - 1.0

        coords = F.pixel_shuffle(coords.flatten(1, 2), self.scale)
        coords = coords.reshape(b, 2, self.groups, h * self.scale, w * self.scale)

        return coords.permute(0, 2, 3, 4, 1).reshape(
            b * self.groups, h * self.scale, w * self.scale, 2
        )

    def forward(self, x):
        offsets = self.offset(x)
        if self.scope is None:
            offsets = offsets * 0.25
        else:
            offsets = offsets * self.scope(x).sigmoid() * 0.5

        offsets = offsets + self.initial_offsets.to(offsets)
        grid = self._make_grid(x, offsets)

        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, c // self.groups, h, w)
        y = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return y.reshape(b, c, h * self.scale, w * self.scale)