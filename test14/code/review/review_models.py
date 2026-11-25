"""Spatial and pointer candidates for the design review (prototypes, not repository code)."""
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vos_memory_inspector.models._spatial import SpatialMemoryTranslator, SpatialTransformerConfig


def ridge_conv(w, b):
    layer = nn.Conv2d(64, 64, 1)
    with torch.no_grad():
        layer.weight.copy_(w.T.float()[:, :, None, None])
        layer.bias.copy_(b.float())
    return layer


def identity_conv():
    layer = nn.Conv2d(64, 64, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(64)[:, :, None, None])
        layer.bias.zero_()
    return layer


class Residual(nn.Module):
    """Position-wise residual MLP, the repository's residual_mlp spatial head."""

    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Conv2d(64, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, 64, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DepthwiseBlock(nn.Module):
    """Depthwise neighbourhood mixing followed by a position-wise MLP."""

    def __init__(self, hidden=128, kernel=3, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(64, 64, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation, groups=64)
        self.fc1 = nn.Conv2d(64, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, 64, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.depthwise(x))))


class Conv3Block(nn.Module):
    """Full 3x3 mixing in the first layer, then a position-wise output layer."""

    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Conv2d(64, hidden, 3, padding=1)
        self.fc2 = nn.Conv2d(hidden, 64, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SpatialCandidate(nn.Module):
    """skip(x) + sum of residual blocks; blocks read the skip output or the input."""

    def __init__(self, skip, blocks, sequential=False):
        super().__init__()
        self.skip = skip
        self.blocks = nn.ModuleList(blocks)
        self.sequential = sequential

    def forward(self, x):
        output = self.skip(x)
        if self.sequential:
            feature = x
            total = 0
            for block in self.blocks:
                delta = block(feature)
                feature = feature + delta
                total = total + delta
            return output + total
        for block in self.blocks:
            output = output + block(x)
        return output


class Conv3Linear(nn.Module):
    """Linear 3x3 mapping that starts exactly at the 1x1 ridge map."""

    def __init__(self, w, b):
        super().__init__()
        self.layer = nn.Conv2d(64, 64, 3, padding=1)
        with torch.no_grad():
            self.layer.weight.zero_()
            self.layer.weight[:, :, 1, 1] = w.T.float()
            self.layer.bias.copy_(b.float())

    def forward(self, x):
        return self.layer(x)


class ContextSpatial(nn.Module):
    """Repository spatial backbone with its skip optionally set from ridge."""

    def __init__(self, config, ridge=None):
        super().__init__()
        self.spatial = SpatialMemoryTranslator(config)
        if ridge is not None and self.spatial.linear_skip is not None:
            with torch.no_grad():
                self.spatial.linear_skip.weight.copy_(ridge[0].T.float())
                self.spatial.linear_skip.bias.copy_(ridge[1].float())

    def forward(self, x):
        return self.spatial(x)


class RidgeTransformer(nn.Module):
    """Base transformer structure, with the identity skip replaced by frozen-start ridge."""

    def __init__(self, w, b):
        super().__init__()
        self.skip = ridge_conv(w, b)
        self.spatial = SpatialMemoryTranslator(SpatialTransformerConfig())

    def forward(self, x):
        delta = self.spatial(x) - x
        return self.skip(x) + delta


class ContextPlus(nn.Module):
    """Ridge skip + depthwise local block + gated global context."""

    def __init__(self, w, b, hidden=128):
        super().__init__()
        self.skip = ridge_conv(w, b)
        self.local = DepthwiseBlock(hidden)
        config = SpatialTransformerConfig(local_fusion='none', residual_scale_init=0.0)
        self.context = SpatialMemoryTranslator(config)

    def forward(self, x):
        return self.skip(x) + self.local(x) + (self.context(x) - x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, width, kernel=7):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, kernel, padding=kernel // 2, groups=width)
        self.norm = nn.GroupNorm(1, width)
        self.fc1 = nn.Conv2d(width, width * 3, 1)
        self.fc2 = nn.Conv2d(width * 3, width, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(self.depthwise(x)))))


class Ceiling(nn.Module):
    """Large two-scale network used only to estimate how much a bigger model can gain."""

    def __init__(self, w, b, width=192, depth=4):
        super().__init__()
        self.skip = ridge_conv(w, b)
        self.embed = nn.Conv2d(64, width, 1)
        self.high = nn.Sequential(*[ConvNeXtBlock(width) for _ in range(depth)])
        self.down = nn.Conv2d(width, width * 2, 4, stride=4)
        self.low = nn.Sequential(*[ConvNeXtBlock(width * 2) for _ in range(depth)])
        self.up = nn.Conv2d(width * 2, width, 1)
        self.high2 = nn.Sequential(*[ConvNeXtBlock(width) for _ in range(depth)])
        self.out = nn.Conv2d(width, 64, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.high(self.embed(x))
        l = self.up(self.low(self.down(h)))
        h = h + F.interpolate(l, size=h.shape[-2:], mode='bilinear', align_corners=False)
        return self.skip(x) + self.out(self.high2(h))


def make_spatial(name, ridge):
    w, b = ridge['spatial_w'], ridge['spatial_b']
    skip = lambda: ridge_conv(w, b)
    if name == 'affine':
        return SpatialCandidate(skip(), [])
    if name == 'mlp':
        return SpatialCandidate(skip(), [Residual(128)])
    if name == 'mlp_identity':
        return SpatialCandidate(identity_conv(), [Residual(128)])
    if name == 'mlp512':
        return SpatialCandidate(skip(), [Residual(512)])
    if name == 'mlp2':
        return SpatialCandidate(skip(), [Residual(128), Residual(128)], sequential=True)
    if name == 'conv3lin':
        return Conv3Linear(w, b)
    if name == 'ds':
        return SpatialCandidate(skip(), [DepthwiseBlock(128)])
    if name == 'ds2':
        return SpatialCandidate(skip(), [DepthwiseBlock(128, dilation=1), DepthwiseBlock(128, dilation=2)], sequential=True)
    if name == 'ds3':
        return SpatialCandidate(skip(), [DepthwiseBlock(128, dilation=d) for d in (1, 2, 4)], sequential=True)
    if name == 'conv3mlp':
        return SpatialCandidate(skip(), [Conv3Block(128)])
    if name == 'transformer':
        return ContextSpatial(SpatialTransformerConfig())
    if name == 'transformer_ridge':
        return RidgeTransformer(w, b)
    if name == 'mlp_context':
        config = SpatialTransformerConfig(local_fusion='additive_mlp', skip_mode='affine')
        return ContextSpatial(config, (w, b))
    if name == 'ds_context':
        return ContextPlus(w, b)
    if name == 'ceiling':
        return Ceiling(w, b)
    raise ValueError(name)


def load_run(path, device):
    """Spatial candidate plus the analysis ridge pointer, as a (x, p, cond) -> (x', p') function."""
    saved = torch.load(Path(path) / 'model.pt', map_location='cpu', weights_only=False)
    ridge = torch.load(Path(path).parents[1] / 'ridge/ridge.pt', map_location='cpu', weights_only=True)
    model = make_spatial(saved['name'], ridge)
    model.load_state_dict(saved['state'])
    model = model.to(device).float().eval()
    w, b = ridge['pointer_w'].to(device), ridge['pointer_b'].to(device)

    def run(x, p, cond):
        return model(x), (p.double() @ w + b).float()
    return run


class PointerMLP(nn.Module):
    def __init__(self, w, b, hidden=512, depth=1, norm=False, freeze_skip=False, context=0):
        super().__init__()
        self.skip = nn.Linear(256, 256)
        with torch.no_grad():
            self.skip.weight.copy_(w.T.float())
            self.skip.bias.copy_(b.float())
        self.skip.requires_grad_(not freeze_skip)
        layers = []
        size = 256 + context
        for i in range(depth):
            layers += [nn.Linear(size if i == 0 else hidden, hidden), nn.GELU()]
            if norm:
                layers.append(nn.LayerNorm(hidden))
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 256)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, p, extra=None):
        z = p if extra is None else torch.cat((p, extra), -1)
        return self.skip(p) + self.out(self.body(z))
