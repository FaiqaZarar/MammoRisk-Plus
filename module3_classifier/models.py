"""Backbones for Module 3.

Both backbones are **ImageNet-pretrained and fine-tuned** (CLAUDE.md). CBIS-DDSM has ~3k full
mammograms, far too few to train a 25M-parameter CNN from scratch. Low-level ImageNet filters (edges,
textures) transfer to mammograms.

- ``resnet50``: the baseline. Widely used, so it makes results comparable with the literature.
- ``efficientnet_b3``: compound-scaled, with fewer parameters (~11M with the 2-class head) and better accuracy per FLOP at
  high resolution.
- ``convnext_tiny``: included only because the defence deck listed it. Whether it stays is open
  (DECISIONS D-11).

The CAM target layer is the **last spatial feature map before global average pooling** for every
backbone. Grad-CAM on that layer has the most class-specific semantics, and using the equivalent
layer keeps the comparison between backbones fair.
"""
from __future__ import annotations

SUPPORTED = ("resnet50", "efficientnet_b3", "convnext_tiny")


def build_model(name: str, pretrained: bool = True, num_classes: int = 2, drop_rate: float = 0.2):
    import timm
    if name not in SUPPORTED:
        raise ValueError(f"Unknown backbone {name!r}; choose from {SUPPORTED}")
    # timm replaces the 1000-class ImageNet head with a freshly initialised num_classes head.
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                             drop_rate=drop_rate)


def cam_target_layer(model, name: str):
    """Last conv block / final feature map, per backbone (timm module names)."""
    if name == "resnet50":
        return model.layer4[-1]
    if name == "efficientnet_b3":
        # conv_head (1×1 conv → 1536 ch) then bn2 (BatchNorm + activation in timm). bn2's output is
        # what gets pooled, so it is the analogue of ResNet's layer4 output.
        return model.bn2
    if name == "convnext_tiny":
        return model.stages[-1]
    raise ValueError(name)


def randomisation_order(model):
    """Blocks from the output (top) down to the input, for the cascading randomisation sanity
    check (Adebayo et al., 2018). Top-level children are expanded one level when they are
    containers (ResNet ``layer4`` → its Bottlenecks; EfficientNet ``blocks`` → its stages), so
    the check reports a similarity per block, not per individual conv layer.
    """
    import torch.nn as nn
    blocks = []
    for name, child in model.named_children():
        if isinstance(child, (nn.Sequential, nn.ModuleList)) and len(child) > 1:
            blocks += [(f"{name}.{i}", sub) for i, sub in enumerate(child)]
        else:
            blocks.append((name, child))
    blocks = [(n, m) for n, m in blocks if any(True for _ in m.parameters())]
    return blocks[::-1]


def reinitialise(module) -> None:
    """Re-draw every parameter of ``module`` (its own init if it has one, else N(0, 0.02))."""
    import torch
    for m in module.modules():
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
        else:
            # Modules with their own parameters but no reset method (e.g. ConvNeXt layer-scale
            # gammas) still get new values, so the block is fully randomised.
            with torch.no_grad():
                for p in m.parameters(recurse=False):
                    p.normal_(0, 0.02)
    # BatchNorm running statistics are buffers, not parameters; reset them too.
    for m in module.modules():
        if hasattr(m, "reset_running_stats"):
            m.reset_running_stats()
