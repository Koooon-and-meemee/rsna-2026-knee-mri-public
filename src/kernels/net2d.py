"""Pretrained 2D backbone over slices, aggregated by attention.

Why this replaces the from-scratch 3D CNN. That model was 2.0M parameters
learning everything from 21,148 training series - no prior about edges,
texture, or shape. Four runs of hyperparameter tuning on it bought 0.752 to
0.860, and then resolution made things worse, which is what capacity limits
look like. A backbone pretrained on ImageNet arrives already knowing how to
see; the 4,407 studies then only have to teach it knee pathology.

The shape of the problem drives the design. A study is a stack of ~32 slices
that must produce one 12-way answer, and the finding may be visible in two
slices out of thirty - a meniscal tear is a few millimetres of fissure. So:

**Slices go through the backbone independently**, batched as (B*D, 3, H, W).
Each slice gets its own feature vector. This is what makes ImageNet weights
usable at all: they expect 2D natural-image input, not a 3D volume.

**Slices are combined by ATTENTION, not by mean.** Averaging 32 slice features
dilutes a finding present in two of them by a factor of 16. Attention lets the
model concentrate on the slices that carry evidence, which is the whole reason
the weak findings are weak.

**Grayscale is repeated to 3 channels rather than averaging the stem.**
Averaging the RGB filters into one channel is the usual shortcut and it
discards the colour-opponent structure the first layer learned. Repeating
costs nothing and keeps the pretrained stem intact.

**The backbone sees ImageNet normalisation.** Volumes are stored 0-255 uint8
and normalised per-volume by percentile; feeding that straight in puts the
input off the distribution the weights were trained on.

Submission works offline. Only `pretrained=True` touches the network; the
architecture itself comes from timm's local registry, so the submission kernel
builds `SliceNet(n, pretrained=False)` and loads our trained checkpoint - which
already contains the fine-tuned backbone. No weights dataset needs shipping.
Training kernels are the ones that need `enable_internet`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AttentionPool(nn.Module):
    """Gated attention over the slice axis (Ilse et al., attention-based MIL).

    The gate is what separates "this slice is unusual" from "this slice is
    unusual in the way that matters", which a plain softmax over a linear
    score cannot express.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (B, D, C)
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))   # (B, D, 1)
        a = torch.softmax(a, dim=1)
        return (a * x).sum(1), a.squeeze(-1)   # (B, C), (B, D)


def _load_external_backbone(model, path: str) -> None:
    """Load a non-timm checkpoint into a timm backbone, remapping key names.

    The medically pretrained weights the public solutions use (RadImageNet,
    MedicalNet) are torchvision ResNets saved inside an nn.Sequential, so their
    keys read `backbone.0.weight`, `backbone.4.0.conv1.weight` and so on, while
    timm names the same tensors `conv1.weight`, `layer1.0.conv1.weight`. The
    architectures are identical - 318 tensors either way - only the naming
    differs, and torchvision's Sequential order fixes the mapping.

    Loads strictly after remapping. A silent partial load here would train a
    randomly initialised backbone while the log claimed otherwise, which is the
    same class of failure as the depth flag that existed in argparse and nowhere
    else, so anything unmatched raises.
    """
    import re
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    seq = {"0": "conv1", "1": "bn1", "4": "layer1",
           "5": "layer2", "6": "layer3", "7": "layer4"}
    out, skipped = {}, 0
    for k, v in sd.items():
        m = re.match(r"backbone\.(\d+)\.(.*)", k)
        if m and m.group(1) in seq:
            out[f"{seq[m.group(1)]}.{m.group(2)}"] = v
        elif m:
            skipped += 1          # relu/maxpool/fc carry no parameters
        else:
            out[k] = v            # already timm-named

    missing, unexpected = model.load_state_dict(out, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"{path}: remap left {len(missing)} missing and {len(unexpected)} "
            f"unexpected keys - refusing to train on a partly random backbone.\n"
            f"  missing: {missing[:5]}\n  unexpected: {unexpected[:5]}"
        )
    print(f"loaded external backbone {path}: {len(out)} tensors, {skipped} skipped")


class SliceNet(nn.Module):
    def __init__(self, n_out: int, backbone: str = "convnext_nano.in12k_ft_in1k",
                 pretrained: bool = True, drop: float = 0.1,
                 backbone_weights: str | None = None,
                 neighbour_channels: bool = False,
                 slice_attn: int = 0, slice_gru: int = 0,
                 cond_classes: int = 0, unfreeze_last: int = 0):
        super().__init__()
        import timm
        self.neighbour_channels = neighbour_channels
        self.slice_attn_layers = slice_attn
        self.slice_gru_layers = slice_gru

        # ViT-family checkpoints (DINOv2, DeiT, BEiT, ...) are released with a
        # single "norm" applied to the CLS token (global_pool="token"). Forcing
        # global_pool="avg" makes timm build a separate "fc_norm" layer instead
        # and load_state_dict fails - the checkpoint has "norm.*", the model
        # wants "fc_norm.*". Conv backbones (convnext, efficientnet, ...) have
        # no CLS token and no such ambiguity, so "avg" (spatial mean pool) is
        # correct and unaffected by this branch.
        is_vit = any(f in backbone for f in ("vit_", "deit", "beit", "eva"))
        pool = "token" if is_vit else "avg"
        kwargs = {}
        if is_vit:
            # ViT position embeddings are sized for the pretraining resolution
            # (518 for the DINOv2 checkpoints) and reject any other input by
            # default. dynamic_img_size interpolates them to whatever size
            # actually arrives, which is required since our inputs (256-448
            # depending on the run) are all below that.
            kwargs["dynamic_img_size"] = True
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=pool, **kwargs
        )
        if backbone_weights:
            _load_external_backbone(self.backbone, backbone_weights)
        # Train only the last `unfreeze_last` backbone blocks, 2026-09-19.
        #
        # Every DINOv2 run this project has made (m28dino336 0.8500, m34dinonew
        # 0.8559, m36dinolr 0.8634) fine-tuned the whole backbone at lr/10 and
        # landed far below the convnext members, and the axis was written off as
        # "DINO does not work here". The public 0.941 notebook uses the same
        # family with LR_BACKBONE 8e-6 against LR_HEAD 1e-3 - a ratio of 125:1,
        # not 10:1 - and UNFREEZE_LAST=6. A self-supervised ViT fine-tuned that
        # aggressively is not being adapted, it is being overwritten, so what
        # those three runs measured may be the schedule rather than the backbone.
        if unfreeze_last:
            for prm in self.backbone.parameters():
                prm.requires_grad = False
            # Each family names its stack differently, and the point of this
            # flag is to reopen every family August closed on one schedule, so
            # it has to reach all of them rather than the two tried first.
            blocks = None
            for attr in ("blocks",    # vit, efficientnet
                         "stages",    # convnext, maxxvit/coatnet
                         "layers"):   # swin
                blocks = getattr(self.backbone, attr, None)
                if blocks is not None:
                    break
            if blocks is None and hasattr(self.backbone, "layer1"):
                blocks = [getattr(self.backbone, f"layer{i}")   # resnet
                          for i in (1, 2, 3, 4)
                          if hasattr(self.backbone, f"layer{i}")]
            if blocks is None:
                raise ValueError(f"--unfreeze-last: {backbone} exposes none of "
                                 f".blocks / .stages / .layers / .layer1-4")
            for blk in list(blocks)[-unfreeze_last:]:
                for prm in blk.parameters():
                    prm.requires_grad = True
            # The final norm is what the pooled feature is read through; leaving
            # it frozen keeps the head reading statistics fitted to ImageNet.
            for nm in ("norm", "fc_norm", "norm_pre"):
                sub = getattr(self.backbone, nm, None)
                if isinstance(sub, nn.Module):
                    for prm in sub.parameters():
                        prm.requires_grad = True
            n_tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in self.backbone.parameters())
            print(f"unfreeze_last {unfreeze_last}: {n_tr/1e6:.1f}M of "
                  f"{n_all/1e6:.1f}M backbone params trainable "
                  f"({len(list(blocks))} blocks total)")

        dim = self.backbone.num_features
        # Self-attention across slices, before pooling.
        #
        # AttentionPool scores each slice and takes a weighted sum: the slices
        # never see each other. It answers "how much does this slice matter"
        # and cannot answer "does this slice matter GIVEN what the next one
        # shows", which is how a radiologist reads a stack - a signal is
        # confirmed by its neighbours or dismissed as an artefact.
        #
        # Costs almost nothing: the sequence is D=32 tokens of `dim` features,
        # against a backbone that has just run 32 times over 224x224 images.
        self.slice_attn = None
        if slice_attn:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=8, dim_feedforward=dim * 2,
                dropout=drop, batch_first=True, norm_first=True)
            self.slice_attn = nn.TransformerEncoder(layer, num_layers=slice_attn)
        # A recurrent aggregator over the same sequence, 2026-09-04.
        #
        # Every RSNA competition since 2019 has been won with one - hemorrhage
        # 2019 (the write-up is titled "Sequential model wins"), cervical spine
        # 2022, abdominal trauma 2023, lumbar spine 2024 - and the 2023 team
        # reports GRU beating self-attention in their own ablations. This project
        # tried self-attention here (m54attn, pv-hold 0.8784 against 0.8830) and
        # closed the axis; it never tried recurrence, which is a different
        # inductive bias over the same tokens. Self-attention is permutation
        # invariant unless told otherwise and there is no positional encoding
        # above, so m54attn could not use slice ORDER at all. A GRU cannot help
        # but use it, and slice order is anatomy: the sequence runs medial to
        # lateral through the joint.
        #
        # Bidirectional at half width, so the output is `dim` and the pool below
        # is unchanged. The cost is negligible against 32 backbone passes.
        self.slice_gru = None
        if slice_gru:
            self.slice_gru = nn.GRU(dim, dim // 2, num_layers=slice_gru,
                                    batch_first=True, bidirectional=True,
                                    dropout=drop if slice_gru > 1 else 0.0)
        self.pool = AttentionPool(dim)
        # Side information about the series itself, 2026-09-19.
        #
        # A study prediction is the mean over its series, and the model is told
        # nothing about which series it is looking at. Measured on m38seed1 with
        # the averaging count held fixed, a fluid-sensitive/fat-suppressed
        # series is worth +0.0118 macro more than a random series of the same
        # study and a non-FS one is worth -0.0186 - a 0.03 spread in how much a
        # view is worth, which the equal-weight mean spends nothing on knowing.
        #
        # Re-weighting the mean by that flag was tried first and returns
        # +0.0006 held out, i.e. nothing: down-weighting non-FS removes the
        # diversity that pays for its lower solo quality. This is the other
        # place to put the same fact - inside the model, where it can shift a
        # prediction per finding and per image rather than by one scalar.
        #
        # An additive embedding on the pooled feature, zero-initialised, so the
        # first forward pass is numerically identical to the unconditioned
        # model and any difference is something training chose to add.
        self.cond = None
        if cond_classes:
            self.cond = nn.Embedding(cond_classes, dim)
            nn.init.zeros_(self.cond.weight)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(dim, n_out)
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def features(self, x):                     # (B, 1, D, H, W) -> (B, C)
        b, _, d, h, w = x.shape
        if self.neighbour_channels:
            # 2.5D. The backbone's three input channels were carrying the same
            # grayscale slice three times - two of them at zero information.
            # Filling them with (i-1, i, i+1) costs nothing: same tensor shape,
            # same FLOPs, same pretrained stem, and the RGB stem is built for
            # correlated channels, which adjacent slices are.
            #
            # It also explains why depth 48 bought nothing. More slices did not
            # help because every slice was still processed alone; the model had
            # no path to through-plane information at all. This gives it one
            # without adding a single slice.
            v = x[:, 0]                                       # (B, D, H, W)
            prev = torch.cat([v[:, :1], v[:, :-1]], dim=1)    # edges repeat
            nxt = torch.cat([v[:, 1:], v[:, -1:]], dim=1)
            s = torch.stack([prev, v, nxt], dim=2).reshape(b * d, 3, h, w)
        else:
            s = x.permute(0, 2, 1, 3, 4).reshape(b * d, 1, h, w)
            s = s.expand(-1, 3, -1, -1)        # grayscale -> 3ch, stem intact
        s = (s - self.mean) / self.std         # ImageNet statistics
        f = self.backbone(s).reshape(b, d, -1)
        if self.slice_attn is not None:
            f = self.slice_attn(f)
        if self.slice_gru is not None:
            # Residual, so an untrained GRU cannot destroy the backbone's
            # features on the first epochs the way a fresh transformer block
            # can - the head still sees the slice features it would have seen.
            f = f + self.slice_gru(f)[0]
        return self.pool(f)[0]

    def forward(self, x, cond=None):           # x: (B, 1, D, H, W) in 0..1
        f = self.features(x)
        if self.cond is not None and cond is not None:
            f = f + self.cond(cond)
        return self.head(self.drop(f))


class StudyNet(nn.Module):
    """One prediction per STUDY, from all of its series at once.

    The label is the reason this exists. Findings are recorded per study, but
    training so far attached them to every series in that study - so an axial
    series was told a meniscal tear was present when the tear is only legible
    on sagittal images. The model cannot tell which series the label came from,
    and the loss-minimising response is to learn whatever looks the same in all
    of them: coarse study-level appearance rather than the lesion.

    That is exactly what the plane probe found. Sagittal, coronal and axial each
    score 0.857/0.857/0.841 alone, and medial meniscus reads 0.896 sagittal
    against 0.847 axial - nearly plane-invariant, which a model that had located
    a few millimetres of fissure could not be.

    So attention runs twice, over slices within a series and then over series
    within a study, and the loss is applied where the label actually lives. The
    second attention is what lets the model say "this finding is evidenced by
    this series" instead of being forced to agree with all of them.
    """

    def __init__(self, n_out: int, backbone: str = "convnext_nano.in12k_ft_in1k",
                 pretrained: bool = True, drop: float = 0.1):
        super().__init__()
        self.slice_net = SliceNet(n_out, backbone, pretrained, drop)
        dim = self.slice_net.backbone.num_features
        self.series_pool = AttentionPool(dim)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(dim, n_out)

    def forward(self, x, n_series, return_attn: bool = False):
        """x: (S_total, 1, D, H, W) - every series of every study, concatenated.
        n_series: list of per-study series counts summing to S_total.

        Studies hold different numbers of series (1 to 14, median 5), so they
        are packed flat and split here rather than padded to the maximum. Padding
        to 14 would put up to 13x of wasted slices through the backbone, which
        is the entire cost of the model.
        """
        f = self.slice_net.features(x)                     # (S_total, C)
        out, attn = [], []
        for chunk in torch.split(f, n_series):
            pooled, a = self.series_pool(chunk.unsqueeze(0))
            out.append(pooled)
            attn.append(a.squeeze(0))
        pooled = torch.cat(out, 0)                         # (B, C)
        logits = self.head(self.drop(pooled))
        return (logits, attn) if return_attn else logits
