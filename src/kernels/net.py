"""The image model, kept in one file because two kernels must agree on it.

Training saves a state_dict; the submission kernel rebuilds the module and
loads it. If the two definitions drift by even a channel count, load_state_dict
fails loudly - or worse, succeeds against a stale checkpoint that no longer
means what the training run meant. So neither kernel defines this itself; both
embed this file verbatim.
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_out, width):
        super().__init__()
        c, chans, blocks = width, 1, []
        for i in range(4):
            blocks += [
                nn.Conv3d(chans, c, 3, padding=1, bias=False),
                nn.BatchNorm3d(c), nn.SiLU(inplace=True),
                nn.Conv3d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm3d(c), nn.SiLU(inplace=True),
                # depth is only 32, so the first pool spares it
                nn.MaxPool3d((1, 2, 2) if i == 0 else 2),
            ]
            chans, c = c, min(c * 2, 192)
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(chans, n_out)

    def forward(self, x):
        h = self.body(x)
        return self.head(F.adaptive_avg_pool3d(h, 1).flatten(1))
