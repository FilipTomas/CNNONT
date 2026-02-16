import torch
import torch.nn as nn
import torch.nn.functional as F

class Film(nn.Module):
    def __init__(self, n_global, n_chanels, hidden = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_global, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_chanels * 2)
        )

    def forward(self, x, g):
        gamma_beta = self.net(g)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        out = x * (gamma) + beta
        return out


class SepConv1d(nn.Module):
    def __init__(self, hidden: int, k: int, d: int):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=k,
            dilation=d,
            padding=((k - 1) * d) // 2,
            groups=hidden,
            bias=False,
        )
        self.pw = nn.Conv1d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        return self.pw(self.dw(x))
# %%
## Define model
class CNNONT(nn.Module):
    def __init__(self, n_global, hidden = 64, k = 7, film_hidden = 64, dilation_levels = 7, dilation_step = 1, n_full_convs = 3, n_classes = 3):
        super().__init__()

        #dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        dilations_list = [1, 2, 4, 8, 16, 32, 64]
        dilations = dilations_list[:dilation_levels:dilation_step]
       # self.stem = nn.Conv1d(1, hidden, kernel_size = 1, bias=False)
        self.stem = nn.Conv1d(1, hidden, kernel_size = 1)
        # self.convs = nn.ModuleList([
        #     nn.Conv1d(
        #         in_channels=hidden,
        #         out_channels=hidden,
        #         kernel_size=k,
        #         dilation=d,
        #         padding=((k - 1) * d) // 2,
        #         bias=False,          # match your current default; set False if you want
        #     ) if i < n_full_convs else SepConv1d(hidden, k, d)
        #     for i, d in enumerate(dilations)
        # ])
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden,
                out_channels=hidden,
                kernel_size=k,
                dilation=d,
                padding=((k - 1) * d) // 2,
               # bias=False,          # match your current default; set False if you want
            )
            for d in dilations
        ])
        
        self.films = nn.ModuleList([
            Film(n_global, hidden, film_hidden)
            for _ in dilations
        ])

        self.norms = nn.ModuleList([
            nn.GroupNorm(8, hidden)
            for _ in dilations
        ])

       # self.head = nn.Conv1d(hidden, n_classes, kernel_size=1, bias=False)
        self.head = nn.Conv1d(hidden, n_classes, kernel_size=1)

    def forward(self, x, g):
        x = self.stem(x)
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            x = conv(x)
            x = norm(x)
            x = film(x, g)
            x = F.relu(x)
        return self.head(x)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.register_buffer("alpha", alpha.detach().float())
            else:
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        logits:  [B, C, L]
        targets: [B, L]
        """
        B, C, L = logits.shape

        logits = logits.permute(0, 2, 1).reshape(-1, C)   # [B*L, C]
        targets = targets.reshape(-1)                     # [B*L]

        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid]

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            at = self.alpha[targets]
        else:
            at = 1.0

        loss = -at * (1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss