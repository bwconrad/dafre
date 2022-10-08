import torch
import torch.nn.functional as F


class SoftTargetCrossEntropy(torch.nn.Module):
    """Cross Entropy w/ smoothing or soft targets
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class SoftBalancedSoftmax(torch.nn.Module):
    """Balanced Softmax w/ soft targets
    Adapted from: https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py
    """

    def __init__(self, samples_per_class: torch.Tensor):
        super().__init__()
        self.samples_per_class = samples_per_class

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        spc = self.samples_per_class.type_as(x)
        spc = spc.unsqueeze(0).expand(x.shape[0], -1)
        x = x + spc.log()
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


if __name__ == "__main__":
    f = torch.load("samples_per_class.pkl")
    print(f.size())
    # crit = SoftBalancedSoftmax(f)
    crit = SoftTargetCrossEntropy()
    x = torch.rand((16, 3263))
    y = torch.rand((16, 3263))
    print(crit(x, y))
