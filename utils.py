from torch import nn


class DownMSELoss(nn.Module):
    def __init__(self, size=6, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.avgpooling = nn.AvgPool2d(kernel_size=size)
        self.tot = size * size
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, dmap, gt_density):
        gt_density = self.avgpooling(gt_density) * self.tot
        if dmap.shape != gt_density.shape:
            raise ValueError(
                f"dmap({dmap.shape}) and gt_density({gt_density.shape}) should have the same shape"
            )
        return self.mse(dmap, gt_density)
