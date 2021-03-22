import torch
from torch import nn

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        targets,cps = targets
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_cp = cps.expand(n, n).ne(cps.expand(n, n).t())

        #neq and ncp
        mask_cp_id = (mask==0)*mask_cp

        #eq and ncp
        mask_cp_id2=mask*mask_cp

        dist_ap, dist_an = [], []
        for i in range(n):
            try:
                dist_ap.append(dist[i][mask_cp_id2[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask_cp_id[i]].min().unsqueeze(0))
            except:
                pass
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss