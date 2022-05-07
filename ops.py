"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from logging import raiseExceptions
import torch
import torchvision.ops.boxes as box_ops

from torch import nn
import torch.nn.functional as F

from torch import Tensor, binary_cross_entropy_with_logits
from typing import List, Tuple

def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
        boxes_1: List[Tensor]
            First set of bounding boxes (M, 4)
        boxes_1: List[Tensor]
            Second set of bounding boxes (M, 4)
        shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        eps: float
            A small constant used for numerical stability

    Returns:
    --------
        Tensor
            Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)

def binary_focal_loss(
    x: Tensor, y: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-6
) -> Tensor:
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Parameters:
    -----------
        x: Tensor[N, K]
            Post-normalisation scores
        y: Tensor[N, K]
            Binary labels
        alpha: float
            Hyper-parameter that balances between postive and negative examples
        gamma: float
            Hyper-paramter suppresses well-classified examples
        reduction: str
            Reduction methods
        eps: float
            A small constant to avoid NaN values from 'PowBackward'

    Returns:
    --------
        loss: Tensor
            Computed loss tensor
    """
    loss = (1 - y - alpha).abs() * ((y-x).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy(
            x, y, reduction='none'
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))

def quality_focal_loss(
    x: Tensor, y: Tensor,
    gamma: float = 2.0,
    reduction='none',
    eps: float = 1e-6,
) -> Tensor:

    scaling_factor = (x-y).abs.pow(gamma)

    loss = F.binary_cross_entropy(x, y, reduction) * scaling_factor
    nums = (y==1).sum()
    if nums == 0:
        loss = loss.sum()
    else:
        loss = loss.sum()/(eps + nums)

    return loss


def OHEM_loss(
    x: Tensor, y: Tensor,
    rate: float = 0.75,
    eps: float = 1e-6,
) -> Tensor:

    loss = F.binary_cross_entropy(x, y, reduction='none')
    #print(loss,loss.size())
    sorted_loss, idx = torch.sort(loss, descending=True)
    sorted_loss_size = sorted_loss.size()
    #print(sorted_loss,sorted_loss_size[0])
    #print(x.size()[0],x.size(0))
    #sys.exit()
    keep = int(min(sorted_loss_size[0],int(x.size()[0] * rate)))
    if keep == 0:
        keep = 1
    elif keep < 0:
        raise Exception('number of keep loss cannot be negative')
    if keep < sorted_loss_size[0]:
        sorted_loss = sorted_loss[:keep]
    
    return sorted_loss.sum() / keep


class GHMC(nn.Module):
    """
    Formula:
        gradient length (g) = torch.sigmoid(x).detach() - target
        garadient (GD(g)) = 1/number of samples in bins
        ghm_loss = aggregation  of (CE(pred, label) / GD(g))

    Args:
        bins: number of gradient buns
        momentum: moving average
        weight: weight of GHMC loss
    """
   
    def __int__(self, bins=10, momentum = 0.1, weight = 1.0):
        super(GHMC,self).__init__
        self.bins = bins
        self.momentum = momentum
        self.weight = weight
        #edges of gradient bin
        self.edges = torch.arange(bins + 1).float() / bins
        #edges = [0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,0.9000, 1.0000]
       
        #register to state.dict
        self.register_buffer('edges',self.edges)
        
        if momentum > 0:
            acc = torch.zeros(bins)
            #register to state.dict
            self.register_buffer('acc', acc)
    
    def forward(self, pred, label, label_weight):
        """
        Args:
            label_weight: 1 if the sample is valid, 0 if it is to suppress

        Return:
            ghm loss for classification
        """
        edges = self.edges
        momentum = self.momentum
        #gradient for the bins
        weights = torch.zeros_like(pred)
        valid_label = label_weight > 0

        #count number of valid samples
        total_valid_label = max(valid_label.float().sum().item(),1)
        #valid bins. Empty gradient bins are 
        valid_bins = 0

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - label)

        #count number of samples in bins
        for i in range(self.bins):
            valid_sample_in_bins = (g >= edges[i]) & (g <= edges[i+1]) & valid_label
            num_samples_in_bins = valid_sample_in_bins.sum().item()
            if num_samples_in_bins > 0:
                if momentum > 0:
                    #gradient 
                    self.acc[i] = momentum * self.acc + (1 - momentum) * num_samples_in_bins
                    weights[valid_sample_in_bins] = total_valid_label / self.acc[i]
                else:
                    weights[valid_sample_in_bins] = total_valid_label / num_samples_in_bins
                
                valid_bins = valid_bins + 1
        
        weights = weights / valid_bins
        
        #binary cross entropy with logits to calculate the final loss
        return self.weight * F.binary_cross_entropy(pred, label, weights, reduction='none') / total_valid_label
