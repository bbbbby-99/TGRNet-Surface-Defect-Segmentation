
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable



class Tripletencoder(nn.Module):
    def __init__(self,FPN=True):
        super(Tripletencoder, self).__init__()
        self.in_planes = 64
        # Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.tri = nn.Parameter(torch.zeros(1))
        self.FPN = FPN
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def forward(self, c2,c3,c4):
        # Top-down
        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer2(c2))
        # Smooth
        if self.FPN:
            p2 = self.smooth3(p2)
        else:
            p2 = self.smooth3(p2) * 0 + c2
        return p2


def Tripletencoder101():
    return Tripletencoder()

