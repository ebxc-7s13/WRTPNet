import torch
import torch.nn as nn
import torch.nn.functional as F

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2; x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]; x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]; x4 = x02[:, :, :, 1::2]
    ll = x1 + x2 + x3 + x4
    hl = -x1 - x2 + x3 + x4; lh = -x1 + x2 - x3 + x4; hh = x1 - x2 - x3 + x4
    return ll, torch.cat((hl, lh, hh), 1)

def iwt_init(ll, highs):
    hl, lh, hh = torch.chunk(highs, 3, dim=1)
    x1 = (ll - hl - lh + hh) / 2.0
    x2 = (ll - hl + lh - hh) / 2.0
    x3 = (ll + hl - lh - hh) / 2.0
    x4 = (ll + hl + lh + hh) / 2.0
    b,c,h,w = ll.shape
    out = torch.zeros((b,c,h*2,w*2), device=ll.device, dtype=ll.dtype)
    out[:,:,0::2,0::2] = x1; out[:,:,1::2,0::2] = x2
    out[:,:,0::2,1::2] = x3; out[:,:,1::2,1::2] = x4
    return out

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32, dropout=0.0):
        super().__init__()
        def block(i,o):
            layers = [nn.Conv2d(i,o,3,1,1), nn.LeakyReLU(0.1, inplace=True), 
                      nn.Conv2d(o,o,3,1,1), nn.LeakyReLU(0.1, inplace=True)]
            if dropout > 0: layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)
        self.e1 = block(in_ch, base); self.e2 = block(base, base*2); self.p = nn.MaxPool2d(2)
        self.b = block(base*2, base*4); self.u = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) 
        self.d2 = block(base*6, base*2); self.d1 = block(base*3, base); self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p(e1)); b = self.b(self.p(e2))
        d2 = self.d2(torch.cat([self.u(b), e2], dim=1)); d1 = self.d1(torch.cat([self.u(d2), e1], dim=1))
        return self.out(d1)

class WRTPNet(nn.Module):
    def __init__(self, in_ch=3, base=96, num_blocks=6): 
        super().__init__()
        self.head_ll = nn.Conv2d(in_ch, base, 3, 1, 1); self.head_high = nn.Conv2d(in_ch*3, base, 3, 1, 1)
        self.body_ll = nn.Sequential(*[ResBlock(base) for _ in range(num_blocks)])
        self.body_high = nn.Sequential(*[ResBlock(base) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(nn.Conv2d(base*2, base, 1), nn.ReLU(inplace=True), nn.Conv2d(base, base, 3, 1, 1))
        self.tail = nn.Conv2d(base, in_ch*4, 1)
    def forward(self, x):
        ll, highs = dwt_init(x)
        f_ll = self.head_ll(ll); f_high = self.head_high(highs); f_ll = self.body_ll(f_ll); f_high = self.body_high(f_high)
        f_fuse = self.fusion(torch.cat([f_ll, f_high], dim=1)); res = self.tail(f_fuse)
        out_ll = ll + res[:, 0:3]; out_highs = highs + res[:, 3:]
        return iwt_init(out_ll, out_highs)
