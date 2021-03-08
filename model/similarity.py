import torch
import torch.nn.functional as F
import torch.nn as nn
def similarity(a, b):
    cosine_eps = 1e-7
    bsize, ch_sz, sp_sz, _ = a.size()[:]

    tmp_a = a
    tmp_a = tmp_a.contiguous().view(bsize, ch_sz, -1)

    tmp_b = b
    tmp_b = tmp_b.contiguous().view(bsize, ch_sz, -1)
    tmp_c = tmp_b.contiguous().permute(0, 2, 1)
    tmp_ab = torch.bmm(tmp_a, tmp_c)
    tmp_b = torch.bmm(tmp_ab, tmp_b)
    tmp_a = torch.bmm(tmp_ab, tmp_a)
    tmp_b = tmp_b.contiguous().permute(0, 2, 1)
    tmp_a_norm = torch.norm(tmp_a, 2, 1, True)
    tmp_b_norm = torch.norm(tmp_b, 2, 2, True)

    similarity = torch.bmm(tmp_b, tmp_a) / (torch.bmm(tmp_b_norm, tmp_a_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
    similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
            similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
    corr_a = similarity.view(bsize, 1, sp_sz, sp_sz)
    corr_a = F.interpolate(corr_a, size=(a.size()[2], a.size()[3]),
                               mode='bilinear', align_corners=True)
    return corr_a

if __name__ == '__main__':
    a = torch.ones(256,60,60)
    b = torch.ones(256, 60, 60)
    prior_mask = similarity(a,b)
