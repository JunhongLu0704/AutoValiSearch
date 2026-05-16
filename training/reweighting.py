import torch
import torch.nn.functional as F
from training.schedule import lr_setter

@torch.amp.autocast('cuda', enabled=False)
def weight_learner(cfeatures, pre_features, pre_weight1, args,  global_epoch=0, iter=0, rsw_weight=None):
    N = cfeatures.size(0)
    use_history_memory = pre_features is not None and pre_weight1 is not None
    if rsw_weight is None:
        rsw_weight = torch.ones(N, 1, device=cfeatures.device, dtype=torch.float32)

    weight = torch.ones(N, 1, device=cfeatures.device, dtype=torch.float32, requires_grad=True)
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)
    all_features = cfeatures.detach().float()
    all_rsw_weight = rsw_weight.detach().float()

    if use_history_memory:
        all_features = torch.cat([all_features, pre_features.detach().float()])
        all_rsw_weight = torch.cat([all_rsw_weight, torch.ones_like(pre_weight1).detach().float()])

    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = weight
        if use_history_memory:
            all_weight = torch.cat([all_weight, pre_weight1.detach()])
        optimizerbl.zero_grad()
        normalized_weight = F.softmax(all_weight * all_rsw_weight, dim=0)
        lossb = lossb_expect(all_features, normalized_weight, args.num_f)
        lossp = normalized_weight.pow(args.decay_pow).sum()
        lossg = lossb / args.lambdap + lossp
        lossg.backward()
        optimizerbl.step()

    weight = (weight * all_rsw_weight[:N]).detach()
    softmax_weight = F.softmax(weight, dim=0).squeeze(-1)
    if use_history_memory and softmax_weight.max() < 0.1:
        if global_epoch == 0 and iter < 10:
            pre_features[:N] = (pre_features[:N] * iter + cfeatures) / (iter + 1)
            pre_weight1[:N] = (pre_weight1[:N] * iter + weight) / (iter + 1)
        else:
            pre_features[:N] = pre_features[:N] * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
            pre_weight1[:N] = pre_weight1[:N] * args.presave_ratio + weight * (1 - args.presave_ratio)
    elif softmax_weight.max() >= 0.1:
        softmax_weight = torch.ones_like(softmax_weight) / N

    return softmax_weight.to(cfeatures.dtype), pre_features, pre_weight1


@torch.no_grad()
def random_fourier_features_gpu(x, w=None, b=None, num_f=1, sigma=None, seed=None):
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    # if sigma is None or sigma == 0:
    #     sigma = 1
    # if w is None:
    #     w = 1 / sigma * (torch.randn(size=(num_f, c), device=x.device, dtype=x.dtype))
    #     b = 2 * torch.pi * torch.rand(size=(r, num_f), device=x.device, dtype=x.dtype)
    #     b = b.repeat((n, 1, 1))
    w = torch.randn(size=(num_f, c), device=x.device, dtype=x.dtype)
    b = 2 * torch.pi * torch.rand(size=(r, num_f), device=x.device, dtype=x.dtype)
    b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f, device=x.device, dtype=x.dtype))

    mid = torch.matmul(x, w.t())

    mid = mid + b
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0]
    mid *= torch.pi / 2.0

    Z = Z * torch.cat((torch.cos(mid), torch.sin(mid), x), dim=-1)

    return Z


def lossb_expect(cfeaturec, weight, num_f):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f)  # B, C, N
    cfeaturecs = cfeaturecs.permute(2, 0, 1)  # N, B, C
    w = weight.view(1, -1, 1)  # 1, B, 1
    wx = cfeaturecs * w  # N, B, C
    e = torch.sum(wx, dim=1, keepdim=True)  # N, C, 1
    res = torch.matmul(wx.transpose(1, 2), cfeaturecs) - torch.matmul(e.transpose(1, 2), e)  # N, C, C
    cov_matrix = res * res  # N, C, C
    loss = torch.sum(cov_matrix) - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2))
    return loss / cfeaturecs.size(0) / cfeaturecs.size(2)


