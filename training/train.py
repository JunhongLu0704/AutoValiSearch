import os
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast

from training.reweighting import weight_learner, random_fourier_features_gpu
from utils.meters import AverageMeter, ProgressMeter



def train(train_loader, model, optimizer, epoch, args, tensor_writer=None, save_log=True, scaler=None):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    end = time.time()
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    frsw = LinearRandomWeight(model.fc1.in_features).cuda(args.gpu)

    for i, (images, target, _) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        if args.amp:
            with autocast('cuda'):
                output, cfeatures = model(images)
                cfeatures = cfeatures.float()
        else:
            output, cfeatures = model(images)

        if epoch >= args.epochp and not args.not_stable_learn and cfeatures.size(0) > 16:
            if args.rsw:
                rsw_weight = frsw(cfeatures, args.rsw_min)
            else:
                rsw_weight = cfeatures.new_ones(cfeatures.size(0), 1)
            if args.use_history_memory:
                pre_features, pre_weight1 = model.pre_features, model.pre_weight1
            else:
                pre_features, pre_weight1 = None, None
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i, rsw_weight)

            # if weight1.max() != weight1.min():
            #     # draw_correlation_heatmap(cfeatures, rsw_weight, weight1, epoch, i, get_corr)
            #     draw_umap_feature_diversity(
            #         cfeatures=cfeatures,
            #         frsw=frsw,
            #         args=args,
            #         pre_features=pre_features,
            #         pre_weight1=pre_weight1,
            #         target=target,
            #         epoch=epoch,
            #         iteration=i,
            #         weight_learner=weight_learner,
            #         num_views=4,
            #         n_neighbors=15,
            #         min_dist=0.1,
            #         save_dir='images'
            #     )

            if args.use_history_memory:
                model.pre_features, model.pre_weight1 = pre_features, pre_weight1
        else:
            weight1 = cfeatures.new_ones(cfeatures.size(0), ) / cfeatures.size(0)

        loss = torch.sum(F.cross_entropy(output, target, reduction='none') * weight1)
        # 鍙嶅悜浼犳挱鍜屼紭鍖?
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            method_name = args.image_root.split('/')[-1] + '/' + args.data.split('/')[-1]
            progress.display(i, method_name)
            if save_log:
                progress.write_log(i, args.log_path)

    if save_log:
        progress.write_log(i, args.log_path)


class LinearRandomWeight(nn.Module):
    def __init__(self, dim=256):
        super(LinearRandomWeight, self).__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        self.relu = nn.Sigmoid()
        self.linear_ = nn.Linear(dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.weight_init()

    def weight_init(self):
        torch.nn.init.uniform_(self.linear.weight, a=-1.0, b=1.0)
        torch.nn.init.uniform_(self.linear_.weight, a=-1.0, b=1.0)

    @torch.no_grad()
    def forward(self, x, rsw_min=0.2):
        self.weight_init()
        weight = self.sigmoid(self.linear_(self.relu(self.linear(x)))).clamp(rsw_min, 1.0)
        return weight / weight.mean()

def get_corr(cfeaturec, weight):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=5)  # B, C, N
    cfeaturecs = cfeaturecs.permute(2, 0, 1)  # N, B, C
    w = weight.view(1, -1, 1)  # 1, B, 1
    wx = cfeaturecs * w  # N, B, C
    e = torch.sum(wx, dim=1, keepdim=True)  # N, C, 1
    res = torch.matmul(wx.transpose(1, 2), cfeaturecs) - torch.matmul(e.transpose(1, 2), e)  # N, C, C
    cov_matrix = (res * res).sum(0) / 5
    return cov_matrix.fill_diagonal_(0.)

def draw_umap_feature_diversity(
        cfeatures,
        frsw,
        args,
        pre_features,
        pre_weight1,
        target,
        epoch,
        iteration,
        weight_learner,
        num_views=4,
        n_neighbors=15,
        min_dist=0.1,
        save_dir='images',
        file_prefix='umap_feature_diversity',
        random_state=42
    ):
    """
    UMAP鍙鍖栧瑙嗗浘鐗瑰緛鍒嗗竷

    cfeatures: [N, D] tensor
    frsw: function(cfeatures, rsw_min) -> rsw_weight
    args: args with rsw_min
    pre_features, pre_weight1: auxiliary inputs to weight_learner
    target: [N,] tensor of class labels
    weight_learner: function(cfeatures, pre_features, pre_weight1, args, epoch, iteration, rsw_weight)
    """

    # 纭繚淇濆瓨鐩綍瀛樺湪
    os.makedirs(save_dir, exist_ok=True)

    feats_list = []
    N = cfeatures.size(0)
    for k in range(num_views):
        rsw_weight = frsw(cfeatures, args.rsw_min)
        weight11 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, iteration, rsw_weight)[0]
        feat = cfeatures.detach() * weight11[:, None]  # [N, D]
        feats_list.append(feat.cpu())  # detach+cpu

    feats_all = torch.cat(feats_list, dim=0)  # [N * num_views, D]

    # 鍙栧嚭鐜版渶澶氱殑绫诲埆
    target_np = target.cpu().numpy()
    most_common_class = Counter(target_np).most_common(1)[0][0]
    class_indices = np.where(target_np == most_common_class)[0]

    if len(class_indices) == 0:
        print(f"[Warning] Class {most_common_class} has no samples, skip drawing.")
        return

    # 璁＄畻鍦?feats_all 閲岀殑绱㈠紩
    indices_in_feats_all = []
    for k in range(num_views):
        offset = k * N
        indices_in_feats_all.append(class_indices + offset)
    indices_in_feats_all = np.concatenate(indices_in_feats_all, axis=0)

    feats_selected = feats_all[indices_in_feats_all]

    # 鏍囧噯鍖?
    scaler = StandardScaler()
    feats_selected = scaler.fit_transform(feats_selected.numpy())

    # 鏋勯€犺鍥炬爣绛?
    labels_selected = np.tile(np.arange(num_views), len(class_indices))

    # UMAP闄嶇淮
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embeds = reducer.fit_transform(feats_selected)

    # 缁樺浘
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=embeds[:, 0], y=embeds[:, 1], hue=labels_selected, palette='Set1')
    plt.legend(title='RSW Sample ID')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{file_prefix}_epoch{epoch}_iter{iteration}_class{most_common_class}.png')
    plt.savefig(save_path, dpi=800)
    plt.close()
    print(f"[Saved] UMAP figure saved to {save_path}")

def draw_correlation_heatmap(cfeatures, rsw_weight, weight1, epoch, i, get_corr):
    # 璁＄畻鐩稿叧鎬х煩闃?
    corr_before = get_corr(cfeatures, F.softmax(rsw_weight, dim=0))
    corr_decorr = get_corr(cfeatures, weight1)

    # 闄嶉噰鏍峰钩婊?(姹犲寲)
    corr_before = F.avg_pool2d(corr_before[None, None, :, :], kernel_size=16).squeeze().cpu()
    corr_decorr = F.avg_pool2d(corr_decorr[None, None, :, :], kernel_size=16).squeeze().cpu()

    # 缁熶竴鑹叉爣
    vmin = min(corr_decorr.min(), corr_before.min())
    vmax = max(corr_decorr.max(), corr_before.max())

    # 缁樺浘
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im1 = axs[0].imshow(corr_before, cmap='viridis', vmin=vmin, vmax=vmax)
    im2 = axs[1].imshow(corr_decorr, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im1, ax=axs, orientation='horizontal', shrink=0.8, aspect=40, pad=0.08)
    plt.savefig(f'images/debug_corr_epoch{epoch}_iter{i}.png', dpi=800)
    plt.close()

