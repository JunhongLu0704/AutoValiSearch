import time
from collections import defaultdict
import os
import torch.utils.data.distributed
from torch.amp import autocast
from utils.matrix import accuracy
from utils.meters import AverageMeter, ProgressMeter


@torch.no_grad()
def validate(val_loader,
             model,
             epoch: int = 0,
             test: bool = True,
             args=None,
             tensor_writer=None,
             separate_domain: bool = False,
             all_domains=None):
    """
    流式统计版验证函数。
    - 逐 batch 计算 top-k 并累加到 AverageMeter，避免一次性 cat。
    - 若 `separate_domain=True`，在 **每个 batch** 内按域拆分再统计。
    """

    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    if separate_domain:
        domain_top1 = defaultdict(lambda: AverageMeter('Acc@1', ':6.2f'))
        domain_top5 = defaultdict(lambda: AverageMeter('Acc@5', ':6.2f'))

    progress = ProgressMeter(len(val_loader),
                             [batch_time, data_time, top1, top5],
                             prefix='Test: ' if test else 'Val: ')

    model.eval()
    end = time.time()

    for i, (images, target, domains) in enumerate(val_loader):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            domains = domains.cuda(args.gpu, non_blocking=True)
        data_time.update(time.time() - end)

        if args.amp:
            with autocast('cuda'):
                output, _ = model(images)
        else:
            output, _ = model(images)

        batch_size = target.size(0)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # tensor shape (1,)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        if separate_domain:
            # 该 batch 出现过的域
            for d in domains.unique():
                d_int = d.item()
                if d_int == -1:
                    continue
                # 域名解析
                if all_domains:
                    domain_name = all_domains[d_int] if d_int < len(all_domains) else "Unknown"
                else:
                    domain_name = "Unknown"
                mask = (domains == d)  # 当前 batch 属于该域的样本
                num_d = int(mask.sum().item())
                if num_d == 0:
                    continue  # 理论上不会出现
                acc1_d, acc5_d = accuracy(output[mask], target[mask], topk=(1, 5))
                domain_top1[domain_name].update(acc1_d.item(), num_d)
                domain_top5[domain_name].update(acc5_d.item(), num_d)

        # ------------------------------------------------------------
        # 记录 batch 时间 / 打印进度
        # ------------------------------------------------------------
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            method_name = os.path.join(os.path.basename(args.image_root),
                                       os.path.basename(args.data))
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'.format(top1=top1, top5=top5))
    progress.write_log(len(val_loader), args.log_path)  # 最后一行用 len(val_loader) 以免 i 未定义
    with open(args.log_path, 'a') as f1:
        f1.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
                 .format(top1=top1, top5=top5))

    if separate_domain:
        # 标记完全缺失的域
        if all_domains:
            missing = set(all_domains) - set(domain_top1.keys())
            for d in missing:
                info = f'Domain: {d} has no samples in this validation run.'
                print(info)
                with open(args.log_path, 'a') as f1:
                    f1.write(info + '\n')

        for d in sorted(domain_top1.keys()):
            info = (f'Domain: {d} '
                    f'Acc@1: {domain_top1[d].avg:.2f} '
                    f'Acc@5: {domain_top5[d].avg:.2f}')
            print(info)
            with open(args.log_path, 'a') as f1:
                f1.write(info + '\n')

            if tensor_writer is not None:
                tag_prefix = 'test' if test else 'val'
                tensor_writer.add_scalar(f'ACC@1/{tag_prefix}/{d}', domain_top1[d].avg, epoch)
                tensor_writer.add_scalar(f'ACC@5/{tag_prefix}/{d}', domain_top5[d].avg, epoch)

        return top1.avg, domain_top1

    if tensor_writer is not None:
        tag_prefix = 'test' if test else 'val'
        tensor_writer.add_scalar(f'ACC@1/{tag_prefix}', top1.avg, epoch)
        tensor_writer.add_scalar(f'ACC@5/{tag_prefix}', top5.avg, epoch)

    return top1.avg

