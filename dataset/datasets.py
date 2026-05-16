from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import torch
from PIL import Image
import os
import warnings
import numpy as np


def extract_all_domains(dataset_root, data_types=['train', 'val', 'test']):
    """
    扫描数据集目录下的所有文件，提取并返回所有唯一的域名。

    Args:
        dataset_root (str): 数据集根目录路径。
        data_types (list): 需要扫描的数据类型列表，例如 ['train', 'val', 'test']。

    Returns:
        list: 所有唯一的域名列表。
    """
    domains = set()
    for data_type in data_types:
        pattern = os.path.join(dataset_root, f'*{data_type}.txt')
        files = glob.glob(pattern)
        for file in files:
            basename = os.path.basename(file)
            if '_' in basename:
                domain = basename.split('_')[0]
            else:
                domain = "Unknown"
            domains.add(domain)
    return sorted(domains)


def build_domain_mappings(all_domains):
    """
    根据所有域名列表，构建域到索引和索引到域的映射。

    Args:
        all_domains (list): 所有唯一的域名列表。

    Returns:
        tuple: (domain_to_idx, idx_to_domain)
    """
    domain_to_idx = {domain: idx for idx, domain in enumerate(all_domains)}
    idx_to_domain = {idx: domain for domain, idx in domain_to_idx.items()}
    return domain_to_idx, idx_to_domain


class CustomDataset(Dataset):
    def __init__(self, img_paths, transform=None, dataset="PACS", domains=None, domain_to_idx=None, cached_images=None):
        """
        Args:
            img_paths (list of tuples): ????????(??????, ???)
            transform (callable, optional): ???????????
            dataset (str, optional): ????????
            domains (list of str, optional): ???????????????
            domain_to_idx (dict, optional): ???????????
            cached_images (list[np.ndarray], optional): ?????? RGB ??????
        """
        self.img_paths = img_paths
        self.transform = transform
        self.dataset = dataset
        self.domains = domains
        self.domain_to_idx = domain_to_idx
        self.cached_images = cached_images

    def __len__(self):
        return len(self.img_paths)

    def enable_image_cache(self, *, max_bytes=None, label='dataset'):
        cached = []
        total_bytes = 0
        for img_path, _ in self.img_paths:
            image = Image.open(img_path).convert('RGB')
            array = np.asarray(image, dtype=np.uint8).copy()
            total_bytes += int(array.nbytes)
            if max_bytes is not None and total_bytes > int(max_bytes):
                self.cached_images = None
                return {
                    'enabled': False,
                    'label': label,
                    'count': len(cached),
                    'cache_bytes': int(total_bytes),
                    'cache_gb': round(total_bytes / (1024 ** 3), 4),
                    'reason': 'max_bytes_exceeded',
                }
            cached.append(array)
        self.cached_images = cached
        return {
            'enabled': True,
            'label': label,
            'count': len(cached),
            'cache_bytes': int(total_bytes),
            'cache_gb': round(total_bytes / (1024 ** 3), 4),
            'reason': 'ok',
        }

    def __getitem__(self, index):
        img_path, label = self.img_paths[index]
        if self.cached_images is not None:
            image = Image.fromarray(self.cached_images[index], mode='RGB')
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.dataset == "PACS":
            label -= 1

        domain = self.domains[index] if self.domains is not None else "Unknown"
        domain_idx = self.domain_to_idx.get(domain, -1)  # ??? -1 ????????
        domain_idx = torch.tensor(domain_idx, dtype=torch.long)  # ????????
        return image, label, domain_idx


def concat_load_datasets(dataset_root, image_root, data_type, transform, dataset, domain_to_idx):
    """
    加载并合并指定类型（train, val, test）的数据集，同时记录每个样本所属的域。

    Args:
        dataset_root (str): 数据集根目录路径。
        image_root (str): 图片根目录路径。
        data_type (str): 数据类型，例如 'train', 'val', 'test'。
        transform (callable): 数据预处理转换。
        dataset (str): 数据集名称。
        domain_to_idx (dict): 域到索引的映射。

    Returns:
        CustomDataset or None: 返回加载的自定义数据集，如果没有有效数据则返回 None。
    """
    files = glob.glob(os.path.join(dataset_root, f'*{data_type}.txt'))
    img_paths = []
    domains = []
    for file in files:
        basename = os.path.basename(file)
        if '_' in basename:
            domain = basename.split('_')[0]
        else:
            domain = "Unknown"

        with open(file, 'r') as f:
            lines = f.read().splitlines()
            if not lines:
                continue

            for line in lines:
                path, label = line.split()
                img_paths.append((os.path.join(image_root, path), int(label)))
                domains.append(domain)

    if img_paths:
        return CustomDataset(img_paths, transform=transform, dataset=dataset, domains=domains,
                             domain_to_idx=domain_to_idx)
    else:
        print(f"No valid {data_type} datasets found.")
        return None


def cache_loader_data(loader):
    """
    缓存 DataLoader 的所有数据，避免重复从磁盘加载。
    """
    data_cache = []
    for batch in loader:
        if len(batch) == 3:
            images, labels, domains = batch
            data_cache.append((images.clone(), labels.clone(), domains.clone()))
        elif len(batch) == 2:
            images, labels = batch
            domains = torch.full((images.size(0),), -1, dtype=torch.long, device=images.device)  # 使用 -1 表示未知域
            data_cache.append((images.clone(), labels.clone(), domains))
        else:
            raise ValueError("Unexpected batch size")
    return data_cache


def maybe_enable_dataset_image_cache(dataset, *, enabled, max_gb, label):
    if dataset is None or not enabled:
        return {
            'enabled': False,
            'label': label,
            'count': len(dataset) if dataset is not None else 0,
            'cache_bytes': 0,
            'cache_gb': 0.0,
            'reason': 'disabled',
        }
    max_bytes = None if max_gb is None else float(max_gb) * (1024 ** 3)
    return dataset.enable_image_cache(max_bytes=max_bytes, label=label)


def build_dataLoader(args, cached=False):
    image_root = args.image_root
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.min_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4, .4, .4, .4),
        transforms.RandomGrayscale(args.gray_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 自动提取所有域
    all_domains = extract_all_domains(args.data, data_types=['train', 'val', 'test'])
    print(f"Detected domains: {all_domains}")

    # 构建域到索引的映射
    domain_to_idx, idx_to_domain = build_domain_mappings(all_domains)

    # 加载训练、验证和测试数据集
    train_dataset = concat_load_datasets(args.data, image_root, 'train', train_transform, args.dataset, domain_to_idx)
    val_dataset = concat_load_datasets(args.data, image_root, 'val', val_transform, args.dataset, domain_to_idx)
    test_dataset = concat_load_datasets(args.data, image_root, 'test', val_transform, args.dataset, domain_to_idx)

    # 根据是否使用分布式训练，设置采样器
    if args.distributed:
        print("Initializing distributed sampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        cached = False
    else:
        train_sampler = None
    if cached or args.workers == 0:
        persistent_workers = False
    else:
        persistent_workers = True

    # 构建训练、验证和测试的 DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=False,
                             persistent_workers=persistent_workers)

    # 缓存数据（如果需要）
    if cached:
        print('Caching all images to memory. Pay attention to memory usage')
        val_loader = cache_loader_data(val_loader)
        test_loader = cache_loader_data(test_loader)
        # train_loader = [cache_loader_data(train_loader) for epoch in range(args.start_epoch, args.epochs)]

    # 返回映射，以便在其他地方使用（例如 validate 函数）
    return train_loader, val_loader, test_loader, train_sampler, all_domains, idx_to_domain
