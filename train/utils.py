import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist

class DistributedLengthGroupedSampler(Sampler):
    """
    分桶采样器：根据样本长度进行分组，减少同一个 Batch 内的 Padding 浪费。
    支持 DDP 分布式训练。
    """
    def __init__(self, dataset, batch_size, world_size, rank, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 假设 dataset 提供了 lengths 属性
        if not hasattr(dataset, 'lengths'):
             raise ValueError("Dataset must have 'lengths' attribute for bucketing.")
        
        self.lengths = dataset.lengths
        
    def __iter__(self):
        # 确定性 shuffling
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = list(range(len(self.dataset)))
        
        # 1. 根据长度对索引进行排序
        # 这样长度相近的就在一起了
        indices.sort(key=lambda x: self.lengths[x])
        
        # 2. 构建 Buckets (Batches)
        # 此时的 batches 列表里的每个 batch 内部长度是相似的
        # 注意：这里还没有进行 DDP 分片，我们先生成全局的 batches
        # 为了保证随机性，我们不是完全严格按长度排序训练，
        # 而是生成“长度相近的池子”，但最简单的做法是：
        #   先排序 -> 切分成 huge_batches -> 内部 shuffle -> 再切分成 mini_batches
        # 这里使用更简单有效的方法：
        #   排序 -> 按 batch_size 切分 -> 对 batches 进行 shuffle
        
        # 全局 batch 列表
        all_batches = [
            indices[i : i + self.batch_size * self.world_size]
            for i in range(0, len(indices), self.batch_size * self.world_size)
        ]
        
        # 去掉最后一个可能不完整的 batch (可选，drop_last)
        if len(all_batches[-1]) < self.batch_size * self.world_size:
            all_batches.pop()
            
        # 3. Shuffle Batches
        # 打乱 Batch 的顺序，这样模型不会先学短的再学长的
        if self.shuffle:
            batch_indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_indices]
            
        # 4. 分配给各个 Rank
        # 每个 batch 包含 N * world_size 个样本
        # 我们需要将其拆分给各个 GPU
        final_indices = []
        for batch in all_batches:
            # batch 长度为 batch_size * world_size
            # 当前 rank 取属于自己的一份
            start = self.rank * self.batch_size
            end = start + self.batch_size
            final_indices.extend(batch[start:end])
            
        return iter(final_indices)

    def __len__(self):
        # 估算长度
        total_samples = len(self.dataset)
        # Drop last logic
        num_batches = total_samples // (self.batch_size * self.world_size)
        return num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch