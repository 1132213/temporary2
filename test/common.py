import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    设置随机种子以确保实验的可重复性
    
    Args:
        seed: 随机种子值，默认为 42
    """
    # Python 内置 random 模块
    random.seed(seed)
    
    # NumPy 随机种子
    np.random.seed(seed)
    
    # PyTorch 随机种子
    torch.manual_seed(seed)
    
    # 如果使用 CUDA，设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置 PyTorch 的确定性行为（可能影响性能）
    # 注意：这可能会降低性能，但能确保完全可重复
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量（某些库可能会使用）
    os.environ['PYTHONHASHSEED'] = str(seed)

