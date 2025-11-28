from __future__ import annotations
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np

class TimeMMDInstructDataset(Dataset):
    """
    Time-MMD 指令微调数据集
    支持 'report' 和 'search' 两种文本源，并自动处理日期对齐。
    """
    def __init__(
        self,
        root: str,
        subset: str,
        seq_len: int,
        input_channels: int,
        text_source: str = "search", # 新增：指定文本源
        split: str = "train",
        split_ratio: float = 0.9
    ):
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.text_source = text_source
        
        # 1. 确定文件路径
        root_path = Path(root)
        num_path = root_path / "numerical" / subset / f"{subset}.csv"
        
        # 根据 source 决定文件名
        # report -> {subset}_report.csv
        # search -> {subset}_search.csv
        suffix = "report" if text_source == "report" else "search"
        text_path = root_path / "textual" / subset / f"{subset}_{suffix}.csv"
        
        if not num_path.exists():
            raise FileNotFoundError(f"Numerical data not found: {num_path}")
        if not text_path.exists():
            raise FileNotFoundError(f"Textual data not found: {text_path}")

        self.num_df = pd.read_csv(num_path).sort_values(by="start_date")
        self.text_df = pd.read_csv(text_path)
        
        # 2. 预处理：建立 日期 -> 报告 的映射
        self.text_map = self._build_text_map()
        self.numeric_data = self._load_numeric_data()
        
        # 3. 生成样本索引
        self.samples = self._make_indices()
        
        if len(self.samples) == 0:
            # 打印调试信息帮助定位问题
            print(f"!!! ERROR: No samples found for {subset} using source '{text_source}' !!!")
            if len(self.num_df) > 0:
                first_date = self.num_df.iloc[self.seq_len-1]["start_date"]
                print(f"Debug: First numeric window ends at {first_date}")
            if len(self.text_map) > 0:
                print(f"Debug: First 3 available text dates: {list(self.text_map.keys())[:3]}")
            raise ValueError("Dataset is empty. Check date alignment or text source.")

        # 4. 划分训练/验证集
        split_idx = int(len(self.samples) * split_ratio)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
            
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples from {subset} ({text_source})")

    def _build_text_map(self):
        text_map = {}
        for row in self.text_df.itertuples():
            fact = getattr(row, "fact", "")
            start_date = getattr(row, "start_date", None)
            
            if isinstance(fact, str) and fact.strip() and not pd.isna(start_date):
                # 归一化日期格式
                date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
                text_map[date_str] = fact.strip()
        return text_map

    def _load_numeric_data(self):
        cols = [c for c in self.num_df.columns if pd.api.types.is_numeric_dtype(self.num_df[c]) and c != "MapDate"]
        # 修复 FutureWarning: 使用 ffill().fillna(0.0) 替代 fillna(method='ffill')
        data = self.num_df[cols].ffill().fillna(0.0).values
        return torch.tensor(data, dtype=torch.float32)

    def _make_indices(self):
        indices = []
        valid_dates = set(self.text_map.keys())
        total_len = len(self.numeric_data)
        
        # 根据 source 设置日期偏移
        # search: 通常是对前一天的报道，或者是当天发生的新闻对应当天的搜索量
        # Time-MMD 的惯例：search 数据通常需要 offset -1 天来对齐数值（或者数值对应 search？）
        # 参考 common.py 中的成功逻辑：if source == 'search', offset = -1
        offset_days = -1 if self.text_source == "search" else 0
        
        # 为了获得更多样本，这里 stride 设为 1 (Full Training)
        # 如果数据量太大，可以改为 seq_len // 2
        stride = 1 
        
        for i in range(0, total_len - self.seq_len, stride):
            end_idx = i + self.seq_len
            
            # 获取窗口末端的日期
            date_val = self.num_df.iloc[end_idx-1]["start_date"]
            
            # 应用偏移
            date_obj = pd.to_datetime(date_val) + pd.Timedelta(days=offset_days)
            date_str = date_obj.strftime("%Y-%m-%d")
            
            if date_str in valid_dates:
                indices.append((i, date_str))
        return indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx, date_key = self.samples[idx]
        
        window = self.numeric_data[start_idx : start_idx + self.seq_len]
        
        if window.shape[1] > self.input_channels:
            window = window[:, :self.input_channels]
        elif window.shape[1] < self.input_channels:
            pad = torch.zeros(self.seq_len, self.input_channels - window.shape[1])
            window = torch.cat([window, pad], dim=1)
            
        # 简单的 System Prompt
        prefix = (
            "User: Analyze the following time series data.\n"
            f"Context: Climate data ending on {date_key}.\n"
            "Assistant: "
        )
        report = self.text_map[date_key]
        
        return {
            "series": window,
            "prefix": prefix,
            "suffix": report
        }

def _record_to_array(record: Dict[str, Any], record_idx: int) -> np.ndarray:
    """
    将 JSONL 记录中的时序数据解析为二维 numpy 数组。
    输入数据应该是 [T, C] 格式（时间步在外层，通道在内层）。
    
    输出格式为 [时间步, 通道]，即 [T, C]
    
    Args:
        record: JSON记录
        record_idx: 记录索引（用于错误提示）
    """
    ts_data = record.get("input_ts")
    if ts_data is None:
        ts_data = record.get("ts_data")
    if ts_data is None:
        raise ValueError(
            f"Record {record_idx} 缺少内联时序字段 `input_ts`，"
            "请先将遗留的 `*_ts_path` 数据迁移为嵌入式 JSON。"
        )
    
    array = np.asarray(ts_data, dtype=np.float32)
    
    # 处理不同维度的数据
    if array.ndim == 2:
        # 已经是 [T, C] 格式，保持原样
        pass
    elif array.ndim == 1:
        # 单通道数据，转换为 [T, 1]
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        # 多维数组，展平
        array = array.reshape(array.shape[0], -1)
    
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


class JSONLInstructDataset(Dataset):
    """
    JSONL 格式的指令微调数据集
    支持从 JSONL 文件加载指令、时间序列路径和输出文本。
    数据格式应为 [T, C]（时间步在外层，通道在内层）。
    支持自动检测通道数（当input_channels为None时）。
    """
    def __init__(
        self,
        jsonl_path: str,
        seq_len: int,
        input_channels: Optional[int] = None,  # 如果为None，将从数据中自动检测
        split: str = "train",
        split_ratio: float = 0.9,
    ):
        self.seq_len = seq_len
        
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
        
        # 加载 JSONL 文件
        records = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        
        if len(records) == 0:
            raise ValueError(f"JSONL file is empty: {jsonl_file}")
        
        # 自动检测通道数（如果需要）
        if input_channels is None:
            # 从第一个记录中检测通道数
            first_record = records[0]
            sample_array = _record_to_array(first_record, 0)
            detected_channels = sample_array.shape[1]
            self.input_channels = detected_channels
            print(f"[{split.upper()}] Auto-detected {detected_channels} channels from data")
        else:
            self.input_channels = input_channels
        
        # 划分训练/验证集
        split_idx = int(len(records) * split_ratio)
        if split == "train":
            self.records = records[:split_idx]
        else:
            self.records = records[split_idx:]
        
        print(f"[{split.upper()}] Loaded {len(self.records)} samples from {jsonl_file.name} (channels: {self.input_channels})")
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        
        # 加载时间序列数据（优先读取嵌入式字段）
        array = _record_to_array(record, idx)
        
        # 确保长度匹配
        if array.shape[0] > self.seq_len:
            # 截取最后 seq_len 个时间步
            array = array[-self.seq_len:]
        elif array.shape[0] < self.seq_len:
            # 填充零
            pad_len = self.seq_len - array.shape[0]
            pad = np.zeros((pad_len, array.shape[1]))
            array = np.vstack([array, pad])
        
        # 处理通道数
        if array.shape[1] > self.input_channels:
            array = array[:, :self.input_channels]
        elif array.shape[1] < self.input_channels:
            pad_channels = self.input_channels - array.shape[1]
            pad = np.zeros((self.seq_len, pad_channels))
            array = np.hstack([array, pad])
        
        window = torch.tensor(array, dtype=torch.float32)
        
        # 获取指令和输出文本
        instruction = record.get("instruction", "").strip()
        output_text = record.get("output_text", "").strip()
        
        # 构建 prefix（指令格式）
        if instruction:
            prefix = f"User: {instruction}\nAssistant: "
        else:
            prefix = "User: Analyze the following time series data.\nAssistant: "
        
        return {
            "series": window,
            "prefix": prefix,
            "suffix": output_text
        }

class ChatTSDataset(Dataset):
    """
    ChatTS 格式的指令微调数据集 (支持动态长度 + Patch对齐)
    支持从 JSONL 文件加载包含 <ts><ts/> 标记的文本和多个时间序列。
    
    数据格式：
    {
        "input": "文本 <ts><ts/> 更多文本 <ts><ts/>",
        "timeseries": [[...], [...], ...],  # 二维数组，每个元素是一个时间序列
        "output": "输出文本"
    }
    """
    def __init__(
        self,
        jsonl_path: str,
        seq_len: Optional[int] = None, # 以前是 seq_len: int，现在变为 Optional，充当 max_len
        input_channels: Optional[int] = None,
        split: str = "train",
        split_ratio: float = 0.9,
        patch_stride: int = 16, # 新增：用于对齐 Padding
    ):
        self.max_len = seq_len # 重命名语义，仅用于限制最大显存占用
        self.patch_stride = patch_stride
        
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
        
        # 加载 JSONL 文件
        records = []
        self.lengths = [] # 新增：用于分桶 (Bucketing)

        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                    
                    # 估算样本长度用于分桶
                    # 长度 = 文本长度 + (所有时序长度 / patch_stride)
                    # 这是一个粗略估计，足以用于将相似长度的样本分在一起
                    input_text_len = len(rec.get("input", ""))
                    ts_len_sum = 0
                    for ts in rec.get("timeseries", []):
                        ts_len_sum += len(ts)
                    
                    # 估算 token 数
                    estimated_len = input_text_len + (ts_len_sum // 16) 
                    self.lengths.append(estimated_len)
                except json.JSONDecodeError:
                    continue
        
        if len(records) == 0:
            raise ValueError(f"JSONL file is empty: {jsonl_file}")
        
        # 自动检测通道数（每个时间序列都是单通道）
        if input_channels is None:
            # ChatTS 格式中每个时间序列都是一维的，input_channels 应该是1
            self.input_channels = 1
            print(f"[{split.upper()}] Auto-detected {self.input_channels} channel(s) for ChatTS format")
        else:
            self.input_channels = input_channels
        
        # 划分训练/验证集
        split_idx = int(len(records) * split_ratio)
        if split == "train":
            self.records = records[:split_idx]
            self.lengths = self.lengths[:split_idx]
        else:
            self.records = records[split_idx:]
            self.lengths = self.lengths[split_idx:]
        
        print(f"[{split.upper()}] Loaded {len(self.records)} samples from {jsonl_file.name} (Dynamic Length Mode)")
    
    def __len__(self):
        return len(self.records)

    def _pad_series(self, array: np.ndarray) -> np.ndarray:
        """
        动态 Padding 逻辑：
        1. 对齐到 patch_stride 的倍数
        2. 使用最后一个值填充 (Edge Padding)
        """
        curr_len = len(array)
        stride = self.patch_stride
        
        # 计算目标长度（向上取整到 stride 的倍数）
        if curr_len % stride == 0:
            target_len = curr_len
        else:
            target_len = (curr_len // stride + 1) * stride
        
        # 限制最大长度 (如果设置了 max_len)
        if self.max_len is not None and target_len > self.max_len:
            # 截断，但仍要保证是 stride 倍数
            target_len = (self.max_len // stride) * stride
            # 如果截断后比原长度小，说明原数据太长，直接切片
            if target_len < curr_len:
                return array[:target_len] 
        
        pad_len = target_len - curr_len
        
        if pad_len > 0:
            # 使用最后一个值进行填充
            # array shape: [L, 1] or [L]
            last_val = array[-1] if len(array) > 0 else 0.0
            pad_values = np.full((pad_len, 1), last_val, dtype=array.dtype)
            
            # 确保 array 是 [L, 1]
            if array.ndim == 1:
                array = array.reshape(-1, 1)
                
            array = np.vstack([array, pad_values])
            
        return array
    
    def __getitem__(self, idx):
        record = self.records[idx]
        
        # 获取 input 文本和时间序列数据
        input_text = record.get("input", "").strip()
        timeseries_list = record.get("timeseries", [])
        output_text = record.get("output", "").strip()
        
        # 处理时间序列数据
        processed_series = []
        for ts_data in timeseries_list:
            # 将时间序列转换为 numpy 数组
            array = np.asarray(ts_data, dtype=np.float32)
            
            # 确保是一维数组
            if array.ndim > 1:
                array = array.flatten()
            
            # 处理为 [T, 1]
            array = array.reshape(-1, 1)
            
            # NaN 处理
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # === 动态 Padding ===
            array = self._pad_series(array)
            
            # 转换为 tensor
            ts_tensor = torch.tensor(array, dtype=torch.float32)
            processed_series.append(ts_tensor)
        
        # 如果没有时间序列数据，创建一个最小长度的 Tensor 以防报错
        if len(processed_series) == 0:
            processed_series = [torch.zeros(self.patch_stride, self.input_channels, dtype=torch.float32)]
        
        # 构建返回字典
        return {
            "input_text": input_text,  # 包含 <ts><ts/> 标记的原始文本
            "timeseries_list": processed_series,  # 处理后的时间序列列表
            "output_text": output_text,  # 输出文本
        }


def instruct_collate_fn(batch):
    series = torch.stack([item["series"] for item in batch])
    prefixes = [item["prefix"] for item in batch]
    suffixes = [item["suffix"] for item in batch]
    return series, prefixes, suffixes


def chatts_collate_fn(batch):
    """
    ChatTS 格式的 collate 函数
    处理包含多个时间序列的批次数据
    """
    input_texts = [item["input_text"] for item in batch]
    timeseries_lists = [item["timeseries_list"] for item in batch]
    output_texts = [item["output_text"] for item in batch]
    
    return {
        "input_texts": input_texts,
        "timeseries_lists": timeseries_lists,
        "output_texts": output_texts,
    }