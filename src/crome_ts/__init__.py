"""
CROME-TS 模块化实现。
"""

from .model import (
    InputPreprocessor,
    PatchTSTEncoder,
    QFormer,
    CROMEAdapter,
    # StatProjector,  <-- 移除这个
    RobustFiLMGenerator, # <-- 新增这个
    CROMETSModel,
    InstructionTokenizer,
    FrozenLLM,
    StatBypassCROMETS1,
)

__all__ = [
    "InputPreprocessor",
    "PatchTSTEncoder",
    "QFormer",
    "CROMEAdapter",
    # "StatProjector", <-- 移除这个
    "RobustFiLMGenerator", # <-- 新增这个
    "CROMETSModel",
    "InstructionTokenizer",
    "FrozenLLM",
    "StatBypassCROMETS1",
]