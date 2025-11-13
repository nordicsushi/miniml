from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict


class PipelineInputRef(BaseModel):
    """引用 Pipeline 顶层的某个输入（如 AzureML v2 的 pipeline inputs）"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str


class UpstreamOutputRef(BaseModel):
    """引用上游组件的某个输出"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node: str  # 上游组件在 Pipeline 中的别名
    output: str  # 上游组件的 outputs spec 中的键


class ConstantValue(BaseModel):
    """常量绑定"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: Any


BindingValue = PipelineInputRef | UpstreamOutputRef | ConstantValue
