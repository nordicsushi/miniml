from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class InputSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    type: str = Field(default="any")  # 简化类型系统；后续可扩展成强类型/模式
    optional: bool = False
    default: Any | None = None
    description: str | None = None


class OutputSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    type: str = Field(default="any")
    description: str | None = None
