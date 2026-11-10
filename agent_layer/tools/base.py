"""工具协议：统一声明 + 只读约束 + 结构化产出。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

from ..core.types import StructuredUnit, ToolSpec

if TYPE_CHECKING:
    from ..core.context import SessionContext


@dataclass
class ToolResult:
    units: list[StructuredUnit] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)  # 检索类工具返回材料
    notes: str = ""


class Tool(Protocol):
    spec: ToolSpec

    def run(self, args: dict, ctx: "SessionContext") -> ToolResult: ...
