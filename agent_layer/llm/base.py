"""LLM 抽象层。

Agent 层只依赖该协议；模型能力约束（数据/训练目标/输出空间）在模型侧实现，
行为边界由 Agent 层的 IntentGate + BoundaryGuard 强制。
替换真实模型（本地模型 / API）时只需实现 complete()。
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str, *, schema: dict | None = None) -> str:
        """生成文本。schema 非空时应输出符合该结构的 JSON 字符串。"""
        ...
