"""LLM 降级包装器与按环境构建工厂。

设计意图与边界约束：
- 真实 API 不可用时自动降级到离线 MockLLMClient，保证流水线始终可运行、可验收；
- 降级事件必须留痕：优先调用 on_degrade 回调（供审计系统记录），
  无回调时向 stderr 打印警告——绝不做静默切换；
- 包装器本身只实现 LLMClient 协议，不引入任何额外行为逻辑。

create_llm_from_env() 工厂：环境变量 LEGAL_AI_LLM_API_KEY 齐全时返回
FallbackLLMClient(OpenAICompatClient, MockLLMClient)，否则直接返回 MockLLMClient
并在 stderr 提示当前为 Mock 模式。
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from .base import LLMClient
from .mock import MockLLMClient
from .openai_compat import LLMError, OpenAICompatClient


class FallbackLLMClient:
    """主备 LLM 包装：primary 抛 LLMError 时降级到 fallback 并留痕。"""

    def __init__(
        self,
        primary: LLMClient,
        fallback: LLMClient,
        on_degrade: Callable[[Exception], None] | None = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.on_degrade = on_degrade

    def complete(self, prompt: str, *, schema: dict | None = None) -> str:
        try:
            return self.primary.complete(prompt, schema=schema)
        except LLMError as e:
            self._notify_degrade(e)
            return self.fallback.complete(prompt, schema=schema)

    def _notify_degrade(self, error: Exception) -> None:
        if self.on_degrade is not None:
            self.on_degrade(error)
        else:
            print(
                f"[警告] 主 LLM 调用失败，已降级到备用实现：{error}",
                file=sys.stderr,
            )


def create_llm_from_env(
    on_degrade: Callable[[Exception], None] | None = None,
) -> LLMClient:
    """按环境变量构建 LLM：有 API Key 走真实端点（带降级），否则 Mock。"""
    if os.environ.get("LEGAL_AI_LLM_API_KEY"):
        try:
            primary = OpenAICompatClient()
        except LLMError as e:
            print(f"[警告] 真实 LLM 初始化失败（{e}），使用 Mock 模式。",
                  file=sys.stderr)
            return MockLLMClient()
        return FallbackLLMClient(primary, MockLLMClient(),
                                 on_degrade=on_degrade)
    print(
        "[提示] 未设置 LEGAL_AI_LLM_API_KEY，当前为 Mock 模式"
        "（规则式离线抽取，确定性输出）。",
        file=sys.stderr,
    )
    return MockLLMClient()
