"""上下文管理：会话状态与来源优先的预算裁剪。

对应启动书 8.2：不维护跨任务长期自主状态；任务结束即归档。
裁剪时永不丢弃 source 引用（可追溯原则优先于篇幅）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import CapabilityLevel, StructuredUnit


@dataclass
class SessionContext:
    """单次任务的完整工作上下文。"""

    query: str
    max_level: CapabilityLevel
    documents: list[dict] = field(default_factory=list)   # 检索到的材料 {id, text, source}
    units: list[StructuredUnit] = field(default_factory=list)  # 中间/最终结构化单元

    def add_units(self, units: list[StructuredUnit]) -> None:
        self.units.extend(units)

    @property
    def corpus_text(self) -> str:
        return "\n".join(doc["text"] for doc in self.documents)


class ContextManager:
    """来源优先的上下文预算控制。"""

    def __init__(self, char_budget: int = 6000):
        self.char_budget = char_budget

    def trim_documents(self, docs: list[dict]) -> list[dict]:
        """超出预算时截断正文，但完整保留 id 与 source 字段。"""
        kept: list[dict] = []
        used = 0
        for doc in docs:
            overhead = len(doc.get("id", "")) + len(doc.get("source", ""))
            remaining = self.char_budget - used - overhead
            if remaining <= 0:
                # 来源元信息永远保留，正文为空也不丢弃条目
                kept.append({**doc, "text": ""})
                continue
            text = doc["text"][:remaining]
            kept.append({**doc, "text": text})
            used += overhead + len(text)
        return kept
