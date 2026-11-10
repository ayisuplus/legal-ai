"""离线 Mock LLM：规则式实现，保证无 API Key 也可运行完整流水线。

仅用于 MVP 演示与单测。真实模型接入时实现 LLMClient 协议即可热插拔。
"""

from __future__ import annotations

import json
import re

# 中文日期，如 2025年3月12日 / 3月15日
_DATE_RE = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日")
# 人名：甲乙丙丁 / 张某 / 李某 / 证人A 等
_PERSON_RE = re.compile(r"(?:[甲乙丙丁戊己庚辛]|[张王李赵刘陈杨黄周吴徐孙马朱胡郭何]某|证人\s?[A-Z])")


class MockLLMClient:
    """规则式抽取，确定性的，便于审计与复现。"""

    def complete(self, prompt: str, *, schema: dict | None = None) -> str:
        if schema and schema.get("task") == "extract_entities":
            text = schema.get("text", "")
            persons = sorted(set(_PERSON_RE.findall(text)))
            dates = sorted({
                f"{y}年{m}月{d}日" if y else f"{m}月{d}日"
                for y, m, d in _DATE_RE.findall(text)
            })
            return json.dumps({"persons": persons, "dates": dates}, ensure_ascii=False)
        # 兜底：不做自由生成，显式报告能力缺失
        return json.dumps(
            {"note": "MockLLM 不支持该任务", "prompt_head": prompt[:50]},
            ensure_ascii=False,
        )
