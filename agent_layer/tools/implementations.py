"""MVP 工具实现（法条检索已升级为「本地向量检索为主 + BM25 降级」双路径）。

法条检索策略（与 LLM Fallback 同风格，见 agent_layer/llm/fallback.py）：
- auto（默认）：首次检索时尝试初始化向量索引，任何异常（ImportError /
  模型下载失败 / 运行时错误）捕获后 stderr 警告留痕并降级 BM25；
  每实例只尝试一次，失败后不再重试；
- vector：强制向量路径，初始化/检索失败直接抛错（便于测试验收）；
- bm25：强制 BM25 路径（行为与迭代三基线完全一致）。
模式由环境变量 LEGAL_AI_RETRIEVAL_MODE 控制（构造参数可覆盖）。
两条路径下无命中时行为一致：输出 MISSING_INFORMATION 单元，绝不伪造法条。
后续接入真实法条库时，ToolSpec 与协议不变，仅需替换检索引擎。
"""

from __future__ import annotations

import json
import os
import re
import sys

from ..core.context import SessionContext
from ..core.types import CapabilityLevel, OutputType, StructuredUnit, ToolSpec
from ..llm.base import LLMClient
from .base import ToolResult
from .retrieval import BM25Index, FALLBACK_STATUTES, load_statute_corpus

#: 检索模式合法取值（非法值按 auto 处理并 stderr 告警）
_RETRIEVAL_MODES = ("auto", "vector", "bm25")

# ---------------------------------------------------------------- 示例语料

#: 兼容旧引用：与内置兜底语料相同（真实语料见 data/statutes.jsonl）
SAMPLE_STATUTES = FALLBACK_STATUTES

SAMPLE_CASE_DOCS = [
    {
        "id": "doc-contract",
        "source": "合同原件第 1 页",
        "text": "甲方（张某）与乙方（李某）于2025年3月12日签订货物买卖合同，"
                "约定乙方于2025年3月20日前支付货款人民币50万元。",
    },
    {
        "id": "doc-witness-a",
        "source": "证人A询问笔录第 2 页",
        "text": "证人A称：李某于3月15日通过银行转账向张某支付了货款。",
    },
    {
        "id": "doc-bank",
        "source": "银行流水第 1 页",
        "text": "银行流水显示：2025年3月18日，李某账户向张某账户转账50万元。",
    },
]

_DATE_RE = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日")


# ---------------------------------------------------------------- 工具实现

class StatuteSearchTool:
    spec = ToolSpec(
        name="statute_search",
        description="按关键词检索法条（Level 0）",
        required_level=CapabilityLevel.L0_RETRIEVAL,
        output_types=(OutputType.STATUTE, OutputType.SOURCE),
    )

    def __init__(
        self,
        corpus: list[dict] | None = None,
        top_k: int = 5,
        retrieval_mode: str | None = None,
    ):
        # 语料加载优先级：显式传入 > 环境变量 > 项目 data/statutes.jsonl > 内置兜底
        self.corpus = corpus if corpus is not None else load_statute_corpus()
        self.top_k = top_k
        # 检索模式：构造参数 > 环境变量 > 默认 auto；非法值告警后按 auto 处理
        mode = (
            retrieval_mode
            or os.environ.get("LEGAL_AI_RETRIEVAL_MODE")
            or "auto"
        ).strip().lower()
        if mode not in _RETRIEVAL_MODES:
            print(f"[警告] 未知检索模式 {mode!r}，按 auto 处理"
                  f"（合法取值：{'/'.join(_RETRIEVAL_MODES)}）。", file=sys.stderr)
            mode = "auto"
        self.retrieval_mode = mode
        # BM25 索引始终构建：既是 bm25 模式的主路径，也是 auto/vector 的降级底
        self._bm25_index = BM25Index(self.corpus)
        # 向量索引惰性初始化：_vector_attempted 保证每实例最多尝试一次
        self._vector_index = None
        self._vector_attempted = False

    def _ensure_vector_index(self):
        """惰性初始化向量索引；auto 模式失败留痕返回 None，vector 模式抛错。"""
        if self._vector_index is not None:
            return self._vector_index
        if self._vector_attempted:
            return None  # 已知失败：本实例不再重试，避免重复开销
        self._vector_attempted = True
        try:
            from .vector_retrieval import VectorIndex  # 延迟导入，裸 python 不崩
            self._vector_index = VectorIndex(self.corpus)
        except Exception as e:
            if self.retrieval_mode == "vector":
                raise  # 强制向量模式：失败抛错便于测试验收
            print(f"[警告] 向量检索初始化失败（{type(e).__name__}: {e}），"
                  f"本次及后续检索降级 BM25。", file=sys.stderr)
            self._vector_index = None
        return self._vector_index

    def _vector_search(self, query_str: str) -> list[dict] | None:
        """向量路径检索；auto 模式下任何失败留痕并返回 None（由调用方降级）。"""
        try:
            vindex = self._ensure_vector_index()
            if vindex is None:
                return None
            # VectorIndex.search 已按 min_score 门槛过滤
            return [doc for doc, _ in vindex.search(query_str, self.top_k)]
        except Exception as e:
            if self.retrieval_mode == "vector":
                raise
            print(f"[警告] 向量检索运行失败（{type(e).__name__}: {e}），"
                  f"本次及后续检索降级 BM25。", file=sys.stderr)
            self._vector_index = None  # 运行时失败同样禁用向量路径
            return None

    def _search(self, query_str: str) -> list[dict]:
        """按模式分发检索：向量为主、BM25 兜底，两路径均只返回命中文档。"""
        if self.retrieval_mode != "bm25":
            hits = self._vector_search(query_str)
            if hits is not None:
                return hits
        return [
            doc for doc, score in self._bm25_index.search(query_str, self.top_k)
            if score > 0
        ]

    def run(self, args: dict, ctx: SessionContext) -> ToolResult:
        keywords = args.get("keywords") or re.findall(r"[\u4e00-\u9fff]{2,}", ctx.query)
        # 检索发生在案件材料载入之后时，用材料中的法律相关名词补充检索词
        if ctx.documents:
            corpus = ctx.corpus_text
            keywords += [k for k in ("合同", "解除", "违约", "履行") if k in corpus]
        hits = self._search(" ".join(keywords))
        units = [
            StructuredUnit(type=OutputType.STATUTE, content=s["text"], source=s["source"])
            for s in hits
        ]
        if not hits:
            units = [StructuredUnit(
                type=OutputType.MISSING_INFORMATION,
                content="未检索到与请求相关的法条。",
            )]
        return ToolResult(units=units, documents=hits)


class CaseSearchTool:
    spec = ToolSpec(
        name="case_search",
        description="检索案件材料（Level 0）",
        required_level=CapabilityLevel.L0_RETRIEVAL,
        output_types=(OutputType.CASE, OutputType.SOURCE),
    )

    def run(self, args: dict, ctx: SessionContext) -> ToolResult:
        # MVP：示例案件材料全量载入，真实场景下替换为检索逻辑
        return ToolResult(documents=list(SAMPLE_CASE_DOCS))


class EntityExtractTool:
    spec = ToolSpec(
        name="entity_extract",
        description="抽取人物、时间等实体（Level 1）",
        required_level=CapabilityLevel.L1_ORGANIZE,
        output_types=(OutputType.ENTITY,),
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, args: dict, ctx: SessionContext) -> ToolResult:
        merged: dict[str, set[str]] = {}
        for doc in ctx.documents:
            if doc["source"].startswith("《"):
                continue  # 法条不是案件事实材料，不做实体抽取
            raw = self.llm.complete(
                "抽取人物与日期", schema={"task": "extract_entities", "text": doc["text"]}
            )
            data = json.loads(raw)
            for p in data.get("persons", []):
                merged.setdefault(f"人物：{p}", set()).add(doc["source"])
            for d in data.get("dates", []):
                merged.setdefault(f"日期：{d}", set()).add(doc["source"])
        units = [
            StructuredUnit(type=OutputType.ENTITY, content=content,
                           source="；".join(sorted(sources)))
            for content, sources in merged.items()
        ]
        return ToolResult(units=units)


class TimelineTool:
    spec = ToolSpec(
        name="timeline_build",
        description="按时间顺序整理事件（Level 1）",
        required_level=CapabilityLevel.L1_ORGANIZE,
        output_types=(OutputType.TIMELINE, OutputType.FACT),
    )

    def run(self, args: dict, ctx: SessionContext) -> ToolResult:
        # 推断默认年份：材料中显式出现的最频年份，用于缺失年份的日期排序
        years = [int(y) for doc in ctx.documents
                 for y, _, _ in _DATE_RE.findall(doc["text"]) if y]
        default_year = max(set(years), key=years.count) if years else 0
        events = []
        for doc in ctx.documents:
            for m in _DATE_RE.finditer(doc["text"]):
                y, mo, d = m.groups()
                key = (int(y) if y else default_year, int(mo), int(d))
                # 取包含日期的分句作为事件描述
                sent = next(
                    (s for s in re.split(r"[。；]", doc["text"]) if m.group(0) in s),
                    doc["text"],
                )
                events.append((key, m.group(0), sent.strip(), doc["source"]))
        events.sort(key=lambda e: e[0])
        units = [
            StructuredUnit(
                type=OutputType.TIMELINE,
                content=f"{date_str}：{desc}",
                source=src,
            )
            for _, date_str, desc, src in events
        ]
        return ToolResult(units=units)


class ConflictDetectTool:
    spec = ToolSpec(
        name="conflict_detect",
        description="检测材料间的日期/数字冲突（Level 2，只报告冲突不判断谁说谎）",
        required_level=CapabilityLevel.L2_ANALYSIS,
        output_types=(OutputType.CONFLICT,),
    )

    def run(self, args: dict, ctx: SessionContext) -> ToolResult:
        # 按“同一事件（实际付款/转账行为）在不同材料中的日期”做最小示例检测：
        # 只看包含“支付了/转账”的分句，避免把合同签订日等无关日期混入。
        claims = []
        for doc in ctx.documents:
            for sent in re.split(r"[。；]", doc["text"]):
                if "支付了" not in sent and "转账" not in sent:
                    continue
                for m in _DATE_RE.finditer(sent):
                    _, mo, d = m.groups()
                    claims.append((f"{mo}月{d}日", doc["source"]))
        units: list[StructuredUnit] = []
        distinct_dates = {c[0] for c in claims}
        if len(distinct_dates) > 1:
            detail = "；".join(f"{src}记载付款日期为 {dt}" for dt, src in claims)
            units.append(StructuredUnit(
                type=OutputType.CONFLICT,
                content=f"关于付款日期，材料之间存在冲突：{detail}。系统仅报告矛盾存在，不判断哪份材料更可信。",
                source="；".join(sorted({src for _, src in claims})),
            ))
        return ToolResult(units=units)


class MissingInfoTool:
    spec = ToolSpec(
        name="missing_info_detect",
        description="检测关键信息缺失（Level 2）",
        required_level=CapabilityLevel.L2_ANALYSIS,
        output_types=(OutputType.MISSING_INFORMATION,),
    )

    REQUIRED_HINTS = ("合同", "交付", "违约")

    def run(self, args: dict, ctx: SessionContext) -> ToolResult:
        corpus = "\n".join(
            doc["text"] for doc in ctx.documents
            if not doc["source"].startswith("《")  # 只检查案件事实材料
        )
        missing = [h for h in self.REQUIRED_HINTS if h not in corpus]
        units = [
            StructuredUnit(
                type=OutputType.MISSING_INFORMATION,
                content=f"现有材料中未发现与“{h}”相关的信息，建议人工补充或确认。",
            )
            for h in missing
        ]
        return ToolResult(units=units)


def build_default_registry(llm: LLMClient) -> dict:
    """构建默认工具注册表（Level 4 工具不存在 —— 物理缺席）。"""
    tools = [
        StatuteSearchTool(),
        CaseSearchTool(),
        EntityExtractTool(llm),
        TimelineTool(),
        ConflictDetectTool(),
        MissingInfoTool(),
    ]
    return {t.spec.name: t for t in tools}
