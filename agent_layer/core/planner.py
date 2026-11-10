"""受限任务规划：从预定义流水线模板实例化静态 Plan。

对应启动书 8.1 受限流程：
    检索 → 抽取 → 结构化 → 来源绑定 → 冲突与缺失检测 → 输出 → 终止

与通用 Agent 的本质区别：
- 不做开放式目标分解，不做"根据结果再规划"；
- Plan 生成后静态执行，执行中不新增步骤、不调用计划外工具；
- 信息不足时输出 MISSING_INFORMATION 并终止，而不是自主扩展任务。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Step(str, Enum):
    RETRIEVE = "retrieve"               # 信息检索
    EXTRACT = "extract"                 # 内容抽取
    STRUCTURE = "structure"             # 信息结构化（时间线/关系）
    BIND_SOURCES = "bind_sources"       # 来源绑定
    DETECT_CONFLICTS = "detect_conflicts"  # 冲突与缺失检测
    COMPOSE = "compose"                 # 输出组装
    TERMINATE = "terminate"             # 终止


@dataclass(frozen=True)
class Plan:
    intent: str
    steps: tuple[Step, ...]


class TaskPlanner:
    """模板化规划器：按意图从受限流水线裁剪步骤。"""

    #: 完整受限流水线（默认）
    FULL_PIPELINE = (
        Step.RETRIEVE,
        Step.EXTRACT,
        Step.STRUCTURE,
        Step.BIND_SOURCES,
        Step.DETECT_CONFLICTS,
        Step.COMPOSE,
        Step.TERMINATE,
    )

    #: 纯检索意图的裁剪流水线
    RETRIEVAL_PIPELINE = (Step.RETRIEVE, Step.COMPOSE, Step.TERMINATE)

    _RETRIEVAL_KEYWORDS = ("查找", "检索", "引用", "定位", "列出法条", "找出条文")
    _CONFLICT_KEYWORDS = ("冲突", "矛盾", "不一致", "对不上")
    _TIMELINE_KEYWORDS = ("时间线", "时间顺序", "经过")

    def create_plan(self, query: str) -> Plan:
        if any(k in query for k in self._RETRIEVAL_KEYWORDS):
            return Plan(intent="retrieval", steps=self.RETRIEVAL_PIPELINE)
        steps = list(self.FULL_PIPELINE)
        if not any(k in query for k in self._CONFLICT_KEYWORDS + ("缺失",)):
            steps.remove(Step.DETECT_CONFLICTS)
        if not any(k in query for k in self._TIMELINE_KEYWORDS + ("整理", "梳理")):
            steps.remove(Step.STRUCTURE)
        return Plan(intent="analysis", steps=tuple(steps))
