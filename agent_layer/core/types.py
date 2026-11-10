"""核心数据类型：结构化输出单元与能力等级。

对应启动书第七章：系统不鼓励自由生成完整法律意见，只输出结构化信息单元。
裁判类类型（GUILTY / SENTENCE / FINAL_JUDGMENT …）在枚举中不存在 ——
不是"检查后拒绝"，而是"从数据结构上无法表达"。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum


class OutputType(str, Enum):
    """允许的结构化输出类型（启动书 7.1）。"""

    FACT = "FACT"                                # 事实陈述（须绑定来源）
    SOURCE = "SOURCE"                            # 来源说明
    STATUTE = "STATUTE"                          # 法条引用（须绑定来源）
    CASE = "CASE"                                # 判例/材料引用（须绑定来源）
    TIMELINE = "TIMELINE"                        # 时间线
    ENTITY = "ENTITY"                            # 实体（人物/组织/标的）
    RELATION = "RELATION"                        # 实体关系
    CONFLICT = "CONFLICT"                        # 材料矛盾（只报告冲突，不判断谁说谎）
    MISSING_INFORMATION = "MISSING_INFORMATION"  # 信息缺失
    UNCERTAINTY = "UNCERTAINTY"                  # 不确定性显式声明
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"  # 需要人工判断


#: 必须携带 source 引用的输出类型（可追溯原则，启动书 10.2）
SOURCE_REQUIRED_TYPES = {
    OutputType.FACT,
    OutputType.STATUTE,
    OutputType.CASE,
}


class CapabilityLevel(IntEnum):
    """能力等级（启动书第九章）。Level 4（裁判）物理缺席。"""

    L0_RETRIEVAL = 0   # 纯检索：搜索、引用、定位
    L1_ORGANIZE = 1    # 信息整理：摘要、分类、实体抽取、时间线
    L2_ANALYSIS = 2    # 结构分析：冲突检测、缺失检测、法条相关性
    L3_EXPLANATION = 3  # 辅助解释：法条通常涉及的问题（不落个案结论）


class BoundaryViolation(Exception):
    """能力边界违规：工具越权或输出越界时抛出。"""


@dataclass
class StructuredUnit:
    """系统的唯一合法输出载体。"""

    type: OutputType
    content: str
    source: str | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class ToolSpec:
    """工具声明：名称、所需权限等级、可产出类型、无副作用保证。"""

    name: str
    description: str
    required_level: CapabilityLevel
    output_types: tuple[OutputType, ...]
    side_effect_free: bool = True  # MVP 阶段强制只读
