"""权限模型：会话等级 + 工具/输出类型的等级门控。

对应启动书第九章。运行时不可提权；Level 4 无对应工具与输出类型。
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import CapabilityLevel, OutputType, ToolSpec, BoundaryViolation


#: 各输出类型所需的最低能力等级
OUTPUT_TYPE_MIN_LEVEL = {
    OutputType.FACT: CapabilityLevel.L0_RETRIEVAL,
    OutputType.SOURCE: CapabilityLevel.L0_RETRIEVAL,
    OutputType.STATUTE: CapabilityLevel.L0_RETRIEVAL,
    OutputType.CASE: CapabilityLevel.L0_RETRIEVAL,
    OutputType.ENTITY: CapabilityLevel.L1_ORGANIZE,
    OutputType.TIMELINE: CapabilityLevel.L1_ORGANIZE,
    OutputType.RELATION: CapabilityLevel.L1_ORGANIZE,
    OutputType.CONFLICT: CapabilityLevel.L2_ANALYSIS,
    OutputType.MISSING_INFORMATION: CapabilityLevel.L2_ANALYSIS,
    OutputType.UNCERTAINTY: CapabilityLevel.L0_RETRIEVAL,
    OutputType.HUMAN_REVIEW_REQUIRED: CapabilityLevel.L0_RETRIEVAL,
}


@dataclass(frozen=True)
class PermissionPolicy:
    """一次会话的权限策略（冻结，运行时不可修改/提权）。"""

    max_level: CapabilityLevel = CapabilityLevel.L2_ANALYSIS

    def check_tool(self, spec: ToolSpec) -> None:
        if not spec.side_effect_free:
            raise BoundaryViolation(
                f"工具 {spec.name} 存在副作用，违反只读约束（启动书 8.2）。"
            )
        if spec.required_level > self.max_level:
            raise BoundaryViolation(
                f"工具 {spec.name} 需要 {spec.required_level.name}，"
                f"当前会话最高 {self.max_level.name}，拒绝调用。"
            )

    def check_output_type(self, output_type: OutputType) -> bool:
        min_level = OUTPUT_TYPE_MIN_LEVEL[output_type]
        return min_level <= self.max_level
