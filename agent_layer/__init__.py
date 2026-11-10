"""能力有界法律智能 —— Agent 层（受限编排器）。

设计原则（见 docs/agent_layer_architecture.md）：
1. 无自主行动循环：run() 单向执行 gate → plan → execute → guard → terminate。
2. 输出仅允许结构化信息单元（StructuredUnit），裁判类输出类型物理缺席。
3. FACT/STATUTE/CASE 单元必须绑定来源（可追溯原则）。
4. 工具按 Level 0-3 分级授权，Level 4（裁判）在本层不存在。
"""

from .core.types import CapabilityLevel, OutputType, StructuredUnit
from .core.orchestrator import AgentOrchestrator, AgentResult

__all__ = [
    "CapabilityLevel",
    "OutputType",
    "StructuredUnit",
    "AgentOrchestrator",
    "AgentResult",
]
