"""受限编排器：gate → plan → execute → guard → terminate（单向，无自主循环）。

这是 Agent 层的主入口。与通用 Agent 的区别：
- 不存在"观察输出再决定下一步"的环路；Plan 静态生成、静态执行；
- 每一步的产出都经 BoundaryGuard 校验后才进入上下文；
- 被拒绝的单元降级为 HUMAN_REVIEW_REQUIRED，全程留痕。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .context import ContextManager, SessionContext
from .guardrails import BoundaryGuard, IntentGate
from .permissions import PermissionPolicy
from .planner import Plan, Step, TaskPlanner
from .trace import TraceRecorder
from .types import (
    BoundaryViolation,
    CapabilityLevel,
    OutputType,
    StructuredUnit,
)

#: 流水线步骤 → 工具名 的静态映射（计划外工具不可被调用）
STEP_TOOL_MAP = {
    Step.RETRIEVE: ("case_search", "statute_search"),  # 先载材料，法条检索可用材料名词补充检索词
    Step.EXTRACT: ("entity_extract",),
    Step.STRUCTURE: ("timeline_build",),
    Step.DETECT_CONFLICTS: ("conflict_detect", "missing_info_detect"),
    # BIND_SOURCES / COMPOSE / TERMINATE 由编排器内部完成，不对应工具
}


@dataclass
class AgentResult:
    status: str                     # "completed" | "refused" | "error"
    units: list[StructuredUnit] = field(default_factory=list)
    message: str = ""
    trace_path: str | None = None

    def pretty(self) -> str:
        lines = [f"[status] {self.status}"]
        if self.message:
            lines.append(f"[message] {self.message}")
        for u in self.units:
            src = f"  (来源: {u.source})" if u.source else ""
            lines.append(f"  <{u.type.value}> {u.content}{src}")
        if self.trace_path:
            lines.append(f"[trace] {self.trace_path}")
        return "\n".join(lines)


class AgentOrchestrator:
    def __init__(
        self,
        tools: dict,
        llm=None,
        policy: PermissionPolicy | None = None,
        trace_dir: str | Path = "traces",
        char_budget: int = 6000,
    ):
        self.tools = tools
        self.policy = policy or PermissionPolicy()
        self.gate = IntentGate()
        self.guard = BoundaryGuard(self.policy)
        self.planner = TaskPlanner()
        self.ctx_mgr = ContextManager(char_budget=char_budget)
        self.trace_dir = Path(trace_dir)

    # ---------------------------------------------------------------- 主流程

    def run(self, query: str) -> AgentResult:
        trace = TraceRecorder()
        max_level = self.policy.max_level
        trace.record("request_received", query=query, max_level=max_level.name)

        # ① 输入侧闸门：裁判诱导请求直接拒绝并降级
        gate_result = self.gate.check(query)
        if not gate_result.allowed:
            trace.record("guard_decision", stage="intent_gate", allowed=False,
                         reason=gate_result.reason)
            units = [
                StructuredUnit(
                    type=OutputType.HUMAN_REVIEW_REQUIRED,
                    content="该请求要求系统作出裁判性结论，超出本系统能力边界。",
                ),
                StructuredUnit(type=OutputType.UNCERTAINTY, content=gate_result.reason),
            ]
            return self._terminate(trace, AgentResult(
                status="refused", units=units,
                message="请求已被 IntentGate 拦截（裁判诱导）。",
            ))
        trace.record("guard_decision", stage="intent_gate", allowed=True)

        # ② 受限规划：模板化静态 Plan
        plan = self.planner.create_plan(query)
        trace.record("plan_created", intent=plan.intent,
                     steps=[s.value for s in plan.steps])

        ctx = SessionContext(query=query, max_level=max_level)

        # ③ 静态执行流水线
        try:
            for step in plan.steps:
                if step in (Step.COMPOSE, Step.TERMINATE):
                    continue
                self._execute_step(step, plan, ctx, trace)
        except BoundaryViolation as e:
            trace.record("guard_decision", stage="permission", allowed=False,
                         reason=str(e))
            return self._terminate(trace, AgentResult(
                status="error", message=f"权限违规：{e}",
            ))

        # ④ 输出侧护栏 + 组装
        result = self._compose(ctx, trace)

        # ⑤ 终止（无再规划）
        return self._terminate(trace, result)

    # ---------------------------------------------------------------- 步骤执行

    def _execute_step(self, step: Step, plan: Plan, ctx: SessionContext,
                      trace: TraceRecorder) -> None:
        for tool_name in STEP_TOOL_MAP.get(step, ()):  # 只调用计划内映射的工具
            tool = self.tools[tool_name]
            self.policy.check_tool(tool.spec)  # 权限门控（越权抛 BoundaryViolation）
            trace.record("tool_called", step=step.value, tool=tool_name)
            result = tool.run({}, ctx)
            trace.record("tool_returned", step=step.value, tool=tool_name,
                         n_units=len(result.units), n_docs=len(result.documents))
            if result.documents:
                ctx.documents.extend(self.ctx_mgr.trim_documents(result.documents))
            if result.units:
                # 中间产物也先过护栏，越界单元不进入上下文
                report = self.guard.validate_units(result.units)
                for unit, reason in report.rejected:
                    trace.record("guard_decision", stage="boundary_guard",
                                 allowed=False, unit_type=unit.type.value,
                                 reason=reason)
                ctx.add_units(report.accepted)

    # ---------------------------------------------------------------- 输出组装

    def _compose(self, ctx: SessionContext, trace: TraceRecorder) -> AgentResult:
        units = list(ctx.units)

        # 来源绑定的最终确认：FACT/STATUTE/CASE 缺来源的一律降级
        report = self.guard.validate_units(units)
        final = list(report.accepted)
        for unit, reason in report.rejected:
            trace.record("guard_decision", stage="final_compose", allowed=False,
                         unit_type=unit.type.value, reason=reason)
            final.append(StructuredUnit(
                type=OutputType.HUMAN_REVIEW_REQUIRED,
                content=f"一条 {unit.type.value} 输出被拦截（{reason}），需人工复核原始材料。",
            ))

        if not final:
            final.append(StructuredUnit(
                type=OutputType.MISSING_INFORMATION,
                content="未能从现有材料中整理出有效信息，材料可能不足。",
            ))

        for u in final:
            trace.record("unit_emitted", unit=u.to_dict())
        return AgentResult(status="completed", units=final)

    # ---------------------------------------------------------------- 终止

    def _terminate(self, trace: TraceRecorder, result: AgentResult) -> AgentResult:
        trace.record("terminated", status=result.status)
        path = trace.export_jsonl(
            self.trace_dir / f"trace_{int(time.time() * 1000)}.jsonl"
        )
        result.trace_path = str(path)
        return result
