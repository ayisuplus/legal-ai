"""边界守护：输入侧 IntentGate + 输出侧 BoundaryGuard。

对应启动书 4.3 / 十章 / 十一章：
- IntentGate：裁判诱导型请求在进入流水线前即被拒绝并降级；
- BoundaryGuard：输出单元三重校验（类型白名单、来源绑定、裁判语言扫描）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .permissions import PermissionPolicy
from .types import OutputType, StructuredUnit, SOURCE_REQUIRED_TYPES


#: 裁判诱导型请求模式（输入侧）。命中即拒绝，不进入流水线。
#:
#: 防误伤设计原则：不裸匹配「判决/判处/定罪/量刑/刑事责任」等词——
#: 检索、整理类请求（L0/L1 能力）合法包含这些词，如「查找关于判决执行的第X条」
#: 「检索量刑指导意见相关法条」「整理判决书的时间线」。
#: 每条模式必须带有以下特征之一才命中：
#:   a) 请求系统作出裁判的句式特征：该/怎么/如何/吗/呢/是否/多少/合理吗 等；
#:   b) 对具体个案对象的裁判请求：他/她/被告/张三 等指代 + 裁判动词。
INDUCING_PATTERNS = [
    r"判(几年|多久|多重)",
    r"谁有罪|是否有罪|有罪吗",
    r"构成犯罪|构成什么罪",
    r"谁.{0,4}(承担责任|负责)|责任.{0,4}怎么分|责任比例",
    r"能赢吗|胜诉(率|概率|可能)",
    r"该(怎么判|判多少|怎么处罚)",
    r"证人.{0,4}(可不可信|可信度|说谎)",
    r"直接告诉我谁(对|错|有理)",
    # 「判决/判处 + 个案结论」：判决张三有罪、判决他败诉、判他胜诉。
    # 防误伤：结论词（有罪/无罪/胜诉/败诉）为必需，「判决书送达」「判决执行」不命中。
    r"判处?.{0,6}(有罪|无罪|败诉|胜诉)",
    # 「判处 + 刑种 + 求评判」：判处死刑是否合理、判处三年合适吗。
    # 防误伤：限定「判处」（不含「判决」）且须带评判后缀，裸「判处死刑」不拦
    # （可能是法条检索）；「审查判决是否合理的标准」因非「判处」不命中。
    r"判处.{0,10}(是否合理|是否合适|合理吗|合适吗|过重吗|是否过重)",
    # 「该如何判决」变体：该如何判决、该怎样判处。
    # （现有「该(怎么判|...)」以子串方式已覆盖「该怎么判决」，此处只补「如何/怎样」。）
    r"该(如何|怎样)判(决|处)?",
    # 「定罪请求」：定什么罪、定何罪。
    r"定(什么|啥|何)罪",
    # 「对个案对象定罪」：给他定罪、对被告定罪。
    # 防误伤：限于个案指代词前缀；「定罪的标准/要件」等研究性表述无此前缀，不命中。
    r"(给|对)(他|她|其|被告|对方|当事人|张三|李四|某人).{0,4}定罪",
    # 「是否构成定罪」：是否构成定罪、是否能定罪。
    r"是否(构成|能|可)定罪",
    # 「量刑请求」：怎么量刑、如何量刑。
    r"(怎么|如何|怎样)量刑",
    # 「量刑多少」：量刑多少、量刑几年。
    r"量刑(多少|几年|多久|多重)",
    # 「索要量刑建议」：给个量刑建议、提出量刑建议。
    # 防误伤：须带索取动词（给/提出/出具/做），「检索量刑指导意见」「量刑建议制度
    # 的规定」不命中。
    r"(给|提出|出具|做).{0,3}量刑建议",
    # 「刑事责任归属疑问」：谁承担刑事责任、谁负刑事责任。
    # （现有「谁.{0,4}承担责任」不含「刑事」插字，此处显式补齐。）
    r"谁.{0,4}(承担|负).{0,2}刑事责任",
    # 「负刑事责任吗」：他需要负刑事责任吗、承担刑事责任吗。
    # 防误伤：疑问语气词（吗/呢）必需，「承担刑事责任的构成要件」不命中。
    r"(承担|负)刑事责任(吗|呢)",
    # 「是否承担刑事责任」：张三是否应当承担刑事责任、是否需要负刑事责任。
    r"是否(需要|应当|应该|要)?(承担|负).{0,2}刑事责任",
    # 「判处罚」：该判什么处罚、判何种处罚（补全现有「该怎么处罚」未覆盖的组合）。
    r"判(什么|何种)处罚",
    # 「坐牢/判刑疑问」：他会坐牢吗、会被判刑吗。
    # 防误伤：语气词必需，「会判刑的情形有哪些」不命中。
    r"(坐牢|被判刑|判刑)(吗|呢)",
    # 「缓刑请求」：能判缓刑吗、可以适用缓刑吗（语气词必需）。
    r"(能|可以|可能).{0,4}缓刑(吗|呢)",
]

#: 裁判性语言模式（输出侧）。单元正文命中即拒绝该单元。
JUDGMENT_PATTERNS = [
    r"判处.{0,10}(有期徒刑|拘役|管制|死刑|罚金)",
    r"构成.{0,6}罪",
    r"应当承担.{0,10}责任",
    r"建议(法院)?(判处|判决|支持|驳回)",
    r"有罪|无罪释放",
    r"胜诉率|胜诉概率.{0,6}\d",
    r"责任比例.{0,6}\d",
    r"证人.{0,6}不可信",
]


@dataclass
class GateResult:
    allowed: bool
    reason: str = ""


class IntentGate:
    """输入侧闸门：识别裁判诱导型请求。"""

    def check(self, query: str) -> GateResult:
        for pattern in INDUCING_PATTERNS:
            if re.search(pattern, query):
                return GateResult(
                    allowed=False,
                    reason=(
                        f"请求命中裁判诱导模式 /{pattern}/。"
                        "本系统不提供裁判性结论（Level 4 能力未开放），"
                        "该问题需要由具备法定职权的人员判断。"
                    ),
                )
        return GateResult(allowed=True)


@dataclass
class GuardReport:
    accepted: list[StructuredUnit] = field(default_factory=list)
    rejected: list[tuple[StructuredUnit, str]] = field(default_factory=list)


class BoundaryGuard:
    """输出侧护栏：对每一个结构化单元做三重校验。"""

    def __init__(self, policy: PermissionPolicy):
        self.policy = policy

    def validate_unit(self, unit: StructuredUnit) -> str | None:
        """返回 None 表示通过；否则返回拒绝原因。"""
        # 1) 类型白名单 + 等级门控（裁判类型在枚举中不存在，天然免疫）
        if not isinstance(unit.type, OutputType):
            return f"非法输出类型 {unit.type!r}"
        if not self.policy.check_output_type(unit.type):
            return f"输出类型 {unit.type.value} 超出当前会话权限等级"
        # 2) 来源绑定（可追溯原则）
        if unit.type in SOURCE_REQUIRED_TYPES and not unit.source:
            return f"{unit.type.value} 单元必须绑定来源（source）"
        # 3) 裁判性语言扫描
        for pattern in JUDGMENT_PATTERNS:
            if re.search(pattern, unit.content):
                return f"单元内容命中裁判性语言 /{pattern}/，涉嫌越界"
        return None

    def validate_units(self, units: list[StructuredUnit]) -> GuardReport:
        report = GuardReport()
        for unit in units:
            reason = self.validate_unit(unit)
            if reason is None:
                report.accepted.append(unit)
            else:
                report.rejected.append((unit, reason))
        return report
