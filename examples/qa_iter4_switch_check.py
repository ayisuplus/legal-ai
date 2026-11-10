"""QA 迭代四独立验证：降级开关三态实测（严过关编写）。

a) LEGAL_AI_RETRIEVAL_MODE=bm25：行为与迭代三基线一致（含「量子计算机」
   BM25 已知假阳性），且不触碰向量路径（stderr 无任何向量警告）；
b) LEGAL_AI_RETRIEVAL_MODE=vector + 错误 MODEL_DIR：初始化失败必须抛错，
   绝不静默降级；
c) 非法 MODE 值：stderr 告警并按 auto 处理。

运行（venv）：
    set LEGAL_AI_EMBED_MODEL_DIR=<本地模型目录>   # 仅 a/c 用，b 内部覆盖
    python examples/qa_iter4_switch_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_layer.core.context import SessionContext
from agent_layer.core.types import CapabilityLevel
from agent_layer.tools.implementations import StatuteSearchTool


def _ctx(query: str) -> SessionContext:
    return SessionContext(query=query, max_level=CapabilityLevel.L2_ANALYSIS)


def check(name: str, ok: bool, detail: str, failures: list[str]) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


def main() -> int:
    failures: list[str] = []

    # --- a) bm25 强制模式 -------------------------------------------------
    os.environ["LEGAL_AI_RETRIEVAL_MODE"] = "bm25"
    tool = StatuteSearchTool()
    r = tool.run({"keywords": ["量子计算机"]}, _ctx("量子计算机"))
    texts = [u.content for u in r.units]
    has_fp = any("违约金" in t for t in texts)  # BM25 已知假阳性：statute-585
    check(
        "a1) bm25 模式保留已知假阳性",
        has_fp,
        f"「量子计算机」命中 statute-585 违约金={has_fp}（与迭代三基线一致）",
        failures,
    )
    r2 = tool.run({"keywords": ["解除", "合同"]}, _ctx("解除合同"))
    check(
        "a2) bm25 模式正常检索",
        any(u.type.name == "STATUTE" for u in r2.units),
        f"「解除 合同」返回 {len(r2.units)} 个单元",
        failures,
    )
    check(
        "a3) bm25 模式未初始化向量索引",
        tool._vector_attempted is False and tool._vector_index is None,  # noqa: SLF001
        f"_vector_attempted={tool._vector_attempted}",  # noqa: SLF001
        failures,
    )

    # --- b) vector 强制模式 + 坏模型目录 → 必须抛错 ------------------------
    os.environ["LEGAL_AI_RETRIEVAL_MODE"] = "vector"
    os.environ["LEGAL_AI_EMBED_MODEL_DIR"] = "E:\\projects\\法律AI\\data\\models\\不存在目录"
    tool_v = StatuteSearchTool()
    raised = False
    try:
        tool_v.run({"keywords": ["合同"]}, _ctx("合同纠纷"))
    except Exception as e:  # noqa: BLE001
        raised = True
        print(f"      捕获异常（符合预期）：{type(e).__name__}: {str(e)[:120]}")
    check("b1) vector 模式坏目录抛错不静默降级", raised, f"raised={raised}", failures)

    # 坏语料缓存路径场景下 vector 正常目录应可工作（对照，确认 vector 本身可用）
    os.environ["LEGAL_AI_EMBED_MODEL_DIR"] = os.environ.get(
        "GOOD_MODEL_DIR",
        r"E:\projects\法律AI\data\models\bge-small-zh-v1.5",
    )
    tool_v2 = StatuteSearchTool()
    try:
        r3 = tool_v2.run({"keywords": ["火星殖民地"]}, _ctx("火星殖民地"))
        check(
            "b2) vector 模式无关查询 → MISSING_INFORMATION",
            len(r3.units) == 1
            and r3.units[0].type.name == "MISSING_INFORMATION",
            f"units={[u.type.name for u in r3.units]}",
            failures,
        )
    except Exception as e:  # noqa: BLE001
        check("b2) vector 模式无关查询 → MISSING_INFORMATION", False,
              f"意外异常 {type(e).__name__}: {e}", failures)

    # --- c) 非法 MODE 值 → 告警 + 按 auto ----------------------------------
    import contextlib
    import io

    os.environ["LEGAL_AI_RETRIEVAL_MODE"] = "bogus_mode"
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        tool_c = StatuteSearchTool()
    warned = "未知检索模式" in buf.getvalue()
    check(
        "c1) 非法 MODE 告警并按 auto",
        warned and tool_c.retrieval_mode == "auto",
        f"告警={warned} mode={tool_c.retrieval_mode}",
        failures,
    )

    print()
    if failures:
        print(f"QA_SWITCH_RESULT: FAIL（{len(failures)} 项）")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("QA_SWITCH_RESULT: ALL_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
