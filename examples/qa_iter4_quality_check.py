"""QA 迭代四独立验证：向量检索质量实测（严过关编写，非工程师自报数据）。

实测内容：
a) 语义泛化：口语化查询应命中 577/563 等合同法条（BM25 做不到的泛化）；
b) 假阳性：无关查询向量路径应空结果（对照 BM25「量子计算机」已知假阳性）；
c) 门槛合理性：打印相关/无关查询得分分布，评估 min_score=0.5 分离度。

运行（venv，需 LEGAL_AI_EMBED_MODEL_DIR 指向本地模型目录）：
    python examples/qa_iter4_quality_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_layer.tools.retrieval import BM25Index, load_statute_corpus
from agent_layer.tools.vector_retrieval import DEFAULT_MIN_SCORE, VectorIndex

CORPUS = load_statute_corpus()

# (查询, 可接受命中的法条编号集合, 说明)。语料共 11 条（465/470/490/496/502/
# 509/511/563/577/584/585），无 578 预期违约条；预期违约情形归入 563(二)。
# 判定标准：向量 top5 命中编号与可接受集合有交集即算泛化成功。
SEMANTIC_CASES = [
    ("一方拖着不付款怎么办", {"577", "563", "584", "585"}, "口语化拖延付款 → 违约/解除"),
    ("对方收了钱一直不交货", {"563", "577"}, "收钱不交货 → 解除权/违约责任"),
    ("合同还没到期对方说不干了", {"563"}, "届期前明示不履行 → 563(二)"),
]

IRRELEVANT_QUERIES = ["火星殖民地", "量子计算机", "世界杯足球赛"]


def check(name: str, ok: bool, detail: str, failures: list[str]) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


def main() -> int:
    failures: list[str] = []
    vindex = VectorIndex(CORPUS)
    bm25 = BM25Index(CORPUS)

    print("=" * 70)
    print("a) 语义泛化（向量 vs BM25）")
    print("=" * 70)
    for query, expect, note in SEMANTIC_CASES:
        vhits = vindex.search(query, top_k=5)
        vids = {doc["id"] for doc, _ in vhits}
        # id 形如 statute-577 → 法条号 577
        vnums = {i.split("-")[-1] for i in vids}
        bhits = [(doc["id"], s) for doc, s in bm25.search(query, 5) if s > 0]
        vscores = {doc["id"].split("-")[-1]: f"{s:.3f}" for doc, s in vhits}
        ok = bool(expect & vnums)
        check(
            f"语义泛化「{query}」",
            ok,
            f"{note} | 向量命中编号={sorted(vnums)} 得分={vscores} | "
            f"BM25命中={[i for i, _ in bhits]}",
            failures,
        )

    print()
    print("=" * 70)
    print("b) 假阳性（向量应空；BM25 对照已知「量子计算机」假阳性）")
    print("=" * 70)
    for query in IRRELEVANT_QUERIES:
        vhits = vindex.search(query, top_k=5)
        # 顺便记录向量侧最高分（未过门槛的原始分）
        raw = vindex.search(query, top_k=5)  # search 已过滤，重算原始分
        q = vindex._get_embedder().embed_query(query)  # noqa: SLF001
        from agent_layer.tools.vector_retrieval import _dot, _normalize

        qn = _normalize(q)
        topscore = max(_dot(qn, v) for v in vindex._embeddings)  # noqa: SLF001
        bhits = [(doc["id"], round(s, 1)) for doc, s in bm25.search(query, 5) if s > 0]
        check(
            f"无关查询「{query}」",
            len(vhits) == 0,
            f"向量命中数={len(vhits)}（原始最高分={topscore:.3f} < 门槛 "
            f"{DEFAULT_MIN_SCORE}）| BM25命中={bhits}",
            failures,
        )

    print()
    print("=" * 70)
    print("c) 门槛合理性：相关/无关查询原始得分分布")
    print("=" * 70)
    from agent_layer.tools.vector_retrieval import _dot, _normalize

    emb = vindex._get_embedder()  # noqa: SLF001
    relevant = [q for q, _, _ in SEMANTIC_CASES] + [
        "解除合同 违约责任",
        "格式条款 提示说明义务",
    ]
    irrelevant = IRRELEVANT_QUERIES + ["怎么做红烧肉", "下周天气如何"]
    rel_min = 1.0
    irr_max = 0.0
    print(f"{'类别':<6}{'查询':<22}{'最高分':>8}  命中法条号")
    for tag, queries in (("相关", relevant), ("无关", irrelevant)):
        for query in queries:
            qn = _normalize(emb.embed_query(query))
            scores = sorted(
                ((doc["id"].split("-")[-1], _dot(qn, v))
                 for doc, v in zip(vindex.documents, vindex._embeddings)),  # noqa: SLF001
                key=lambda x: x[1], reverse=True,
            )
            top_id, top_score = scores[0]
            if tag == "相关":
                rel_min = min(rel_min, top_score)
            else:
                irr_max = max(irr_max, top_score)
            print(f"{tag:<6}{query:<22}{top_score:>8.3f}  {top_id}")
    gap_ok = rel_min > DEFAULT_MIN_SCORE > irr_max
    check(
        "门槛分离度",
        gap_ok,
        f"相关最低分={rel_min:.3f} > {DEFAULT_MIN_SCORE} > 无关最高分={irr_max:.3f}",
        failures,
    )

    print()
    print("=" * 70)
    if failures:
        print(f"QA_QUALITY_RESULT: FAIL（{len(failures)} 项）")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("QA_QUALITY_RESULT: ALL_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
