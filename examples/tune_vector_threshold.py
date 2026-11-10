"""向量检索 min_score 调参辅助脚本（一次性工具，非回归资产）。

以 min_score=0 打印各典型查询对全部法条的 cosine 得分分布，
为 DEFAULT_MIN_SCORE 取值提供实测依据。仅 venv python 可跑。

运行：python examples/tune_vector_threshold.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_layer.tools.retrieval import load_statute_corpus
from agent_layer.tools.vector_retrieval import VectorIndex

QUERIES = [
    "解除合同 违约责任",
    "一方拖着不付款怎么办",
    "火星殖民地",
    "量子计算机",
    "违约金约定过高能调整吗",
    "合同什么时候生效",
]


def main() -> None:
    docs = load_statute_corpus()
    index = VectorIndex(docs, min_score=0.0)
    for q in QUERIES:
        hits = index.search(q, top_k=len(docs))
        print(f"\n查询：{q}")
        for doc, score in hits[:5]:
            print(f"  {score:.4f}  {doc['id']}  {doc['source'][:22]}")


if __name__ == "__main__":
    main()
