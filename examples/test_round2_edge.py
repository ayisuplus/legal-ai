"""QA Round 2 边界复测（严过关）：Bug #1 修复有效性 + 检索质量未劣化 + 边界情形。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_layer.tools.implementations import StatuteSearchTool
from agent_layer.tools.retrieval import BM25Index, load_statute_corpus
from agent_layer.core.context import SessionContext

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


tool = StatuteSearchTool()
idx = BM25Index(load_statute_corpus())

# 1. 新一批无关查询（与法条语料无任何词汇重叠）：必须输出 MISSING_INFORMATION
for q in ("世界杯足球赛", "恐龙灭绝", "菜谱红烧肉"):
    ctx = SessionContext(query=q, max_level=0)
    r = tool.run({"keywords": [q]}, ctx)
    check(f"无关查询 {q!r} → MISSING_INFORMATION 且无文档",
          [u.type.value for u in r.units] == ["MISSING_INFORMATION"]
          and not r.documents,
          f"types={[u.type.value for u in r.units]}, n_docs={len(r.documents)}")

# 1b. 词汇重叠但主题无关的查询（"量子计算机" 与第585条共享 bigram"计算"）：
#     纯词面 BM25 的固有假阳性，允许返回，但必须是不伪造的真实语料且绑定 source。
corpus_ids = {d["id"] for d in load_statute_corpus()}
ctx = SessionContext(query="量子计算机", max_level=0)
r = tool.run({"keywords": ["量子计算机"]}, ctx)
check("词汇重叠查询返回结果不伪造（真实语料 + 绑定 source）",
      all(d["id"] in corpus_ids for d in r.documents)
      and all(u.source for u in r.units if u.type.value == "STATUTE"),
      f"n_docs={len(r.documents)}")

# 2. 正常查询检索质量未劣化（top-1 命中预期法条）
expectations = {
    "解除合同": "statute-563",
    "违约金": "statute-585",
    "迟延履行": "statute-563",  # 563 含两处"迟延履行"
    "格式条款": "statute-496",
}
for q, want in expectations.items():
    hits = [(d["id"], round(s, 2)) for d, s in idx.search(q, 3) if s > 0]
    check(f"正常查询 {q!r} top-1 == {want}",
          hits and hits[0][0] == want, f"hits={hits}")

# 3. 单字查询：按声称的退化逻辑工作（有结果而非永远空）
hits = [(d["id"], round(s, 2)) for d, s in idx.search("债", 3) if s > 0]
check("单字查询 '债' 退化为单字命中（有结果）", len(hits) > 0, f"hits={hits[:3]}")

# 4. 中英文混合查询不崩溃
hits_mixed = [(d["id"], round(s, 2)) for d, s in idx.search("合同 contract 2025", 3) if s > 0]
check("中英文混合查询不崩溃", isinstance(hits_mixed, list), f"hits={hits_mixed[:2]}")

# 5. 空查询 / 纯标点 / 极长查询不崩溃
try:
    r_empty = idx.search("", 3)
    r_punct = idx.search("！！！。。。", 3)
    r_long = idx.search("合同" * 5000, 3)
    check("空/标点/极长查询不崩溃",
          r_empty == [] and isinstance(r_punct, list) and isinstance(r_long, list),
          f"empty={r_empty}, punct_n={len(r_punct)}, long_n={len(r_long)}")
except Exception as e:  # noqa: BLE001
    check("空/标点/极长查询不崩溃", False, f"异常：{e!r}")

print("\nROUND2_EDGE_RESULT:", "ALL_PASS" if ok else "HAS_FAILURE")
sys.exit(0 if ok else 1)
