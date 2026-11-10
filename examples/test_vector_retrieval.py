"""向量检索测试：功能 / 降级 / 缓存（迭代四新增，双解释器兼容）。

验证项：
A. 向量路径功能（需 fastembed，裸 python 自动 skip）：
   1. 关键词查询「解除合同 违约责任」top 结果含民法典 563/577；
   2. 语义查询「一方拖着不付款怎么办」命中 563/577（语义泛化优于词面）；
   3. 无关查询「火星殖民地」不返回法条（MISSING_INFORMATION）；
   4. 词面重叠但语义无关的「量子计算机」不像 BM25 那样假阳性命中 585。
B. 降级路径（两种解释器都跑）：
   5. LEGAL_AI_RETRIEVAL_MODE=bm25 时行为与迭代三基线一致；
   6. auto 模式模拟 fastembed 导入失败：stderr 留痕 + 自动降级 BM25
      + 每实例只尝试一次（重复检索不重复告警）；
   7. vector 强制模式在依赖缺失时抛错（不静默降级）。
C. 语料向量缓存（注入假 embedder，纯标准库，两种解释器都跑）：
   8. 首次构建生成缓存文件；二次构建加载缓存不重复 embed（embed 调用数为 0）；
   9. 语料改动后 hash 不匹配、缓存失效自动重建；缓存文件损坏时告警并重建。
D. 真实模型缓存加载（需 fastembed）：二次构建不改写缓存文件（mtime 不变）。

运行：python examples/test_vector_retrieval.py
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_layer.core.context import SessionContext
from agent_layer.tools.implementations import StatuteSearchTool
from agent_layer.tools.retrieval import load_statute_corpus
from agent_layer.tools.vector_retrieval import VectorIndex

#: fastembed 可用性：决定真实模型用例是否执行（裸 python 自动跳过）
HAS_FASTEMBED = importlib.util.find_spec("fastembed") is not None

RESULTS: list[tuple[str, bool, str]] = []
SKIPPED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}"
          + (f"  -- {detail}" if detail else ""))


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"SKIP  {name}  -- {reason}")


def require_fastembed(name: str) -> bool:
    """真实模型用例门禁：无 fastembed 时记录 skip 并返回 False。"""
    if not HAS_FASTEMBED:
        skip(name, "当前解释器无 fastembed，跳过向量用例")
        return False
    return True


class _HashEmbedder:
    """确定性假嵌入器（纯标准库）：按字符散列到固定维度向量。

    用于缓存逻辑测试——不依赖 fastembed，保证两种解释器下行为一致；
    同时用 embed_calls 计数，可断言「二次构建未重复 embed 语料」。
    """

    DIM = 64

    def __init__(self) -> None:
        self.embed_calls = 0  # 累计 embed 的文本条数（passage + query）

    def _embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        vec = [0.0] * self.DIM
        for ch in text:
            if ch.strip():
                vec[hash(ch) % self.DIM] += 1.0
        return vec

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _vector_ctx(query: str) -> SessionContext:
    return SessionContext(query=query, max_level=0)


# ---------------------------------------------------------------- A. 向量功能
def test_vector_functional() -> None:
    if not require_fastembed("A 向量路径功能（4 项断言）"):
        return
    tool = StatuteSearchTool(retrieval_mode="vector")

    # A1. 关键词查询 top 结果含 563/577
    r = tool.run({"keywords": ["解除合同", "违约责任"]},
                 _vector_ctx("解除合同 违约责任"))
    ids = [d["id"] for d in r.documents]
    check(
        "A1 向量：关键词查询 top 结果含 563 与 577",
        "statute-563" in ids and "statute-577" in ids,
        f"ids={ids}",
    )
    check("A1 向量：top-1 为 563", ids[:1] == ["statute-563"],
          f"top1={ids[:1]}")

    # A2. 语义查询（词面与法条几乎不重叠）命中 563/577
    r2 = tool.run({"keywords": ["一方拖着不付款怎么办"]},
                  _vector_ctx("一方拖着不付款怎么办"))
    ids2 = [d["id"] for d in r2.documents]
    check(
        "A2 向量：语义查询「一方拖着不付款怎么办」命中 563/577",
        bool({"statute-563", "statute-577"} & set(ids2)),
        f"ids={ids2}",
    )

    # A3. 无关查询不返回法条
    r3 = tool.run({"keywords": ["火星殖民地"]}, _vector_ctx("火星殖民地"))
    types = [u.type.value for u in r3.units]
    check(
        "A3 向量：「火星殖民地」输出 MISSING_INFORMATION 且不返回法条",
        types == ["MISSING_INFORMATION"] and not r3.documents,
        f"types={types}, n_docs={len(r3.documents)}",
    )

    # A4. 词面重叠假阳性对比：BM25 对「量子计算机」的命中情况（观察）vs
    #     向量路径不得命中 585
    bm25_tool = StatuteSearchTool(retrieval_mode="bm25")
    rb = bm25_tool.run({"keywords": ["量子计算机"]}, _vector_ctx("量子计算机"))
    bm25_ids = [d["id"] for d in rb.documents]
    rv = tool.run({"keywords": ["量子计算机"]}, _vector_ctx("量子计算机"))
    vec_ids = [d["id"] for d in rv.documents]
    check(
        "A4 向量：「量子计算机」不假阳性命中 585"
        f"（对照：BM25 命中 {bm25_ids or '无'}）",
        "statute-585" not in vec_ids,
        f"vector_ids={vec_ids}",
    )


# ---------------------------------------------------------------- B. 降级
def test_bm25_forced_mode_baseline() -> None:
    tool = StatuteSearchTool(retrieval_mode="bm25")
    check("B5 强制 bm25：模式标识正确", tool.retrieval_mode == "bm25")
    r = tool.run({"keywords": ["解除合同", "违约责任"]},
                 _vector_ctx("解除合同 违约责任"))
    top = r.documents[0] if r.documents else {}
    check("B5 强制 bm25：top-1 命中 563（与迭代三基线一致）",
          top.get("id") == "statute-563", f"top1={top.get('id')!r}")
    r2 = tool.run({"keywords": ["火星殖民地"]}, _vector_ctx("火星殖民地"))
    check("B5 强制 bm25：无关查询输出 MISSING_INFORMATION",
          [u.type.value for u in r2.units] == ["MISSING_INFORMATION"])


def test_auto_degrade_on_import_failure() -> None:
    """sys.modules 注入 fastembed=None 模拟缺依赖：auto 降级 + 留痕 + 只试一次。"""
    saved = sys.modules.get("fastembed", "_ABSENT")
    sys.modules["fastembed"] = None  # import 时触发 ImportError
    try:
        tool = StatuteSearchTool(retrieval_mode="auto")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            r1 = tool.run({"keywords": ["解除合同", "违约责任"]},
                          _vector_ctx("解除合同 违约责任"))
            r2 = tool.run({"keywords": ["违约金"]}, _vector_ctx("违约金"))
        err = buf.getvalue()
        top = r1.documents[0] if r1.documents else {}
        check("B6 auto 降级：缺依赖时不崩溃且结果等同 BM25（top-1=563）",
              top.get("id") == "statute-563", f"top1={top.get('id')!r}")
        check("B6 auto 降级：stderr 留痕", "[警告]" in err,
              f"stderr_head={err[:80]!r}")
        check("B6 auto 降级：每实例只尝试一次（二次检索不重复告警）",
              err.count("[警告]") == 1 and r2.documents,
              f"warnings={err.count('[警告]')}")
    finally:
        if saved == "_ABSENT":
            sys.modules.pop("fastembed", None)
        else:
            sys.modules["fastembed"] = saved


def test_vector_forced_mode_raises_without_dependency() -> None:
    saved = sys.modules.get("fastembed", "ABSENT")
    sys.modules["fastembed"] = None
    try:
        tool = StatuteSearchTool(retrieval_mode="vector")
        raised = False
        try:
            tool.run({"keywords": ["解除合同"]}, _vector_ctx("解除合同"))
        except Exception as e:
            raised = True
            detail = f"{type(e).__name__}"
        check("B7 强制 vector：依赖缺失时抛错（不静默降级）", raised,
              detail if raised else "")
    finally:
        if saved == "ABSENT":
            sys.modules.pop("fastembed", None)
        else:
            sys.modules["fastembed"] = saved


def test_invalid_mode_falls_back_to_auto() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        tool = StatuteSearchTool(retrieval_mode="banana")
    check("B 附加：非法模式告警并按 auto 处理",
          tool.retrieval_mode == "auto" and "[警告]" in buf.getvalue())


# ---------------------------------------------------------------- C. 缓存
def test_embedding_cache() -> None:
    docs = load_statute_corpus()
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "emb.json"

        # C8a. 首次构建：embed 全部语料并生成缓存文件
        e1 = _HashEmbedder()
        VectorIndex(docs, model_name="fake-model", cache_path=store,
                    min_score=0.0, embedder=e1)
        check("C8 缓存：首次构建 embed 全部语料且生成缓存文件",
              e1.embed_calls == len(docs) and store.exists(),
              f"embed_calls={e1.embed_calls}, n_docs={len(docs)}")

        # C8b. 二次构建：加载缓存，不重复 embed（构造后立即断言，先于任何
        #      查询 embed）；检索可用性用「原文自检索 top-1 即自身」验证——
        #      该性质对任意确定性 embedder 成立，不受假 embedder 排序影响
        e2 = _HashEmbedder()
        idx2 = VectorIndex(docs, model_name="fake-model", cache_path=store,
                           min_score=0.0, embedder=e2)
        check("C8 缓存：二次构建零 embed（直接加载缓存）",
              e2.embed_calls == 0, f"embed_calls={e2.embed_calls}")
        doc563 = next(d for d in docs if d["id"] == "statute-563")
        hits = idx2.search(doc563["text"], top_k=1)
        check("C8 缓存：缓存索引检索可用（563 原文自检索 top-1 为自身）",
              bool(hits) and hits[0][0]["id"] == "statute-563",
              f"top1={hits[0][0]['id'] if hits else None!r}")

        # C9a. 语料改动：hash 不匹配 → 缓存失效重建
        changed = docs + [{"id": "statute-x", "source": "《测试法》第一条",
                           "text": "测试新增法条文本"}]
        e3 = _HashEmbedder()
        VectorIndex(changed, model_name="fake-model", cache_path=store,
                    min_score=0.0, embedder=e3)
        check("C9 缓存：语料改动后自动重建（重新 embed）",
              e3.embed_calls == len(changed),
              f"embed_calls={e3.embed_calls}")

        # C9b. 缓存文件损坏：stderr 告警并重建，不崩溃
        store.write_text("{损坏的JSON", encoding="utf-8")
        e4 = _HashEmbedder()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            VectorIndex(docs, model_name="fake-model", cache_path=store,
                        min_score=0.0, embedder=e4)
        check("C9 缓存：缓存损坏时告警并重建",
              "[警告]" in buf.getvalue() and e4.embed_calls == len(docs),
              f"embed_calls={e4.embed_calls}")


def test_real_model_cache_reload() -> None:
    if not require_fastembed("D 真实模型缓存二次加载"):
        return
    docs = load_statute_corpus()
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "emb.json"
        VectorIndex(docs, cache_path=store)  # 首次：真实 embed + 写缓存
        mtime1 = store.stat().st_mtime_ns
        idx2 = VectorIndex(docs, cache_path=store)  # 二次：应直接加载缓存
        mtime2 = store.stat().st_mtime_ns
        hits = idx2.search("解除合同 违约责任", top_k=3)
        check("D 真实模型：二次构建加载缓存（缓存文件不改写）",
              mtime1 == mtime2, f"mtime_changed={mtime1 != mtime2}")
        check("D 真实模型：缓存索引检索 top-1 为 563",
              bool(hits) and hits[0][0]["id"] == "statute-563",
              f"top1={hits[0][0]['id'] if hits else None!r}")


if __name__ == "__main__":
    test_vector_functional()
    test_bm25_forced_mode_baseline()
    test_auto_degrade_on_import_failure()
    test_vector_forced_mode_raises_without_dependency()
    test_invalid_mode_falls_back_to_auto()
    test_embedding_cache()
    test_real_model_cache_reload()

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"解释器：{sys.executable}")
    print(f"fastembed 可用：{HAS_FASTEMBED}")
    print(f"合计 {len(RESULTS)} 项，通过 {len(RESULTS) - len(failed)}，"
          f"失败 {len(failed)}，跳过 {len(SKIPPED)} 组")
    if failed:
        print("失败项：", *failed, sep="\n  - ")
    sys.exit(1 if failed else 0)
