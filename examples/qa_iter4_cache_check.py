"""QA 迭代四独立验证：语料向量缓存行为实测（严过关编写）。

流程（脚本自动备份并恢复原状）：
1) 备份 data/statutes.jsonl 与 data/statute_embeddings.json；
2) 删除缓存 → 构建 VectorIndex：应重建（计时 t1，重新 embed）；
3) 缓存存在再构建：应命中缓存（计时 t2，显著更快，且文件 mtime 不变）；
4) 改动 statutes.jsonl 一条文本 → 构建：hash 校验失效应自动重建（mtime 更新）；
5) 恢复 statutes.jsonl 原样 → 构建：再次重建（恢复原缓存语义），并校验
   最终语料 hash 与备份一致。

运行（venv + LEGAL_AI_EMBED_MODEL_DIR）：
    python examples/qa_iter4_cache_check.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_layer.tools.retrieval import load_statute_corpus
from agent_layer.tools.vector_retrieval import VectorIndex

CORPUS_PATH = ROOT / "data" / "statutes.jsonl"
CACHE_PATH = ROOT / "data" / "statute_embeddings.json"
BAK_DIR = ROOT / "data" / ".qa_backup"


def corpus_hash() -> str:
    return hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def check(name: str, ok: bool, detail: str, failures: list[str]) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


def build_timed() -> tuple[VectorIndex, float]:
    t0 = time.perf_counter()
    idx = VectorIndex(load_statute_corpus())
    return idx, time.perf_counter() - t0


def main() -> int:
    failures: list[str] = []
    BAK_DIR.mkdir(exist_ok=True)
    orig_corpus_hash = corpus_hash()
    shutil.copy2(CORPUS_PATH, BAK_DIR / "statutes.jsonl.bak")
    had_cache = CACHE_PATH.exists()
    if had_cache:
        shutil.copy2(CACHE_PATH, BAK_DIR / "statute_embeddings.json.bak")
    print(f"[setup] 语料 sha256={orig_corpus_hash[:16]}... 原缓存存在={had_cache}")

    try:
        # 2) 删除缓存 → 重建
        CACHE_PATH.unlink(missing_ok=True)
        _, t1 = build_timed()
        rebuilt = CACHE_PATH.exists()
        check("1) 删缓存后重建", rebuilt, f"重建耗时 {t1:.2f}s，缓存文件已生成={rebuilt}",
              failures)

        # 3) 二次构建 → 命中缓存
        mtime1 = CACHE_PATH.stat().st_mtime_ns
        _, t2 = build_timed()
        mtime2 = CACHE_PATH.stat().st_mtime_ns
        cache_hit = mtime1 == mtime2
        check(
            "2) 二次构建命中缓存",
            cache_hit and t2 < t1,
            f"首次 {t1:.2f}s vs 命中 {t2:.3f}s（加速 {t1 / max(t2, 1e-6):.0f}x），"
            f"缓存文件未重写={cache_hit}",
            failures,
        )

        # 4) 改动语料一条文本 → hash 失效应自动重建
        lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        doc = json.loads(lines[0])
        doc["text"] += "（QA临时改动）"
        lines[0] = json.dumps(doc, ensure_ascii=False)
        CORPUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        time.sleep(0.01)
        _, t3 = build_timed()
        mtime3 = CACHE_PATH.stat().st_mtime_ns
        check(
            "3) 语料改动后缓存自动失效重建",
            mtime3 > mtime2,
            f"重建耗时 {t3:.2f}s，缓存 mtime 已更新={mtime3 > mtime2}",
            failures,
        )
    finally:
        # 5) 无论成败都恢复语料原样
        shutil.copy2(BAK_DIR / "statutes.jsonl.bak", CORPUS_PATH)
        print("[teardown] 语料文件已恢复")

    # 恢复原缓存语义：用原语料重建一次，使缓存与原语料匹配
    build_timed()
    check(
        "4) 恢复后语料 hash 一致",
        corpus_hash() == orig_corpus_hash,
        f"恢复后 sha256={corpus_hash()[:16]}...",
        failures,
    )
    # 校验最终缓存的 hash 字段与原语料指纹一致
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    idx = VectorIndex(load_statute_corpus())  # 命中缓存，秒回
    check(
        "5) 最终缓存与原语料指纹一致",
        payload.get("hash") == idx._corpus_hash(),  # noqa: SLF001
        "缓存 hash 字段 == 当前语料指纹",
        failures,
    )
    shutil.rmtree(BAK_DIR, ignore_errors=True)

    print()
    if failures:
        print(f"QA_CACHE_RESULT: FAIL（{len(failures)} 项）")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("QA_CACHE_RESULT: ALL_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
