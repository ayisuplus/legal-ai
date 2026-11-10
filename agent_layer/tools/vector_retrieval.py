"""本地向量检索引擎：fastembed 句向量 + cosine 相似度 + 语料向量缓存。

设计意图与边界约束：
- 这是「纯标准库」铁律的唯一例外：语义检索需要句向量模型，用户已明确允许
  为此引入 fastembed（ONNX 运行时，无 torch，模型量化版约 50MB，首次联网
  下载后缓存到本地，之后完全离线）。例外被严格限制在本模块内部：
  fastembed 一律延迟导入（函数/方法内部 import），裸 python（无 fastembed）
  import 本模块绝不崩溃，由 StatuteSearchTool 的降级路径兜底；
- 检索路径与 LLM Fallback 同风格：向量为主、BM25 降级，任何失败留痕、
  绝不伪造检索结果；
- 不依赖 numpy 做持久化与相似度计算（缓存用 JSON、相似度用纯 python 点积），
  保证本模块自身的数学逻辑在裸 python 下也可测试（可注入假 embedder）。

环境变量：
- LEGAL_AI_EMBED_CACHE：fastembed 模型缓存目录（默认用 fastembed 自身缓存目录）；
- LEGAL_AI_EMBED_MODEL_DIR：本地模型目录（含 model_optimized.onnx / tokenizer.json
  等文件）。设置后经 fastembed specific_model_path 直接加载，完全离线、
  不做任何网络请求——供无法访问 HuggingFace 的环境手工放置模型文件；
- LEGAL_AI_EMBED_STORE ：语料向量缓存文件路径（默认 data/statute_embeddings.json）。

缓存一致性：缓存文件记录语料内容 hash（sha256(id+text) 串联）与模型名，
加载时逐项校验；语料或模型任一变化即自动重建，绝不使用过期向量。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Protocol

#: 默认句向量模型：fastembed 内置支持的中文句向量，dim=512，量化版约 50MB
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

#: 默认相关度门槛（cosine 相似度）。取值依据：实测「解除合同 违约责任」等
#: 相关查询得分 ≥ 0.55，「火星殖民地」「量子计算机」等无关查询最高分 < 0.5，
#: 取中间值 0.5 作为门槛（调参记录见 docs/agent_layer_architecture.md 6.6 节）。
DEFAULT_MIN_SCORE = 0.5

#: 默认语料向量缓存路径（agent_layer/tools/vector_retrieval.py → 项目根/data/）
_DEFAULT_STORE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "statute_embeddings.json"
)

#: 缓存文件格式版本（结构变化时递增，旧缓存自动失效重建）
_CACHE_VERSION = 1


class Embedder(Protocol):
    """句向量嵌入器协议：隔离 fastembed 依赖，便于测试注入假实现。"""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """对语料文档批量嵌入（passage 侧）。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """对查询文本嵌入（query 侧，模型可加检索指令前缀）。"""
        ...


class _FastembedEmbedder:
    """fastembed 真实嵌入器：构造时延迟导入 fastembed 并加载模型（懒初始化）。"""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        # 延迟导入：缺依赖时在此处抛 ImportError，由调用方捕获降级
        from fastembed import TextEmbedding

        kwargs: dict = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        # 本地模型目录优先：完全离线加载，不触发任何网络请求
        model_dir = os.environ.get("LEGAL_AI_EMBED_MODEL_DIR")
        if model_dir:
            kwargs["specific_model_path"] = model_dir
        self._model = TextEmbedding(**kwargs)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # fastembed 返回 numpy 数组迭代器；转纯 python list 以便 JSON 持久化
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        # query_embed 对 bge 系列自动加检索指令前缀，提升非对称检索质量
        return next(iter(self._model.query_embed([text]))).tolist()


def _normalize(vec: list[float]) -> list[float]:
    """L2 归一化；零向量原样返回（点积即 cosine）。"""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    """两个已归一化向量的点积（即 cosine 相似度）。"""
    return sum(x * y for x, y in zip(a, b))


class VectorIndex:
    """对 {id, source, text} 文档集合构建句向量索引并提供 cosine top_k 检索。

    参数：
    - documents：语料文档列表（{id, source, text}）；
    - model_name：fastembed 模型名（默认 BAAI/bge-small-zh-v1.5）；
    - min_score：cosine 相关度门槛，低于门槛的结果不返回；
    - cache_path：语料向量缓存文件路径（None 时读 LEGAL_AI_EMBED_STORE，
      再缺省用项目 data/statute_embeddings.json）；
    - model_cache_dir：fastembed 模型缓存目录（None 时读 LEGAL_AI_EMBED_CACHE，
      再缺省用 fastembed 自身默认目录）；
    - embedder：可选注入的 Embedder 实现（测试用）；None 时懒加载 fastembed。
    """

    def __init__(
        self,
        documents: list[dict],
        *,
        model_name: str = DEFAULT_MODEL,
        min_score: float = DEFAULT_MIN_SCORE,
        cache_path: str | Path | None = None,
        model_cache_dir: str | None = None,
        embedder: Embedder | None = None,
    ):
        if not documents:
            raise ValueError("VectorIndex 需要非空语料")
        self.documents = list(documents)
        self.model_name = model_name
        self.min_score = min_score
        self.cache_path = Path(
            cache_path
            or os.environ.get("LEGAL_AI_EMBED_STORE")
            or _DEFAULT_STORE_PATH
        )
        self.model_cache_dir = (
            model_cache_dir or os.environ.get("LEGAL_AI_EMBED_CACHE") or None
        )
        self._embedder = embedder
        self._embeddings: list[list[float]] = []
        self._build_or_load()

    # ------------------------------------------------------------ 缓存

    def _corpus_hash(self) -> str:
        """语料内容指纹：id 与 text 串联的 sha256（顺序敏感）。"""
        h = hashlib.sha256()
        for doc in self.documents:
            h.update(doc.get("id", "").encode("utf-8"))
            h.update(b"\x00")
            h.update(doc.get("text", "").encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _read_cache(self) -> list[list[float]] | None:
        """读取并校验缓存；任何不一致/损坏返回 None（调用方负责重建）。"""
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return None  # 首次构建：无缓存属正常路径，不告警
        except (OSError, ValueError) as e:
            print(f"[警告] 向量缓存文件损坏（{self.cache_path}：{e}），将重建。",
                  file=sys.stderr)
            return None
        # 逐项校验：版本、模型、语料 hash、id 顺序、向量数量
        if payload.get("version") != _CACHE_VERSION:
            return None
        if payload.get("model") != self.model_name:
            return None
        if payload.get("hash") != self._corpus_hash():
            return None
        if payload.get("ids") != [doc.get("id", "") for doc in self.documents]:
            return None
        embeddings = payload.get("embeddings")
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(self.documents)
        ):
            return None
        return embeddings

    def _write_cache(self, embeddings: list[list[float]]) -> None:
        """持久化语料向量；写失败只告警不抛错（检索功能不受影响）。"""
        payload = {
            "version": _CACHE_VERSION,
            "model": self.model_name,
            "hash": self._corpus_hash(),
            "ids": [doc.get("id", "") for doc in self.documents],
            "embeddings": embeddings,
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp.replace(self.cache_path)  # 原子替换，避免半截文件
        except OSError as e:
            print(f"[警告] 向量缓存写入失败（{self.cache_path}：{e}），"
                  f"本次检索不受影响，下次启动将重新嵌入。", file=sys.stderr)

    # ------------------------------------------------------------ 构建

    def _get_embedder(self) -> Embedder:
        """懒加载嵌入器：首次需要时才 import fastembed / 加载模型。"""
        if self._embedder is None:
            self._embedder = _FastembedEmbedder(self.model_name, self.model_cache_dir)
        return self._embedder

    def _build_or_load(self) -> None:
        cached = self._read_cache()
        if cached is not None:
            # 缓存命中：向量已归一化存储，直接使用（秒级启动、离线可用）
            self._embeddings = cached
            return
        embedder = self._get_embedder()
        texts = [doc.get("text", "") for doc in self.documents]
        self._embeddings = [_normalize(v) for v in embedder.embed_passages(texts)]
        self._write_cache(self._embeddings)

    # ------------------------------------------------------------ 检索

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """返回 [(文档, cosine 得分)] 按得分降序，仅含 ≥ min_score 的结果。

        与 BM25Index 的对齐约定：空查询 / 无过门槛结果时返回空列表，
        由调用方（StatuteSearchTool）输出 MISSING_INFORMATION，绝不伪造。
        """
        if not query.strip():
            return []
        q = _normalize(self._get_embedder().embed_query(query))
        scored = [
            (doc, _dot(q, vec))
            for doc, vec in zip(self.documents, self._embeddings)
        ]
        scored = [(doc, s) for doc, s in scored if s >= self.min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
