"""本地轻量检索引擎：纯标准库 BM25 + 法条语料加载。

设计意图与边界约束：
- 不引入 jieba 等第三方分词依赖：中文按 bigram + 单字混合切分，
  英文/数字按连续字符段切分，足以支撑法条级关键词检索；
- BM25 参数 k1=1.5, b=0.75（常用默认值），确定性打分，可审计可复现；
- 仅提供只读检索能力（Level 0），不做任何语义判断或裁判。

语料加载优先级：
  环境变量 LEGAL_AI_STATUTE_CORPUS 指定的 JSONL 文件
  > 项目内 data/statutes.jsonl
  > 内置 FALLBACK_STATUTES 兜底（文件缺失/损坏时 stderr 警告后兜底，不崩溃）。

JSONL 每行格式：{"id": str, "source": str, "text": str}
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- 内置兜底语料

FALLBACK_STATUTES = [
    {
        "id": "statute-563",
        "source": "《中华人民共和国民法典》第五百六十三条",
        "text": "有下列情形之一的，当事人可以解除合同：（一）因不可抗力致使不能实现合同目的；"
                "（二）在履行期限届满前，当事人一方明确表示或者以自己的行为表明不履行主要债务；"
                "（三）当事人一方迟延履行主要债务，经催告后在合理期限内仍未履行。",
    },
    {
        "id": "statute-577",
        "source": "《中华人民共和国民法典》第五百七十七条",
        "text": "当事人一方不履行合同义务或者履行合同义务不符合约定的，"
                "应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
    },
]

#: 项目内默认语料路径（agent_layer/tools/retrieval.py → 项目根/data/statutes.jsonl）
_DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "statutes.jsonl"
)


# ---------------------------------------------------------------- 分词

def tokenize(text: str) -> list[str]:
    """中文 bigram + 单字混合切分；英文/数字按连续段切分（小写化）。"""
    tokens: list[str] = []
    prev_cjk: str | None = None
    buf: list[str] = []

    def flush_buf() -> None:
        if buf:
            tokens.append("".join(buf).lower())
            buf.clear()

    for ch in text:
        if "一" <= ch <= "鿿":  # CJK 统一表意文字基本区
            flush_buf()
            tokens.append(ch)
            if prev_cjk is not None:
                tokens.append(prev_cjk + ch)
            prev_cjk = ch
        elif ch.isalnum():
            buf.append(ch)
            prev_cjk = None
        else:
            flush_buf()
            prev_cjk = None
    flush_buf()
    return tokens


# ---------------------------------------------------------------- BM25 索引

class BM25Index:
    """对 {id, source, text} 文档集合构建 BM25 索引并提供 top_k 检索。"""

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = list(documents)
        self._doc_tokens: list[list[str]] = [
            tokenize(doc.get("text", "")) for doc in self.documents
        ]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        n = len(self.documents)
        self._avgdl = sum(self._doc_len) / n if n else 0.0
        # 文档频率
        df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        self._df = df

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self._df.get(term, 0)
        # Robertson 形式的正值 idf
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """返回 [(文档, 得分)] 按得分降序，最多 top_k 条。

        相关度门槛：文档必须至少命中查询的一个多字词（bigram / 英文数字词）
        才计分。仅由单字（unigram）命中的文档得 0 分——单字（如"民""事"）
        在法律语料中频率极高，单独命中不构成相关性（修复：无关查询
        "火星殖民地"仅因单字"民"命中而返回无关法条的问题）。
        查询本身不含多字词（如单字查询）时退化为要求命中该单字。
        """
        if not self.documents or not query.strip():
            return []
        terms = tokenize(query)
        multi_terms = {t for t in terms if len(t) > 1}
        require_terms = multi_terms if multi_terms else set(terms)
        scores: list[float] = []
        for toks, dl in zip(self._doc_tokens, self._doc_len):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            if not any(t in tf for t in require_terms):
                scores.append(0.0)  # 未命中任何多字词：不相关，不计分
                continue
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (
                    1.0 - self.b + self.b * dl / self._avgdl
                ) if self._avgdl else f + self.k1
                score += self._idf(term) * (f * (self.k1 + 1.0)) / denom
            scores.append(score)
        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


# ---------------------------------------------------------------- 语料加载

def _load_jsonl(path: Path) -> list[dict]:
    """解析 JSONL 语料；任何格式问题抛异常，由调用方决定兜底。"""
    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not all(k in obj for k in ("id", "source", "text")):
                raise ValueError(f"{path} 第 {lineno} 行缺少 id/source/text 字段")
            docs.append(obj)
    if not docs:
        raise ValueError(f"{path} 为空文件")
    return docs


def load_statute_corpus() -> list[dict]:
    """按优先级加载法条语料；任何一级失败都 stderr 警告并降级，不崩溃。"""
    candidates: list[tuple[str, Path]] = []
    env_path = os.environ.get("LEGAL_AI_STATUTE_CORPUS")
    if env_path:
        candidates.append((f"环境变量 LEGAL_AI_STATUTE_CORPUS", Path(env_path)))
    candidates.append(("项目内置语料", _DEFAULT_CORPUS_PATH))

    for label, path in candidates:
        try:
            docs = _load_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"[警告] {label}加载失败（{path}：{e}），尝试下一级。",
                  file=sys.stderr)
            continue
        return docs

    print("[警告] 法条语料文件不可用，使用内置兜底语料（2 条示例法条）。",
          file=sys.stderr)
    return list(FALLBACK_STATUTES)
