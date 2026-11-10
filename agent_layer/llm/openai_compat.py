"""OpenAI 兼容 LLM 适配器（纯标准库实现，可离线导入、联网调用）。

设计意图与边界约束：
- 仅实现 LLMClient 协议（complete），不引入任何自主行动/再规划逻辑；
- 通过 system 指令约束模型"仅输出 JSON，不得输出任何法律结论/裁判意见"，
  但真正的行为边界仍由 Agent 层 IntentGate + BoundaryGuard 强制（双通道约束），
  本适配器不作为安全防线；
- 任何网络错误/超时/HTTP 错误/JSON 解析失败统一抛出 LLMError，
  绝不静默返回伪造内容；是否降级由调用方（FallbackLLMClient）决定。

配置（环境变量，构造参数可覆盖）：
- LEGAL_AI_LLM_BASE_URL  默认 https://api.openai.com/v1
- LEGAL_AI_LLM_API_KEY   必填，缺失时构造抛 LLMError
- LEGAL_AI_LLM_MODEL     默认 gpt-4o-mini
- LEGAL_AI_LLM_TIMEOUT   默认 30 秒
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request

#: 默认服务端点与模型（可被环境变量/构造参数覆盖）
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_TOKENS = 1024

#: 输出侧纪律：模型被要求只做结构化抽取，不输出法律结论/裁判意见。
_SYSTEM_PROMPT = (
    "你是法律信息处理流水线中的一个结构化抽取子组件。"
    "你必须且只能输出符合用户要求结构的 JSON，不得输出任何解释性文字、"
    "法律结论、裁判意见、责任认定或量刑建议。"
)


class LLMError(Exception):
    """LLM 调用失败的统一异常（携带失败原因，供降级与审计使用）。"""


class OpenAICompatClient:
    """OpenAI /chat/completions 兼容端点的 LLMClient 实现（urllib 纯标准库）。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.base_url = (
            base_url
            or os.environ.get("LEGAL_AI_LLM_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key if api_key is not None
            else os.environ.get("LEGAL_AI_LLM_API_KEY", "")
        )
        self.model = (
            model or os.environ.get("LEGAL_AI_LLM_MODEL") or DEFAULT_MODEL
        )
        self.timeout = float(
            timeout if timeout is not None
            else os.environ.get("LEGAL_AI_LLM_TIMEOUT", DEFAULT_TIMEOUT)
        )
        self.max_tokens = max_tokens
        if not self.api_key:
            raise LLMError(
                "缺少 API Key：请设置环境变量 LEGAL_AI_LLM_API_KEY "
                "或在构造时传入 api_key"
            )

    # ---------------------------------------------------------------- 协议实现

    def complete(self, prompt: str, *, schema: dict | None = None) -> str:
        """生成文本。schema 非空时返回符合该结构的 JSON 字符串。"""
        user_content = self._build_user_content(prompt, schema)
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,  # 确定性优先，便于审计复现
            "max_tokens": self.max_tokens,
        }
        if schema:
            # 优先使用结构化输出约束；端点不支持时降级为普通请求重试一次
            payload["response_format"] = {"type": "json_object"}
            try:
                raw = self._post(payload)
            except LLMError as e:
                if "HTTP 400" not in str(e):
                    raise
                payload.pop("response_format")
                raw = self._post(payload)
            return self._extract_json(raw)
        raw = self._post(payload)
        return self._extract_content(raw)

    # ---------------------------------------------------------------- 请求构造

    def _build_user_content(self, prompt: str, schema: dict | None) -> str:
        if schema and schema.get("task") == "extract_entities":
            text = schema.get("text", "")
            return (
                "请从以下案件材料中抽取人物与日期，"
                "只输出一个 JSON 对象，格式为 "
                '{"persons": [...], "dates": [...]}，'
                "persons 为材料中出现的人物称谓列表，"
                "dates 为材料中出现的日期字符串列表，"
                "不要输出任何其他内容。\n\n材料：\n" + text
            )
        if schema:
            return (
                prompt
                + "\n\n请只输出符合以下结构描述的 JSON，不要输出任何其他内容：\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        return prompt

    # ---------------------------------------------------------------- 网络调用

    def _post(self, payload: dict) -> dict:
        """发送请求并返回解析后的响应体；一切失败抛 LLMError。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            raise LLMError(
                f"HTTP {e.code} 错误：{detail or e.reason}"
            ) from e
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            raise LLMError(f"网络错误/超时：{e}") from e
        except OSError as e:
            raise LLMError(f"连接失败：{e}") from e
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as e:
            raise LLMError(f"响应体不是合法 JSON：{e}") from e

    # ---------------------------------------------------------------- 响应解析

    @staticmethod
    def _extract_content(data: dict) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"响应结构异常，无法提取 message.content：{e}") from e
        if not isinstance(content, str) or not content.strip():
            raise LLMError("响应 content 为空")
        return content

    @classmethod
    def _extract_json(cls, data: dict) -> str:
        """提取 JSON 字符串，容忍 ```json 代码块包裹；无法解析则抛 LLMError。"""
        content = cls._extract_content(data).strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if m:
            content = m.group(1).strip()
        try:
            json.loads(content)  # 仅校验可解析性，原样返回字符串
        except json.JSONDecodeError as e:
            raise LLMError(f"模型输出不是合法 JSON：{e}") from e
        return content
