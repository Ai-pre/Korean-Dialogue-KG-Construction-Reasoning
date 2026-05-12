from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib import error, request


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)


@dataclass(slots=True)
class OpenAICompatibleClient:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: int = 60

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient | None":
        base_url = (
            os.getenv("KG_REASONING_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        model = (
            os.getenv("KG_REASONING_CHAT_MODEL")
            or os.getenv("OPENAI_MODEL")
            or os.getenv("OPENAI_CHAT_MODEL")
        )
        if not base_url or not model:
            return None
        api_key = os.getenv("KG_REASONING_API_KEY") or os.getenv("OPENAI_API_KEY")
        return cls(base_url=base_url.rstrip("/"), model=model, api_key=api_key)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = self._post_json("/chat/completions", payload)
        return response["choices"][0]["message"]["content"].strip()

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict:
        text = self.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
        return parse_json_from_text(text)

    def _post_json(self, route: str, payload: dict) -> dict:
        target = f"{self.base_url}/v1{route}" if not self.base_url.endswith("/v1") else f"{self.base_url}{route}"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(target, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as handle:
                return json.loads(handle.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM request could not reach {target}: {exc}") from exc


def parse_json_from_text(text: str) -> dict:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    block_match = JSON_BLOCK_RE.search(text)
    if block_match:
        return json.loads(block_match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object could be parsed from the model output.")
