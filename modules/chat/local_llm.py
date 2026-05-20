from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class LocalLLMResult:
    ok: bool
    text: str
    error: str = ""


class OllamaClient:
    """Tiny Ollama HTTP client using only Python stdlib."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_sec: float = 90.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def is_available(self) -> bool:
        try:
            req = Request(f"{self.base_url}/api/tags", method="GET")
            with urlopen(req, timeout=self.timeout_sec) as resp:
                if resp.status != 200:
                    return False
                payload = json.loads(resp.read().decode("utf-8"))
            return isinstance(payload, dict) and "models" in payload
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        model: str,
        system: str = "",
        temperature: float = 0.2,
    ) -> LocalLLMResult:
        body = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": float(temperature),
            },
        }
        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{self.base_url}/api/generate",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=self.timeout_sec) as resp:
                if resp.status != 200:
                    return LocalLLMResult(ok=False, text="", error=f"HTTP {resp.status}")
                payload = json.loads(resp.read().decode("utf-8"))
        except URLError as exc:
            return LocalLLMResult(ok=False, text="", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return LocalLLMResult(ok=False, text="", error=str(exc))

        text = str(payload.get("response", "")).strip()
        if not text:
            return LocalLLMResult(ok=False, text="", error="Порожня відповідь моделі.")
        return LocalLLMResult(ok=True, text=text)
