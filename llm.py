import json
import httpx
from typing import Dict, Iterable, Generator, List, Optional

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-reasoner", timeout: float = 680.0):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key
            base_url: Base URL for API (default: https://api.deepseek.com)
            model: Model name (default: deepseek-reasoner)
            timeout: Timeout in seconds (default: 180.0 for reasoning models)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Use longer timeout for reasoning models (they need time to generate reasoning chain)
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """
        Yields token chunks as they arrive (OpenAI-compatible streaming).
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if kwargs:
            payload.update(kwargs)

        url = f"{self.base_url}/v1/chat/completions"
        timeout_config = httpx.Timeout(
            connect=10.0,
            read=self.timeout,
            write=10.0,
            pool=10.0
        )
        with httpx.stream("POST", url, headers=self._headers(), json=payload, timeout=timeout_config) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    chunk = delta.get("content")
                    if chunk:
                        yield chunk
                except Exception:
                    # swallow malformed SSE fragments
                    continue

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Non-streaming chat completion. Supports reasoning models like deepseek-reasoner.
        For reasoning models, returns the final content (not reasoning_content).
        """
        payload = {
            "model": self.model,
            "messages": messages
        }
        if kwargs:
            payload.update(kwargs)

        url = f"{self.base_url}/v1/chat/completions"
        # Use a Timeout object with separate connect and read timeouts
        # Reasoning models can take a long time, so we use a generous read timeout
        timeout_config = httpx.Timeout(
            connect=10.0,  # 10 seconds to establish connection
            read=self.timeout,  # Use the full timeout for reading the response
            write=10.0,  # 10 seconds to write the request
            pool=10.0  # 10 seconds to get a connection from pool
        )
        with httpx.Client(timeout=timeout_config) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            # For reasoning models, extract content (final answer)
            # reasoning_content is also available but not returned
            return message.get("content", "")
