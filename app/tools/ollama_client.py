# app/tools/ollama_client.py
import time
import requests

def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    force_json: bool = False,
    timeout_sec: int = 900,          # 15 minutes
    max_retries: int = 2,
) -> str:
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    if force_json:
        payload["format"] = "json"

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout_sec)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            # backoff: 2s, 5s, 10s
            sleep_s = [2, 5, 10][min(attempt, 2)]
            time.sleep(sleep_s)
        except requests.exceptions.HTTPError as e:
            # Don't retry 4xx by default (usually bad request)
            raise
    raise last_err
