# app/tools/ollama_client.py
import time
import requests
import json

def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    force_json: bool = False,
    timeout_sec: int = 900,          # 15 minutes
    max_retries: int = 2,
    num_predict: int = 4096,         # Max output tokens
    show_progress: bool = False,     # Show progress bar during generation
    seed: int = None,                # Deterministic seed for reproducibility
) -> str:
    url = f"{base_url}/api/chat"
    options = {
        "temperature": float(temperature),
        "num_predict": num_predict,
    }
    if seed is not None:
        options["seed"] = int(seed)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": options,
        "stream": show_progress,  # Enable streaming when showing progress
    }
    if force_json:
        payload["format"] = "json"

    import sys
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout_sec, stream=show_progress)
            r.raise_for_status()

            if show_progress:
                # Streaming mode with progress bar
                from tqdm import tqdm
                content_chunks = []
                done_info = {}
                with tqdm(desc="Generating", unit=" tok", bar_format='{desc}: {n} tokens | {elapsed}', leave=False) as pbar:
                    for line in r.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    content_chunks.append(chunk["message"]["content"])
                                    pbar.update(1)
                                if chunk.get("done", False):
                                    done_info = chunk
                                    break
                            except json.JSONDecodeError:
                                continue
                content = "".join(content_chunks)

                # Enhanced diagnostic logging for empty content
                if not content:
                    print(f"WARNING: Ollama returned empty content in streaming mode.", file=sys.stderr)
                    print(f"  Model: {model}", file=sys.stderr)
                    print(f"  Temperature: {temperature}", file=sys.stderr)
                    print(f"  num_predict: {num_predict}", file=sys.stderr)
                    print(f"  force_json: {force_json}", file=sys.stderr)
                    print(f"  System prompt length: {len(system)} chars", file=sys.stderr)
                    print(f"  User prompt length: {len(user)} chars", file=sys.stderr)
                    if done_info:
                        print(f"  Done info: total_duration={done_info.get('total_duration')}, "
                              f"load_duration={done_info.get('load_duration')}, "
                              f"prompt_eval_count={done_info.get('prompt_eval_count')}, "
                              f"eval_count={done_info.get('eval_count')}", file=sys.stderr)
            else:
                # Non-streaming mode (original behavior)
                resp_json = r.json()
                content = resp_json.get("message", {}).get("content", "")

                # Enhanced diagnostic logging for empty content
                if not content:
                    print(f"WARNING: Ollama returned empty content in non-streaming mode.", file=sys.stderr)
                    print(f"  Model: {model}", file=sys.stderr)
                    print(f"  Temperature: {temperature}", file=sys.stderr)
                    print(f"  num_predict: {num_predict}", file=sys.stderr)
                    print(f"  force_json: {force_json}", file=sys.stderr)
                    print(f"  System prompt length: {len(system)} chars", file=sys.stderr)
                    print(f"  User prompt length: {len(user)} chars", file=sys.stderr)
                    if resp_json:
                        print(f"  Response keys: {list(resp_json.keys())}", file=sys.stderr)
                        if "message" in resp_json:
                            print(f"  Message keys: {list(resp_json['message'].keys())}", file=sys.stderr)
                        print(f"  Done: {resp_json.get('done')}", file=sys.stderr)
                        print(f"  Total duration: {resp_json.get('total_duration')}", file=sys.stderr)
                        print(f"  Eval count: {resp_json.get('eval_count')}", file=sys.stderr)

            return content
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            # backoff: 2s, 5s, 10s
            sleep_s = [2, 5, 10][min(attempt, 2)]
            print(f"WARNING: Ollama timeout/connection error on attempt {attempt+1}/{max_retries+1}. Retrying in {sleep_s}s...", file=sys.stderr)
            time.sleep(sleep_s)
        except requests.exceptions.HTTPError as e:
            # Don't retry 4xx by default (usually bad request)
            print(f"ERROR: Ollama HTTP error {r.status_code}: {r.text[:500]}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"ERROR: Unexpected Ollama error: {type(e).__name__}: {e}", file=sys.stderr)
            raise
    raise last_err
