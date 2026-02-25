# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
SMLP Agent RL Utilities

Shared utilities for LLM calls, SMLP execution, and output filtering.
Used by both the interactive session (smlp_agent_rl_session.py) and the
FastAPI server (smlp_agent_rl_server.py).
"""

import json
import os
import re
import subprocess
import sys
from typing import Dict, Tuple


def _clean_json(raw: str) -> str:
    """
    Clean LLM response to extract pure JSON. Handles:
      - DeepSeek-R1 <think>...</think> reasoning blocks (complete or truncated)
      - Markdown ```json ... ``` or ``` ... ``` fences
      - Leading/trailing whitespace and newlines

    Fallback: if normal cleaning still doesn't yield a '{', scan the raw
    text for the first '{' and extract the JSON object by brace-counting.
    This recovers from truncated <think> blocks with no closing tag.
    """
    raw = raw.strip()

    # Case 1: complete <think>...</think> block — strip it
    if '<think>' in raw and '</think>' in raw:
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)

    # Case 2: truncated <think> block with no closing tag —
    # find first '{' that appears after the think content
    elif '<think>' in raw:
        brace_pos = raw.find('{')
        if brace_pos != -1:
            raw = raw[brace_pos:]
        else:
            # No JSON brace found at all — model only produced thinking,
            # not the answer yet. Return empty so caller shows the error.
            return ''

    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```\s*$',       '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Final fallback: if text still doesn't start with '{',
    # scan for first '{' and extract balanced JSON object
    if raw and not raw.startswith('{'):
        brace_pos = raw.find('{')
        if brace_pos != -1:
            # Extract from first '{' to the matching closing '}'
            depth, end = 0, -1
            for i, ch in enumerate(raw[brace_pos:], brace_pos):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end != -1:
                raw = raw[brace_pos:end]

    return raw.strip()


def _call_llm(prompt: str, provider: str, model: str,
              timeout: int = 120) -> Dict:
    """
    Call an LLM (Ollama or OpenAI) with the given prompt.

    Returns a dict with the parsed JSON response, or {"error": "..."} on failure.
    Null/None values in the response are stripped to reduce noise.
    """

    if provider == "ollama":
        import requests as _requests

        def _ollama_post(prompt_text: str, ctx: int) -> dict:
            """Single Ollama API call. Returns parsed dict or raises."""
            res = _requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":      model,
                    "prompt":     prompt_text,
                    "stream":     False,
                    # keep_alive: 0 unloads the model from VRAM/RAM immediately
                    # after this response. This forces a clean reload on the next
                    # call, ensuring num_ctx is always respected. Without this,
                    # Ollama keeps the model loaded with its first num_ctx value
                    # and returns 500 on subsequent calls that request a different
                    # context size.
                    "keep_alive": -1,
                    "options": {
                        "num_ctx":     ctx,
                        "num_predict": 512,  # JSON never needs more than this
                        "temperature": 0,
                    },
                },
                timeout=timeout,
            )
            res.raise_for_status()
            raw = res.json().get("response", "")
            parsed = json.loads(_clean_json(raw))
            return {k: v for k, v in parsed.items() if v is not None}

        # Attempt 1: full prompt with generous context (8192 covers Mistral well)
        try:
            return _ollama_post(prompt, ctx=8192)
        except json.JSONDecodeError as e:
            return {"error": f"LLM returned non-JSON: {e}"}
        except _requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 500:
                # 500 = context size mismatch or model overload.
                # Retry with minimal prompt (system preamble + query only,
                # no few-shot examples) and smaller context window.
                print("\n  (500 from Ollama — retrying with minimal prompt…)",
                      end="", flush=True)
                # Extract system preamble: everything before "Examples:" or "#1"
                for marker in ["\n\nExamples:", "\n#1 ", "\nNow convert"]:
                    idx = prompt.find(marker)
                    if idx != -1:
                        system_part = prompt[:idx].strip()
                        break
                else:
                    system_part = prompt.split("\n\n")[0].strip()
                # Extract the user query: last quoted block in the prompt
                m = re.search(r'"([^"]+)"\s*$', prompt, re.DOTALL)
                user_query_text = m.group(1) if m else ""
                short_prompt = (
                    f"{system_part}\n\n"
                    "Convert the following description into a JSON CLI options "
                    "dictionary. Output ONLY the JSON, nothing else.\n\n"
                    f'User:\n"{user_query_text}"'
                )
                try:
                    result = _ollama_post(short_prompt, ctx=4096)
                    print(" retry succeeded.", flush=True)
                    return result
                except json.JSONDecodeError as e2:
                    print(f" retry failed: non-JSON: {e2}", flush=True)
                    return {"error": f"LLM returned non-JSON on retry: {e2}"}
                except Exception as e2:
                    print(f" retry failed: {e2}", flush=True)
                    return {"error": f"Ollama retry failed: {e2}"}
            return {"error": f"Ollama error: {e}"}
        except Exception as e:
            return {"error": f"Ollama error: {e}"}

    if provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=60,
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(_clean_json(raw))
            return {k: v for k, v in parsed.items() if v is not None}
        except json.JSONDecodeError as e:
            return {"error": f"LLM returned non-JSON: {e}"}
        except Exception as e:
            return {"error": f"OpenAI error: {e}"}

    return {"error": f"Unknown provider '{provider}'. Use 'ollama' or 'openai'."}


def _filter_noise(output: str) -> str:
    """
    Strip CUDA/TF registration warnings that are irrelevant to SMLP errors.
    
    Filters out TensorFlow/CUDA initialization noise while preserving actual
    SMLP command errors and runtime output.
    """
    _noise_prefixes = (
        "E external/local_xla", "WARNING: All log messages",
        "W0000", "E0000", "I tensorflow", "computation placer",
    )
    def _is_noise(line):
        s = line.strip()
        # TF timestamp lines like "2026-02-15 12:33:52.493595: E ..."
        if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', s):
            return True
        return any(s.startswith(p) for p in _noise_prefixes)
    filtered = "\n".join(l for l in output.splitlines() if not _is_noise(l))
    return filtered.strip()


def _is_command_error(output: str) -> bool:
    """
    Return True if the execution output indicates the command itself was wrong
    (bad option name, invalid value, missing required argument) rather than a
    runtime/data error.

    Used to decide whether to allow rating:
      - command error  → user's corrected command was also wrong → skip rating
      - runtime error  → command was syntactically valid → allow rating
    """
    _patterns = [
        # argparse / CLI errors
        "unrecognized arguments",
        "unrecognized option",
        "invalid choice",
        "error: argument",
        "ambiguous option",
        "expected one argument",
        "the following arguments are required",
        # Python type-conversion errors from bad option values
        "invalid literal for int",
        "could not convert string to float",
        # SMLP-specific option validation messages
        "is not a valid",
        "must be one of",
        "unknown mode",
        "unknown option",
    ]
    low = output.lower()
    return any(p in low for p in _patterns)


def _execute_smlp(cmd: Dict, dry_run: bool) -> Tuple[bool, str]:
    """
    Execute the SMLP command dict.

    Execution strategy (tried in order):
      1. SmlpAgent.run_command(cmd)  — reuses the existing agent's own executor,
         which knows the correct flag format (single-dash abbreviated names).
         This is the same path that worked before with the regular SMLP agent.
      2. Fallback: build CLI args with single-dash abbreviated names directly,
         e.g.  -mode train -data ../data/...  (NOT double-dash --mode).

    Always writes full unfiltered output to smlp_last_run.log for debugging.
    Returns (success: bool, filtered_output: str).
    """
    if dry_run:
        return True, "[dry-run] Command not executed."

    # ── Build the single-dash arg list (correct SMLP format) ─────────────────
    # SMLP uses abbreviated single-dash flags: -mode, -data, -resp, etc.
    # Double-dash (--mode) is wrong and causes "unrecognized arguments" errors.
    args = []
    for k, v in cmd.items():
        args.append(f"-{k}")          # single dash, abbreviated name
        args.append(str(v))
    full_cmd = [sys.executable, "-u", "./run_smlp.py"] + args

    # Print full command so it can be copy-pasted for debugging
    print(f"\n  Full command:")
    print(f"    {' '.join(full_cmd)}\n")

    # ── Strategy 1: use SmlpAgent.run_command if available ───────────────────
    try:
        from smlp_agent import SmlpAgent
        agent = SmlpAgent()
        if hasattr(agent, 'run_command'):
            success, stdout, stderr = agent.run_command(cmd)
            full_output = (stdout or "") + (stderr or "")
            _write_log(full_cmd, 0 if success else 1, stdout or "", stderr or "")
            return success, _filter_noise(full_output)
        elif hasattr(agent, 'execute_command'):
            result = agent.execute_command(cmd)
            full_output = str(result)
            _write_log(full_cmd, 0, full_output, "")
            return True, _filter_noise(full_output)
    except (ImportError, Exception):
        pass   # fall through to subprocess

    # ── Strategy 2: subprocess with single-dash args ──────────────────────────
    try:
        res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=300)
        success = res.returncode == 0
        full_output = (res.stdout or "") + (res.stderr or "")
        _write_log(full_cmd, res.returncode, res.stdout or "", res.stderr or "")
        return success, _filter_noise(full_output)
    except subprocess.TimeoutExpired:
        return False, "Execution timed out after 300 s."
    except FileNotFoundError:
        return False, "run_smlp.py not found. Use --dry-run."
    except Exception as e:
        return False, str(e)


def _write_log(full_cmd, returncode: int, stdout: str, stderr: str):
    """Write full unfiltered output to smlp_last_run.log."""
    log_path = "./smlp_last_run.log"
    with open(log_path, "w") as lf:
        lf.write(f"Command: {' '.join(full_cmd)}\n")
        lf.write(f"Return code: {returncode}\n")
        lf.write("=" * 60 + "\n")
        lf.write("STDOUT:\n")
        lf.write(stdout or "(empty)\n")
        lf.write("=" * 60 + "\n")
        lf.write("STDERR:\n")
        lf.write(stderr or "(empty)\n")
