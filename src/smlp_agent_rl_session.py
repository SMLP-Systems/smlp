#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
SMLP Agent – Interactive RL Session
=====================================

An interactive terminal loop that ties the real LLM (via your existing
LLMInterpreter) to the RL feedback system.

Each turn:
  1. User types a natural-language SMLP query
  2. LLM generates a command dict using the current UCB-selected prompt
  3. User sees the command and can:
       a) Accept as-is          → press Enter
       b) Edit the JSON         → opens $EDITOR (or inline edit mode)
       c) Reject entirely       → type 'skip'
  4. User optionally overrides the auto-computed reward (1–5 scale)
  5. System executes the command (unless --dry-run)
  6. RL state updated: UCB scores, feedback log, example store JSON

Usage
-----
  # Dry-run (no SMLP execution, safe for testing):
  python smlp_agent_rl_session.py --dry-run

  # Real execution (needs run_smlp.py in cwd or on PATH):
  python smlp_agent_rl_session.py

  # Choose provider / model:
  python smlp_agent_rl_session.py --provider openai --model gpt-4o --dry-run
  python smlp_agent_rl_session.py --provider ollama --model mistral --dry-run

  # Restore a previous session's RL state:
  python smlp_agent_rl_session.py --checkpoint ./smlp_rl_checkpoints

Commands inside the session
---------------------------
  <query>      – run the RL loop for this SMLP query
  /stats       – show current RL training statistics
  /prompt      – print the current few-shot prompt
  /store       – print the example store summary table
  /save        – manually save checkpoint
  /help        – show this help
  /quit        – exit (checkpoint auto-saved)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from copy import deepcopy
from typing import Dict, Optional

# ── RL components ─────────────────────────────────────────────────────────────
from smlp_agent_rl import (
    ExampleStore, FeedbackCollector, PromptOptimizer, RLTrainer, RewardModel
)

SEP  = "=" * 70
THIN = "─" * 70
STORE_FILE    = "./smlp_few_shot_examples.json"
FEEDBACK_FILE = "./smlp_rl_feedback.jsonl"
CHECKPOINT    = "./smlp_rl_checkpoints"


# ═════════════════════════════════════════════════════════════════════════════
# Tiny LLM shim — calls your existing LLMInterpreter if available,
# otherwise falls back to a clearly-labelled mock so the session still runs.
# ═════════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, provider: str, model: str,
              timeout: int = 120) -> Dict:
    """
    Send a fully-assembled prompt → LLM → parse JSON response.

    We always call the underlying API directly (Ollama /api/generate or
    OpenAI chat completions) rather than going through LLMInterpreter, for
    two reasons:
      1. LLMInterpreter.plan_from_text() prepends its own copy of
         self.few_shot_prompt, which would double the prompt we built.
      2. It gives us explicit control over timeouts.

    The prompt passed in already contains the system preamble, all few-shot
    examples, AND the user query at the bottom — nothing more is appended.

    Returns a dict, possibly {"error": "..."} on failure.
    """
    import re

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
                    "keep_alive": 0,
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
                # 500 = KV-cache conflict from a previously loaded context size.
                # keep_alive:0 above should prevent this, but as a belt-and-braces
                # retry: rebuild a minimal prompt (system preamble + query only,
                # no few-shot examples) and try again with a smaller ctx.
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
                import re as _re
                m = _re.search(r'"([^"]+)"\s*$', prompt, _re.DOTALL)
                user_query_text = m.group(1) if m else ""
                short_prompt = (
                    f"{system_part}\n\n"
                    "Convert the following description into a JSON CLI options "
                    "dictionary. Output ONLY the JSON, nothing else.\n\n"
                    f'User:\n"{user_query_text}"'
                )
                try:
                    return _ollama_post(short_prompt, ctx=4096)
                except json.JSONDecodeError as e2:
                    return {"error": f"LLM returned non-JSON on retry: {e2}"}
                except Exception as e2:
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


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _print_command(cmd: Dict, label: str = "Generated command"):
    print(f"\n  ┌─ {label} {'─'*(52-len(label))}┐")
    for line in json.dumps(cmd, indent=2).split("\n"):
        print(f"  │  {line}")
    print(f"  └{'─'*56}┘")


def _edit_inline(cmd: Dict) -> Dict:
    """
    Let the user edit the command JSON key-by-key in the terminal.
    Returns the (possibly modified) dict.
    """
    print("\n  Edit mode — press Enter to keep current value, or type new value.")
    print("  Type  +key=value  to add a key,  -key  to remove a key.\n")
    result = deepcopy(cmd)

    # Show existing keys
    for key, val in list(result.items()):
        prompt_str = f"    {key} [{json.dumps(val)}]: "
        try:
            raw = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw == "":
            continue  # keep
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError:
            result[key] = raw  # treat as string

    # Add / remove keys
    print("\n  Add (+key=value) or remove (-key) keys, blank to finish:")
    while True:
        try:
            raw = input("    > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw == "":
            break
        if raw.startswith("+") and "=" in raw:
            k, v = raw[1:].split("=", 1)
            try:
                result[k.strip()] = json.loads(v.strip())
            except json.JSONDecodeError:
                result[k.strip()] = v.strip()
            print(f"    added: {k.strip()} = {result[k.strip()]}")
        elif raw.startswith("-"):
            k = raw[1:].strip()
            if k in result:
                del result[k]
                print(f"    removed: {k}")
            else:
                print(f"    key not found: {k}")
        else:
            print("    Use +key=value or -key")

    return result


def _edit_in_editor(cmd: Dict) -> Dict:
    """Open the JSON in $EDITOR and return the parsed result."""
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                    delete=False) as f:
        json.dump(cmd, f, indent=2)
        tmppath = f.name
    try:
        subprocess.run([editor, tmppath])
        with open(tmppath) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [editor error: {e}] Keeping original.")
        return cmd
    finally:
        os.unlink(tmppath)


def _ask_reward_override(auto_reward: float) -> Optional[float]:
    """
    Ask user to rate the quality of the LLM's *generated* command
    (before any user correction).  This is the RL training signal —
    it measures how useful the LLM's suggestion was, not whether SMLP ran.

    Returns:
      float in [0,1]  — reward to record
      None            — skip (Enter): no feedback recorded this turn

    Input options:
      Enter     → skip evaluation entirely (no feedback recorded)
      a / auto  → accept the auto-computed edit-distance reward
      1–5       → override: 1=LLM was completely wrong, 5=LLM was perfect
    """
    print()
    print(f"  ┌─ Rate the LLM's suggestion (before your corrections) ─────────┐")
    print(f"  │  Auto-computed (edit distance): {auto_reward:.3f}                      │")
    print(f"  │  1→0.00  2→0.25  3→0.50  4→0.75  5→1.00  (reward stored)   │")
    print(f"  │  'a' accept auto · 1-5 override · blank skip                 │")
    print(f"  └───────────────────────────────────────────────────────────────┘")
    try:
        raw = input("  Rating> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if raw == "":
        return None                # skip — no feedback recorded

    if raw in ("a", "auto"):
        return auto_reward         # accept auto as-is

    try:
        score = float(raw)
        if 1 <= score <= 5:
            return (score - 1) / 4   # 1→0.0, 2→0.25, 3→0.5, 4→0.75, 5→1.0
        else:
            print("  Out of range (need 1-5), skipping evaluation.")
            return None
    except ValueError:
        print("  Not recognised, skipping evaluation.")
        return None


def _execute_smlp(cmd: Dict, dry_run: bool) -> tuple:
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
    """Strip CUDA/TF registration warnings that are irrelevant to SMLP errors."""
    import re as _re
    _noise_prefixes = (
        "E external/local_xla", "WARNING: All log messages",
        "W0000", "E0000", "I tensorflow", "computation placer",
    )
    def _is_noise(line):
        s = line.strip()
        if _re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', s):
            return True
        return any(s.startswith(p) for p in _noise_prefixes)
    filtered = "\n".join(l for l in output.splitlines() if not _is_noise(l))
    return filtered.strip()


def _print_stats(trainer: RLTrainer):
    s = trainer.get_training_stats()
    print(f"\n  ┌─ RL Stats {'─'*46}┐")
    print(f"  │  Total feedback      : {s['total_feedback']:<34}│")
    print(f"  │  Overall avg reward  : {s['avg_reward']:<34.3f}│")
    print(f"  │  Recent  avg reward  : {s['recent_avg_reward']:<34.3f}│")
    print(f"  │  Pool size           : {s['num_examples']:<34}│")
    print(f"  │  Total UCB selections: {s['total_selections']:<34}│")
    print(f"  └{'─'*57}┘")


# ═════════════════════════════════════════════════════════════════════════════
# Main interactive loop
# ═════════════════════════════════════════════════════════════════════════════

class InteractiveRLSession:
    """
    Drives one interactive terminal session.  Each call to step() handles
    one user query end-to-end.
    """

    def __init__(self,
                 provider:    str  = "ollama",
                 model:       str  = "deepseek-r1:1.5b",
                 dry_run:     bool = True,
                 checkpoint:  str  = CHECKPOINT,
                 store_file:  str  = STORE_FILE,
                 llm_timeout: int  = 120):

        self.provider    = provider
        self.model       = model
        self.dry_run     = dry_run
        self.checkpoint  = checkpoint
        self.llm_timeout = llm_timeout

        # ── RL stack ──────────────────────────────────────────────────────
        print(f"\n  Loading example store from {store_file} …")
        self.store = ExampleStore(store_file)

        self.feedback_collector = FeedbackCollector(storage_path=FEEDBACK_FILE)

        # Small models (≤2B params) struggle with long prompts containing
        # multiple few-shot examples — they run out of context and truncate
        # mid-generation. Use 1 example for small models, 3 for larger ones.
        _small_models = ("1.5b", "0.5b", "0.5", "1b", "tinyllama")
        _is_small = any(tag in model.lower() for tag in _small_models)
        _examples_per_prompt = 1 if _is_small else 3
        if _is_small:
            print(f"  Small model detected ({model}) — using "
                  f"{_examples_per_prompt} few-shot example per prompt "
                  f"to avoid context overflow.")

        self.prompt_optimizer   = PromptOptimizer(
            max_examples        = 100,
            examples_per_prompt = _examples_per_prompt,
            exploration_factor  = 0.5,
            example_store       = self.store,
        )
        self.trainer = RLTrainer(
            self.feedback_collector,
            self.prompt_optimizer,
            fine_tune_threshold = 20,
        )
        self.trainer.load_checkpoint(checkpoint)

        n_feedback = len(self.feedback_collector.feedback_buffer)
        print(f"  Previous feedback sessions loaded: {n_feedback}")
        print(f"  Example pool size: {len(self.prompt_optimizer.example_pool)}")
        if self.dry_run:
            print("  [DRY-RUN mode — SMLP will not be executed]")

    # ── single turn ───────────────────────────────────────────────────────

    def step(self, user_query: str) -> bool:
        """
        Run one full RL turn for user_query.
        Returns False if the user chose to skip this turn.
        """
        print(f"\n{THIN}")

        # 1. Build current prompt and call LLM
        prompt = self.trainer.get_current_prompt(user_query)
        full_prompt = prompt + f'\n\nUser:\n"{user_query}"'

        print("  Calling LLM … ", end="", flush=True)
        import sys, threading

        _result   = [None]
        _done_evt = threading.Event()

        def _llm_thread():
            _result[0] = _call_llm(full_prompt, self.provider, self.model,
                                   timeout=self.llm_timeout)
            _done_evt.set()

        t = threading.Thread(target=_llm_thread, daemon=True)
        t.start()

        # Print a dot every 5 s so the user knows it's working
        warned = False
        while not _done_evt.wait(timeout=5):
            print(".", end="", flush=True)
            if not warned:
                print(" (model loading, please wait…)", end="", flush=True)
                warned = True

        print(" done.")
        generated = _result[0] or {"error": "No response from LLM"}

        if "error" in generated:
            err = generated["error"]
            print(f"\n  ⚠  LLM error: {err}")
            if "timed out" in err.lower() or "read timeout" in err.lower():
                print(f"\n  The model took longer than {self.llm_timeout}s.")
                print("  ┌─ Recommended fixes ──────────────────────────────────────────┐")
                print("  │  1. Use a smaller model (fastest option):                    │")
                print("  │       ollama pull qwen2:0.5b                                 │")
                print("  │       python smlp_agent_rl_session.py --model qwen2:0.5b     │")
                print("  │  2. Increase the timeout:                                    │")
                print("  │       python smlp_agent_rl_session.py --llm-timeout 300      │")
                print("  │  3. Type the JSON manually with [j] below (always works)     │")
                print("  └──────────────────────────────────────────────────────────────┘")
            else:
                print("  Enter the command manually with [j] below.")
            generated = {}

        _print_command(generated, "Generated command")

        # 2. User decides what to do
        print("\n  Options:")
        print("    [Enter]   accept as-is")
        print("    [e]       edit key-by-key in terminal")
        print("    [E]       open in $EDITOR")
        print("    [j]       paste replacement JSON")
        print("    [skip]    skip this query (no feedback recorded)")
        print("    [/quit]   save & exit")

        try:
            choice = input("\n  Your choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return self._handle_quit()

        if choice in ("/quit", "quit", "q"):
            return self._handle_quit()

        if choice == "skip":
            print("  Skipped — no feedback recorded.")
            return True

        corrected = deepcopy(generated)

        if choice == "e":
            corrected = _edit_inline(generated)
        elif choice == "E":
            corrected = _edit_in_editor(generated)
        elif choice == "j":
            print("  Paste JSON (end with a blank line):")
            lines = []
            while True:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if line == "":
                    break
                lines.append(line)
            try:
                corrected = json.loads("\n".join(lines))
            except json.JSONDecodeError as err:
                print(f"  Invalid JSON ({err}), keeping generated command.")
                corrected = deepcopy(generated)

        if corrected != generated:
            _print_command(corrected, "Corrected command")

        # 3. Execute
        exec_success, exec_output = _execute_smlp(corrected, self.dry_run)
        if self.dry_run:
            print(f"\n  {exec_output}")
        else:
            status = "✓ Success" if exec_success else "✗ Failed"
            print(f"  Execution: {status}")
            print(f"  (full output → smlp_last_run.log)")
            if exec_output:
                print()
                for line in exec_output.splitlines():
                    print(f"    {line}")
                print()

        # ── Detect whether the corrected command itself was invalid ───────────
        # If SMLP reported a bad option name / bad option value, then:
        #   • the user's correction was also wrong (they may not know SMLP well
        #     enough to provide reliable training signal)
        #   • we must not add an incorrect example to the prompt store
        #   • we skip the rating step entirely
        # If the failure was a runtime / data error the command was syntactically
        # valid — rating is still meaningful.
        corrected_cmd_invalid = (
            not self.dry_run
            and not exec_success
            and _is_command_error(exec_output)
        )

        if corrected_cmd_invalid:
            print("  ⚠  The corrected command itself was rejected by SMLP")
            print("     (bad option name or value — see output above).")
            print("     Rating and store addition skipped for this turn.")
            print("     Please fix the command and try the query again.")
            return True   # continue session, nothing recorded

        # 4. Reward — measures quality of LLM's *generated* command, not the
        #    corrected one.  exec_success reflects the corrected command running,
        #    so we do NOT pass it into the LLM reward (it would credit the user's
        #    fix to the LLM).  The edit-distance reward alone captures how close
        #    the LLM's output was to the correct answer.
        reward_model = RewardModel()
        auto_reward  = reward_model.compute_reward(
            generated, corrected,
            execution_success=False   # exec_success excluded — see above
        )
        # Re-show the LLM's original suggestion so the user knows what they're rating
        # (execution output may have pushed it off screen)
        if generated:
            _print_command(generated, "LLM suggestion (what you are rating)")
        else:
            print("  (LLM produced no suggestion — rating will be 0)")
        final_reward = _ask_reward_override(auto_reward)

        # 5. RL update (only if user gave a rating)
        result = None
        if final_reward is not None:
            original_compute = self.feedback_collector.reward_model.compute_reward
            self.feedback_collector.reward_model.compute_reward = \
                lambda *a, **kw: final_reward
            result = self.trainer.process_feedback(
                user_query        = user_query,
                generated_command = generated,
                corrected_command = corrected,
                execution_success = exec_success,
            )
            self.feedback_collector.reward_model.compute_reward = original_compute
            print(f"  Reward recorded: {final_reward:.3f}")
        else:
            print("  Evaluation skipped — no feedback recorded for this turn.")

        # 6. Offer to add corrected command to example store
        #    Only if evaluation was not skipped and reward is high enough.
        effective_reward = final_reward if final_reward is not None else auto_reward
        if corrected and effective_reward >= 0.6:
            try:
                add = input("  Add corrected command to prompt store? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                add = "n"
            if add in ("", "y", "yes"):
                ex_id = self.store.add_example(
                    user_text    = user_query,
                    smlp_command = corrected,
                    tag          = corrected.get("mode", "rl_generated"),
                )
                print(f"  ✓ Added to store as {ex_id}")
                self.prompt_optimizer.example_pool = \
                    self.store.get_examples_as_pool()

        if result and result["should_fine_tune"]:
            print("\n  ★ Fine-tuning threshold reached — consider running")
            print("    fine_tune_from_feedback() to update the Ollama model.")

        # 7. Auto-save checkpoint every 5 recorded feedbacks
        n_feedback = len(self.feedback_collector.feedback_buffer)
        if n_feedback > 0 and n_feedback % 5 == 0:
            self.trainer.save_checkpoint(self.checkpoint)
            print(f"  (checkpoint auto-saved → {self.checkpoint}/)")

        return True

    # ── session-level commands ────────────────────────────────────────────

    def _handle_quit(self):
        print("\n  Saving checkpoint …")
        self.trainer.save_checkpoint(self.checkpoint)
        self.store.save_to_file()
        print(f"  ✓ Checkpoint saved to {self.checkpoint}/")
        print(f"  ✓ Example store saved to {self.store.filepath}")
        return False   # signals the REPL to exit

    def show_stats(self):
        _print_stats(self.trainer)

    def show_prompt(self):
        print()
        prompt = self.trainer.get_current_prompt()
        for line in prompt.split("\n"):
            print("  " + line)

    def show_store(self):
        self.store.print_summary()

    def save(self):
        self.trainer.save_checkpoint(self.checkpoint)
        self.store.save_to_file()
        print("  ✓ Saved.")


# ═════════════════════════════════════════════════════════════════════════════
# REPL
# ═════════════════════════════════════════════════════════════════════════════

HELP_TEXT = """
  SMLP Agent Interactive RL Session
  ───────────────────────────────────
  <query>    Natural language description of an SMLP command
  /stats     Show RL training statistics
  /prompt    Print the current few-shot prompt
  /store     Print the example store table
  /save      Save checkpoint and store now
  /help      Show this message
  /quit      Save and exit
"""

def run_repl(args):
    print(SEP)
    print("  SMLP Agent — Interactive RL Session")
    if args.dry_run:
        print("  Mode: DRY-RUN (SMLP will not be executed)")
    else:
        print("  Mode: LIVE (commands will be passed to run_smlp.py)")
    print(f"  Provider: {args.provider}   Model: {args.model}   Timeout: {args.llm_timeout}s")
    print(SEP)
    print(HELP_TEXT)

    session = InteractiveRLSession(
        provider    = args.provider,
        model       = args.model,
        dry_run     = args.dry_run,
        checkpoint  = args.checkpoint,
        store_file  = args.store,
        llm_timeout = args.llm_timeout,
    )

    print(f"\n{THIN}")
    print("  Ready. Type your SMLP query, or /help for commands.")
    print(THIN)

    while True:
        try:
            raw = input("\n  Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            session._handle_quit()
            break

        if not raw:
            continue

        # Session-level slash commands
        if raw == "/quit":
            session._handle_quit()
            break
        elif raw == "/stats":
            session.show_stats()
        elif raw == "/prompt":
            session.show_prompt()
        elif raw == "/store":
            session.show_store()
        elif raw == "/save":
            session.save()
        elif raw == "/help":
            print(HELP_TEXT)
        elif raw.startswith("/"):
            print(f"  Unknown command: {raw}  (try /help)")
        else:
            # Normal query — run the RL turn
            keep_going = session.step(raw)
            if not keep_going:
                break

    print("\n  Session ended. Goodbye.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SMLP Agent Interactive RL Session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples
            --------
              # Recommended — uses deepseek-r1:1.5b by default:
              python smlp_agent_rl_session.py --dry-run

              # Mistral gives better JSON but needs more time:
              python smlp_agent_rl_session.py --model mistral --llm-timeout 300

              # Live execution (calls run_smlp.py):
              python smlp_agent_rl_session.py

              # OpenAI backend:
              python smlp_agent_rl_session.py --provider openai --model gpt-4o

            Installed models (from `ollama list`)
            --------------------------------------
              deepseek-r1:1.5b   1.1 GB   ~10-20s on CPU   ← default (recommended)
              mistral             4.1 GB   ~90-180s on CPU  (use --llm-timeout 300)
              nomic-embed-text    274 MB   embedding only   (cannot generate text)
        """)
    )
    parser.add_argument("--provider",     default="ollama",
                        choices=["ollama", "openai"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model",        default="deepseek-r1:1.5b",
                        help="Model name (default: deepseek-r1:1.5b). "
                             "Must be already pulled via `ollama pull <model>`.")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Skip actual SMLP execution")
    parser.add_argument("--llm-timeout",  type=int, default=60,
                        dest="llm_timeout",
                        help="Seconds to wait for LLM response (default: 60). "
                             "Use 300 for mistral on CPU.")
    parser.add_argument("--checkpoint",   default=CHECKPOINT,
                        help="Checkpoint directory (default: ./smlp_rl_checkpoints)")
    parser.add_argument("--store",        default=STORE_FILE,
                        help="Example store JSON (default: ./smlp_few_shot_examples.json)")
    args = parser.parse_args()
    run_repl(args)


if __name__ == "__main__":
    main()
