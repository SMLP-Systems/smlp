#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
SMLP Agent RL – FastAPI Backend Server
========================================

Serves both the Web UI and the RL feedback API.

Usage
-----
  # Dry-run (no SMLP execution):
  python smlp_agent_rl_server.py --dry-run

  # Live (calls run_smlp.py):
  python smlp_agent_rl_server.py --provider ollama --model mistral

  # OpenAI backend:
  python smlp_agent_rl_server.py --provider openai --model gpt-4o

Then open http://localhost:8000 in a browser.

Endpoints
---------
  GET  /                       Serve the Web UI (smlp_agent_web_ui.html)
  POST /generate               LLM → suggested command
  POST /feedback               Record user correction + reward
  POST /execute                Run the corrected command via run_smlp.py
  GET  /stats                  RL training statistics
  GET  /store                  Full example store contents
  GET  /prompt                 Current few-shot prompt text
  POST /store/add              Manually add example to store
  DELETE /store/{ex_id}        Remove example from store
"""

import argparse, json, os, subprocess, sys, textwrap
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from smlp_agent_rl import (
    ExampleStore, FeedbackCollector, PromptOptimizer, RLTrainer, RewardModel
)
from smlp_agent_utils import _call_llm, _execute_smlp, _filter_noise, _is_command_error

STORE_FILE    = "./smlp_few_shot_examples.json"
FEEDBACK_FILE = "./smlp_rl_feedback.jsonl"
CHECKPOINT    = "./smlp_rl_checkpoints"
UI_FILE       = "./smlp_agent_web_ui.html"


# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    user_query:        str
    generated_command: Dict
    corrected_command: Dict
    execution_success: bool  = False
    user_reward:       Optional[float] = None   # 0-5 scale; None = auto

class ExecuteRequest(BaseModel):
    command: Dict

class AddExampleRequest(BaseModel):
    user_text:    str
    smlp_command: Dict
    tag:          str = "rl_generated"


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(provider: str, model: str, dry_run: bool) -> FastAPI:

    app = FastAPI(title="SMLP Agent RL", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # ── Shared RL state (one instance for the lifetime of the server) ─────────
    store = ExampleStore(STORE_FILE)
    feedback_collector = FeedbackCollector(storage_path=FEEDBACK_FILE)
    prompt_optimizer   = PromptOptimizer(
        max_examples=100, examples_per_prompt=3,
        exploration_factor=0.5, example_store=store
    )
    trainer = RLTrainer(feedback_collector, prompt_optimizer,
                        fine_tune_threshold=20)
    trainer.load_checkpoint(CHECKPOINT)

    reward_model = RewardModel()

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/")
    def serve_ui():
        if not Path(UI_FILE).exists():
            raise HTTPException(404, f"{UI_FILE} not found")
        return FileResponse(UI_FILE, media_type="text/html")

    @app.post("/generate")
    def generate(req: GenerateRequest):
        """Call LLM with current RL-optimised prompt, return suggested command."""
        prompt      = trainer.get_current_prompt(req.query)
        # Append user query AFTER the closing instruction (same as CLI session)
        full_prompt = prompt + f'\n\nUser:\n"{req.query}"'
        generated   = _call_llm(full_prompt, provider, model)
        if "error" in generated:
            raise HTTPException(status_code=502,
                detail=f"LLM error: {generated['error']}")
        return {
            "generated_command": generated,
            "prompt_used":       prompt,
            "pool_size":         len(prompt_optimizer.example_pool),
        }

    @app.post("/feedback")
    def record_feedback(req: FeedbackRequest):
        """
        Record user correction + reward.  If user_reward is provided (0-5 scale)
        it overrides the auto-computed edit-distance reward.
        """
        auto_reward = reward_model.compute_reward(
            req.generated_command, req.corrected_command, req.execution_success
        )

        if req.user_reward is not None:
            # map 0-5 → 0-1
            final_reward = max(0.0, min(1.0, req.user_reward / 5))
        else:
            final_reward = auto_reward

        # Patch reward model to use our pre-computed value
        original = feedback_collector.reward_model.compute_reward
        feedback_collector.reward_model.compute_reward = \
            lambda *a, **kw: final_reward
        result = trainer.process_feedback(
            user_query        = req.user_query,
            generated_command = req.generated_command,
            corrected_command = req.corrected_command,
            execution_success = req.execution_success,
        )
        feedback_collector.reward_model.compute_reward = original

        # Auto-save checkpoint every 5 feedbacks
        total = len(feedback_collector.feedback_buffer)
        if total % 5 == 0:
            trainer.save_checkpoint(CHECKPOINT)

        stats = trainer.get_training_stats()
        return {
            "auto_reward":       auto_reward,
            "final_reward":      final_reward,
            "should_fine_tune":  result["should_fine_tune"],
            "total_feedback":    stats["total_feedback"],
            "avg_reward":        stats["avg_reward"],
            "recent_avg_reward": stats["recent_avg_reward"],
            "pool_size":         stats["num_examples"],
        }

    @app.post("/execute")
    def execute(req: ExecuteRequest):
        """Execute the command via run_smlp.py (respects server dry_run flag)."""
        # Build command line for display (same as _execute_smlp does internally)
        import sys
        args = []
        for k, v in req.command.items():
            args.append(f"-{k}")
            args.append(str(v))
        cmd_line = f"{sys.executable} -u ./run_smlp.py " + " ".join(args)
        
        success, output = _execute_smlp(req.command, dry_run)
        return {
            "success": success,
            "output": output,
            "command_line": cmd_line
        }

    @app.get("/stats")
    def get_stats():
        s = trainer.get_training_stats()
        return s

    @app.get("/store")
    def get_store():
        return {"examples": store._data["examples"],
                "total":    store.count()}

    @app.get("/prompt")
    def get_prompt():
        prompt = trainer.get_current_prompt()
        return {"prompt": prompt,
                "pool_size": len(prompt_optimizer.example_pool),
                "examples_per_prompt": prompt_optimizer.examples_per_prompt}

    @app.post("/store/add")
    def add_to_store(req: AddExampleRequest):
        ex_id = store.add_example(
            user_text=req.user_text,
            smlp_command=req.smlp_command,
            tag=req.tag,
        )
        # Sync in-memory pool
        prompt_optimizer.example_pool = store.get_examples_as_pool()
        return {"status": "added", "id": ex_id, "total": store.count()}

    @app.delete("/store/{ex_id}")
    def remove_from_store(ex_id: str):
        removed = store.remove_low_performers(min_rate=0.0)   # noop shortcut
        ok = store.remove_example(ex_id)
        if not ok:
            raise HTTPException(404, f"Example {ex_id} not found")
        prompt_optimizer.example_pool = store.get_examples_as_pool()
        return {"status": "removed", "id": ex_id, "total": store.count()}

    @app.get("/config")
    def get_config():
        return {
            "provider": provider,
            "model":    model,
            "dry_run":  dry_run,
        }

    return app


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SMLP Agent RL Server")
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "openai"])
    parser.add_argument("--model",    default="mistral")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Skip actual run_smlp.py execution")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     type=int, default=8000)
    args = parser.parse_args()

    print(f"Starting SMLP Agent RL server on http://{args.host}:{args.port}")
    print(f"  Provider : {args.provider} / {args.model}")
    print(f"  Dry-run  : {args.dry_run}")
    print(f"  UI       : http://{args.host}:{args.port}/")

    # Warmup: pre-load the model so first request doesn't time out
    if args.provider == "ollama":
        print(f"  Warming up {args.model}...", end="", flush=True)
        try:
            import requests
            requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": args.model,
                    "prompt": "test",
                    "stream": False,
                    "keep_alive": -1,  # Keep loaded indefinitely
                },
                timeout=120
            )
            print(" ready.")
        except Exception as e:
            print(f" failed: {e}")
            print(f"  Note: Run 'ollama run {args.model}' manually to pre-load.")
    
    app = create_app(args.provider, args.model, args.dry_run)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
