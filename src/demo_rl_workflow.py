#!/usr/bin/env python3
"""
SMLP Agent RL Enhancement – Persistent Demo
============================================

Each run of this script:
  1. Loads the example store from smlp_few_shot_examples.json
     (includes the real SMLP prompts you provided)
  2. Loads any feedback already collected from smlp_rl_feedback.jsonl
  3. Runs 4 simulated interactions
  4. Writes updated success_rates back into smlp_few_shot_examples.json

Run it multiple times:
  python demo_rl_workflow.py

After each run you will see the success_rate values in the JSON file change.
"""

import json
import textwrap
from smlp_agent_rl import (
    ExampleStore, FeedbackCollector, PromptOptimizer, RLTrainer
)

SEP  = "=" * 70
THIN = "-" * 70

STORE_FILE    = "./smlp_few_shot_examples.json"
FEEDBACK_FILE = "./smlp_rl_feedback.jsonl"
CHECKPOINT    = "./smlp_rl_checkpoints"

def bar(r, w=24):
    f = int(round(r * w))
    return f"[{'█'*f}{'░'*(w-f)}] {r:.3f}"

def section(title):
    print(f"\n{THIN}\n  {title}\n{THIN}")


# ═══════════════════════════════════════════════════════════════════════
print(SEP)
print("  SMLP Agent – RL Enhancement Demo  (persistent across runs)")
print(SEP)

# ── 1. Load example store (the prompt container) ──────────────────────
section("PART 1 · Loading Example Store")

store = ExampleStore(STORE_FILE)
store.print_summary()

# ── 2. Wire up RL components, seeded from store ────────────────────────
section("PART 2 · Initialising RL Components")

feedback_collector = FeedbackCollector(storage_path=FEEDBACK_FILE)
prompt_optimizer   = PromptOptimizer(
    max_examples        = 100,
    examples_per_prompt = 3,
    exploration_factor  = 0.5,
    example_store       = store          # ← seeded from JSON, writes back to it
)
rl_trainer = RLTrainer(
    feedback_collector, prompt_optimizer,
    fine_tune_threshold = 10
)

# Load previous checkpoint so rewards accumulate across runs
rl_trainer.load_checkpoint(CHECKPOINT)

total_before = len(feedback_collector.feedback_buffer)
print(f"\n  Feedback already in buffer : {total_before}")
print(f"  Examples in pool           : {len(prompt_optimizer.example_pool)}")

# ── 3. Show current prompt (uses real system_prompt from JSON) ─────────
section("PART 3 · Current Prompt Sent to the LLM")
prompt = rl_trainer.get_current_prompt("Train and optimize a DT model")
print()
for line in prompt.split("\n"):
    print("  " + line)

# ── 4. Simulate interactions (realistic SMLP corrections) ─────────────
section("PART 4 · Simulating User Interactions This Run")

# These interactions mirror real SMLP usage based on your examples.
# generated = what the LLM produced;  corrected = what the user fixed it to.
THIS_RUN = [
    (
        "Train and lazy-optimize a DT model on '../regr_smlp/data/smlp_toy_basic.csv' "
        "using spec '../regr_smlp/specs/smlp_toy_basic.spec', prefix 'run_a'. "
        "Nested tree encoding, epsilon=0.1, disable plots.",
        # generated – LLM got mode/data/spec right but missed tree_encoding + plots
        {"mode": "optimize", "pareto": True, "model": "dt_sklearn",
         "data": "../regr_smlp/data/smlp_toy_basic.csv",
         "spec": "../regr_smlp/specs/smlp_toy_basic.spec",
         "opt_strategy": "lazy", "epsilon": 0.1, "pref": "run_a"},
        # corrected – user adds the two missing keys
        {"mode": "optimize", "pareto": True, "model": "dt_sklearn",
         "data": "../regr_smlp/data/smlp_toy_basic.csv",
         "spec": "../regr_smlp/specs/smlp_toy_basic.spec",
         "opt_strategy": "lazy", "tree_encoding": "nested",
         "epsilon": 0.1, "pref": "run_a", "plots": False},
        True,
    ),
    (
        "Verify model 'modelABC' against spec '../regr_smlp/specs/my_spec.spec' "
        "on data '../regr_smlp/data/validation.csv'. delta=0.02, radius=0.1, prefix 'vrun2'.",
        # generated – mode/spec/data correct; delta key wrong name
        {"mode": "verify", "model_name": "modelABC",
         "spec": "../regr_smlp/specs/my_spec.spec",
         "data": "../regr_smlp/data/validation.csv",
         "delta_abs": 0.02, "rad_rel": 0.1, "pref": "vrun2"},
        # corrected – delta_abs → delta_rel
        {"mode": "verify", "model_name": "modelABC",
         "spec": "../regr_smlp/specs/my_spec.spec",
         "data": "../regr_smlp/data/validation.csv",
         "delta_rel": 0.02, "rad_rel": 0.1, "pref": "vrun2"},
        True,
    ),
    (
        "Use the LC RAG model at /models/smlp_rag_v2 to answer "
        "'How does SMLP handle categorical features?' "
        "from PDF /docs/smlp_manual_v2.pdf. Use cosine index.",
        # generated – all keys present but wrong rag_eval value
        {"mode": "rag", "rag_type": "lc",
         "rag_text": "/docs/smlp_manual_v2.pdf",
         "rag_trained_model_path": "/models/smlp_rag_v2",
         "questions": "How does SMLP handle categorical features?",
         "index_backend": "cosine", "do_sample": False, "rag_eval": False},
        # corrected
        {"mode": "rag", "rag_type": "lc",
         "rag_text": "/docs/smlp_manual_v2.pdf",
         "rag_trained_model_path": "/models/smlp_rag_v2",
         "questions": "How does SMLP handle categorical features?",
         "index_backend": "cosine", "do_sample": False, "rag_eval": True},
        True,
    ),
    (
        "Eager pareto-optimize a DT model on '../regr_smlp/data/smlp_toy_num_resp_mult.csv', "
        "responses y1,y2,y3, features x1,x2,x3. Spec '../regr_smlp/specs/mult.spec', "
        "prefix 'p_opt', epsilon=0.03, delta_abs=0.005. Save as model_m3. Flat tree. No plots.",
        # generated – near-perfect, only y3/x3 missing from resp/feat
        {"mode": "optimize", "pareto": True, "model": "dt_sklearn",
         "data": "../regr_smlp/data/smlp_toy_num_resp_mult.csv",
         "spec": "../regr_smlp/specs/mult.spec",
         "resp": "y1,y2", "feat": "x1,x2",
         "opt_strategy": "eager", "tree_encoding": "flat",
         "epsilon": 0.03, "delta_abs": 0.005,
         "pref": "p_opt", "save_model": True, "model_name": "model_m3",
         "plots": False},
        # corrected
        {"mode": "optimize", "pareto": True, "model": "dt_sklearn",
         "data": "../regr_smlp/data/smlp_toy_num_resp_mult.csv",
         "spec": "../regr_smlp/specs/mult.spec",
         "resp": "y1,y2,y3", "feat": "x1,x2,x3",
         "opt_strategy": "eager", "tree_encoding": "flat",
         "epsilon": 0.03, "delta_abs": 0.005,
         "pref": "p_opt", "save_model": True, "model_name": "model_m3",
         "plots": False},
        True,
    ),
]

print(f"\n  {'#':<3}  {'Reward':<28}  {'Pool':>4}  Query (truncated)")
print(f"  {'─'*3}  {'─'*28}  {'─'*4}  {'─'*38}")

rewards_this_run = []
for i, (query, generated, corrected, success) in enumerate(THIS_RUN, 1):
    result = rl_trainer.process_feedback(query, generated, corrected, success)
    r      = result["feedback"]["reward"]
    pool_n = result["stats"]["num_examples_in_pool"]
    rewards_this_run.append(r)
    flag = "  ← fine-tune!" if result["should_fine_tune"] else ""
    print(f"  {i:<3}  {bar(r)}  {pool_n:>4}  {query[:38]}{flag}")

# ── 5. Training stats across all runs ─────────────────────────────────
section("PART 5 · Cumulative Training Stats (all runs combined)")
stats = rl_trainer.get_training_stats()
print(f"  Total feedback ever collected : {stats['total_feedback']}")
print(f"  Overall average reward        : {stats['avg_reward']:.3f}")
print(f"  Recent average reward         : {stats['recent_avg_reward']:.3f}")
print(f"  Pool size                     : {stats['num_examples']}")
print(f"  This run rewards              : "
      f"{' | '.join(f'{r:.3f}' for r in rewards_this_run)}")

# ── 6. Updated example store (success_rates changed) ──────────────────
section("PART 6 · Example Store After This Run")
store.print_summary()

# ── 7. Evolved prompt ─────────────────────────────────────────────────
section("PART 7 · Updated Prompt for Next LLM Call")
new_prompt = rl_trainer.get_current_prompt("optimize a DT model")
print()
for line in new_prompt.split("\n"):
    print("  " + line)

# ── 8. Save checkpoint ────────────────────────────────────────────────
section("PART 8 · Saving Checkpoint")
rl_trainer.save_checkpoint(CHECKPOINT)
print(f"  ✓ Feedback log  : {FEEDBACK_FILE}")
print(f"  ✓ Example store : {STORE_FILE}  (success_rates updated in-place)")
print(f"  ✓ Checkpoint    : {CHECKPOINT}/")
print()
print("  Run this script again – the rewards will shift as the RL")
print("  system accumulates feedback and re-ranks examples via UCB.")
print(SEP)
