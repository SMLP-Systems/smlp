# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
SMLP Agent RL Enhancement Module

This module implements Reinforcement Learning capabilities to improve the SMLP Agent's
text-to-command conversion through user feedback and dynamic few-shot prompt optimization.

Key Components:
1. FeedbackCollector: Captures user corrections and computes rewards
2. RewardModel: Evaluates quality of generated commands
3. PromptOptimizer: Uses RL to select and evolve few-shot examples
4. RLTrainer: Orchestrates the RL training loop
"""

import json
import datetime
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import pickle
import os


@dataclass
class FeedbackEntry:
    """Stores a single user feedback instance"""
    timestamp: str
    user_query: str
    generated_command: Dict
    corrected_command: Dict
    execution_success: bool
    reward: float
    metadata: Dict = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class FewShotExample:
    """Represents a few-shot training example"""
    user_text: str
    smlp_command: Dict
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: str = None
    embedding: Optional[np.ndarray] = None  # For similarity-based retrieval
    
    def to_dict(self):
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d
    
    @classmethod
    def from_dict(cls, data):
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)



class ExampleStore:
    """
    Persistent, RL-tracked store of few-shot examples backed by a JSON file.

    This is the single source of truth for the prompt container. The file
    keeps every example readable/editable by hand while the RL system
    updates success_rate and usage_count in-place after every feedback round.

    Workflow
    --------
    1. On startup:      load_from_file()        reads examples + existing RL scores
    2. On each LLM call: get_examples_as_pool()  converts store → FewShotExample list
    3. After feedback:  update_scores()          writes updated scores back to JSON
    4. User correction: add_example()            appends new entry to JSON

    File
    ----
    Pass the path to __init__.  Recommended location alongside the agent:
        ./smlp_few_shot_examples.json
    """

    def __init__(self, filepath: str = "./smlp_few_shot_examples.json"):
        self.filepath = filepath
        self._data: dict = {"_meta": {}, "examples": []}
        self.system_prompt: str = ""
        self.valid_modes: List[str] = []

        if os.path.exists(filepath):
            self.load_from_file()
        else:
            print(f"[ExampleStore] No file at {filepath}. Starting empty.")

    # -- file I/O ----------------------------------------------------------

    def load_from_file(self):
        """Load examples and RL scores from the JSON file."""
        with open(self.filepath, "r") as f:
            self._data = json.load(f)
        meta = self._data.get("_meta", {})
        self.system_prompt = meta.get("system_prompt", "")
        self.valid_modes   = meta.get("valid_modes", [])
        print(f"[ExampleStore] Loaded {len(self._data['examples'])} examples "
              f"from {self.filepath}")

    def save_to_file(self):
        """Write state (including updated RL scores) back to JSON."""
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    # -- read --------------------------------------------------------------

    def get_examples_as_pool(self) -> List:
        """Return all stored examples as FewShotExample objects."""
        pool = []
        for entry in self._data["examples"]:
            ex = FewShotExample(
                user_text    = entry["user_text"],
                smlp_command = entry["smlp_command"],
                success_rate = entry.get("success_rate", 0.5),
                usage_count  = entry.get("usage_count", 0),
                last_used    = entry.get("last_used"),
            )
            pool.append(ex)
        return pool

    def count(self) -> int:
        return len(self._data["examples"])

    def list_tags(self) -> List[str]:
        return sorted(set(e.get("tag","") for e in self._data["examples"]))

    # -- write -------------------------------------------------------------

    def add_example(self, user_text: str, smlp_command: Dict,
                    tag: str = "rl_generated",
                    initial_success_rate: float = 0.6,
                    ex_id: str = None) -> str:
        """Append a new example (e.g. from a user correction) and save."""
        if ex_id is None:
            ex_id = f"ex_{self.count() + 1:04d}"
        entry = {
            "id":           ex_id,
            "tag":          tag,
            "status":       "rl_generated",
            "source":       "user_correction",
            "user_text":    user_text,
            "smlp_command": smlp_command,
            "success_rate": initial_success_rate,
            "usage_count":  0,
            "last_used":    None,
            "added":        datetime.datetime.now().isoformat(),
        }
        self._data["examples"].append(entry)
        self.save_to_file()
        return ex_id

    def update_scores(self, updated_examples: List):
        """
        Write back updated success_rate / usage_count from FewShotExample
        objects matched by user_text, then save.
        """
        text_to_entry = {e["user_text"]: e for e in self._data["examples"]}
        for ex in updated_examples:
            if ex.user_text in text_to_entry:
                text_to_entry[ex.user_text]["success_rate"] = ex.success_rate
                text_to_entry[ex.user_text]["usage_count"]  = ex.usage_count
                text_to_entry[ex.user_text]["last_used"]    = ex.last_used
        self.save_to_file()

    def remove_low_performers(self, min_rate: float = 0.3):
        """Drop RL-generated examples whose success_rate fell below threshold."""
        before = self.count()
        self._data["examples"] = [
            e for e in self._data["examples"]
            if e.get("source") == "hand_crafted"
            or e.get("success_rate", 0.5) >= min_rate
        ]
        removed = before - self.count()
        if removed:
            self.save_to_file()
            print(f"[ExampleStore] Pruned {removed} low-performing examples.")

    def print_summary(self):
        """Print a human-readable table of the example pool."""
        print(f"\n[ExampleStore]  file: {self.filepath}   total: {self.count()}")
        print(f"  {'ID':<10} {'tag':<12} {'source':<15} "
              f"{'success':>7} {'used':>5}  query (first 60 chars)")
        print("  " + "─" * 76)
        for e in sorted(self._data["examples"],
                        key=lambda x: x.get("success_rate", 0), reverse=True):
            print(f"  {e.get('id',''):<10} {e.get('tag',''):<12} "
                  f"{e.get('source',''):<15} "
                  f"{e.get('success_rate', 0):>7.3f} "
                  f"{e.get('usage_count', 0):>5}  "
                  f"{e['user_text'][:60]}")


class RewardModel:
    """
    Computes rewards for generated commands based on multiple criteria.
    
    Reward components:
    1. Edit distance between generated and corrected command
    2. Execution success
    3. Semantic similarity (optional)
    """
    
    def __init__(self, 
                 edit_weight: float = 0.5,
                 success_weight: float = 0.3,
                 semantic_weight: float = 0.2):
        self.edit_weight = edit_weight
        self.success_weight = success_weight
        self.semantic_weight = semantic_weight
    
    def compute_reward(self, 
                      generated: Dict, 
                      corrected: Dict,
                      execution_success: bool,
                      semantic_sim: float = 0.0) -> float:
        """
        Compute reward score in range [0, 1]
        
        Args:
            generated: LLM-generated command dict
            corrected: User-corrected command dict
            execution_success: Whether execution was successful
            semantic_sim: Optional semantic similarity score
            
        Returns:
            Reward score between 0 and 1
        """
        # 1. Edit-based reward (inverse of normalized edit distance)
        edit_reward = self._compute_edit_reward(generated, corrected)
        
        # 2. Execution success reward
        success_reward = 1.0 if execution_success else 0.0
        
        # 3. Semantic similarity (if available)
        semantic_reward = semantic_sim
        
        # Weighted combination
        total_reward = (
            self.edit_weight * edit_reward +
            self.success_weight * success_reward +
            self.semantic_weight * semantic_reward
        )
        
        return total_reward
    
    def _compute_edit_reward(self, generated: Dict, corrected: Dict) -> float:
        """
        Compute edit-based reward using recall-weighted scoring.

        The key insight: the corrected command is the ground truth.
        We want to reward how much of the *correct* answer the LLM got right,
        not penalise it for every key the user happened to add.

        Scoring:
          - correct_key_present_and_correct_value : +1.0  (full credit)
          - correct_key_present_but_wrong_value   : +0.5  (partial credit)
          - correct_key_missing entirely          : +0.0  (no credit)

        Result is normalised by the number of keys in corrected (ground truth).
        Extra spurious keys in generated are ignored (they get handled by
        execution success / downstream validation).
        """
        if not generated and not corrected:
            return 1.0
        if not corrected:          # corrected is empty but generated is not
            return 0.0
        if not generated:          # generated is empty, corrected has content
            return 0.0

        score = 0.0
        for key in corrected:
            if key not in generated:
                score += 0.0       # missing key
            elif generated[key] == corrected[key]:
                score += 1.0       # exact match
            else:
                score += 0.5       # key present, value wrong

        return score / len(corrected)
    
    def compute_levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            return self.compute_levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class FeedbackCollector:
    """
    Collects and manages user feedback on generated commands
    """
    
    def __init__(self, storage_path: str = "./smlp_rl_feedback.jsonl"):
        self.storage_path = storage_path
        self.reward_model = RewardModel()
        self.feedback_buffer = deque(maxlen=1000)  # Keep last 1000 entries in memory
        
        # Load existing feedback if available
        self._load_feedback()
    
    def collect_feedback(self,
                        user_query: str,
                        generated_command: Dict,
                        corrected_command: Dict,
                        execution_success: bool = False,
                        metadata: Dict = None) -> FeedbackEntry:
        """
        Collect user feedback and compute reward
        
        Args:
            user_query: Original natural language query
            generated_command: LLM-generated command dict
            corrected_command: User-corrected command dict
            execution_success: Whether command executed successfully
            metadata: Additional context
            
        Returns:
            FeedbackEntry object
        """
        # Compute reward
        reward = self.reward_model.compute_reward(
            generated_command,
            corrected_command,
            execution_success
        )
        
        # Create feedback entry
        entry = FeedbackEntry(
            timestamp=datetime.datetime.now().isoformat(),
            user_query=user_query,
            generated_command=generated_command,
            corrected_command=corrected_command,
            execution_success=execution_success,
            reward=reward,
            metadata=metadata or {}
        )
        
        # Store in buffer and file
        self.feedback_buffer.append(entry)
        self._save_feedback(entry)
        
        return entry
    
    def get_recent_feedback(self, n: int = 100) -> List[FeedbackEntry]:
        """Get n most recent feedback entries"""
        return list(self.feedback_buffer)[-n:]
    
    def get_high_quality_examples(self, 
                                  min_reward: float = 0.8,
                                  n: int = 50) -> List[FeedbackEntry]:
        """
        Get high-quality examples for few-shot learning
        
        Args:
            min_reward: Minimum reward threshold
            n: Maximum number of examples to return
            
        Returns:
            List of high-quality feedback entries
        """
        high_quality = [
            entry for entry in self.feedback_buffer 
            if entry.reward >= min_reward
        ]
        
        # Sort by reward (descending)
        high_quality.sort(key=lambda x: x.reward, reverse=True)
        
        return high_quality[:n]
    
    def _save_feedback(self, entry: FeedbackEntry):
        """Append feedback entry to JSONL file"""
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
    
    def _load_feedback(self):
        """Load existing feedback from file"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = FeedbackEntry.from_dict(json.loads(line))
                        self.feedback_buffer.append(entry)
            print(f"Loaded {len(self.feedback_buffer)} feedback entries")
        except Exception as e:
            print(f"Error loading feedback: {e}")


class PromptOptimizer:
    """
    Optimizes few-shot prompt using RL-based example selection
    
    Strategy:
    1. Maintain a pool of few-shot examples with quality scores
    2. Use multi-armed bandit (UCB) to select examples
    3. Update scores based on feedback
    4. Periodically prune low-performing examples
    """
    
    def __init__(self,
                 max_examples: int = 100,
                 examples_per_prompt: int = 5,
                 exploration_factor: float = 0.5,
                 example_store: "Optional[ExampleStore]" = None):
        """
        Args:
            max_examples: Maximum examples to keep in pool
            examples_per_prompt: Number of examples to include in each prompt
            exploration_factor: UCB exploration parameter
            example_store: Optional ExampleStore; if provided the pool is
                           seeded from it on startup and scores are written
                           back after every feedback round.
        """
        self.max_examples = max_examples
        self.examples_per_prompt = examples_per_prompt
        self.exploration_factor = exploration_factor
        self.example_store: Optional[ExampleStore] = example_store

        self.example_pool: List[FewShotExample] = []
        self.total_selections = 0

        # Seed pool from ExampleStore if provided
        if example_store is not None:
            self.example_pool = example_store.get_examples_as_pool()
            print(f"[PromptOptimizer] Seeded pool with "
                  f"{len(self.example_pool)} examples from ExampleStore.")
        
    def add_example(self, user_text: str, smlp_command: Dict,
                    tag: str = "rl_generated"):
        """
        Add a new example to the in-memory pool AND to the JSON ExampleStore
        so it persists across runs.
        """
        # Avoid exact duplicates
        for existing in self.example_pool:
            if existing.user_text == user_text:
                return  # Already in pool

        example = FewShotExample(
            user_text    = user_text,
            smlp_command = smlp_command,
            success_rate = 0.6,  # Slightly above neutral – user confirmed it
            usage_count  = 0,
            last_used    = datetime.datetime.now().isoformat()
        )
        self.example_pool.append(example)

        # Persist to JSON
        if self.example_store is not None:
            self.example_store.add_example(user_text, smlp_command, tag=tag)

        # Prune if pool is too large
        if len(self.example_pool) > self.max_examples:
            self._prune_examples()
    
    def select_examples(self, query: str = None) -> List[FewShotExample]:
        """
        Select examples for the few-shot prompt using UCB algorithm
        
        Args:
            query: Optional query for similarity-based selection
            
        Returns:
            List of selected examples
        """
        if len(self.example_pool) <= self.examples_per_prompt:
            return self.example_pool.copy()
        
        # Compute UCB scores for each example
        ucb_scores = []
        for example in self.example_pool:
            # UCB1 formula: mean_reward + C * sqrt(ln(total) / n_example)
            if example.usage_count == 0:
                ucb_score = float('inf')  # Always try unused examples
            else:
                exploitation = example.success_rate
                exploration = self.exploration_factor * np.sqrt(
                    np.log(self.total_selections + 1) / example.usage_count
                )
                ucb_score = exploitation + exploration
            
            ucb_scores.append((ucb_score, example))
        
        # Sort by UCB score and select top examples
        ucb_scores.sort(reverse=True, key=lambda x: x[0])
        selected = [ex for _, ex in ucb_scores[:self.examples_per_prompt]]
        
        # Update usage counts
        self.total_selections += 1
        for ex in selected:
            ex.usage_count += 1
            ex.last_used = datetime.datetime.now().isoformat()
        
        return selected
    
    def update_example_scores(self,
                             selected_examples: List[FewShotExample],
                             reward: float):
        """
        Update success rates of selected examples based on reward using
        exponential moving average, then persist changes to ExampleStore.
        """
        alpha = 0.1  # EMA learning rate

        for example in selected_examples:
            for pool_ex in self.example_pool:
                if pool_ex.user_text == example.user_text:
                    pool_ex.success_rate = (
                        (1 - alpha) * pool_ex.success_rate + alpha * reward
                    )
                    break

        # Persist updated scores back to JSON
        if self.example_store is not None:
            self.example_store.update_scores(self.example_pool)
    
    def _prune_examples(self):
        """Remove low-performing examples to maintain pool size"""
        # Sort by success rate
        self.example_pool.sort(key=lambda x: x.success_rate, reverse=True)
        
        # Keep top performers
        self.example_pool = self.example_pool[:self.max_examples]
    
    def generate_prompt(self,
                        selected_examples: List[FewShotExample],
                        example_store: "Optional[ExampleStore]" = None) -> str:
        """
        Generate the few-shot prompt that will be sent to the LLM.

        The system preamble is taken from ExampleStore._meta.system_prompt if
        an ExampleStore is provided; otherwise a generic fallback is used.
        This means editing smlp_few_shot_examples.json is the single place
        to change the preamble without touching code.

        Format
        ------
        <system_preamble>

        Examples:
        #1 <tag>
        Input:
        "<user_text>"
        Output:
        { ... }

        Now convert the following description into a JSON CLI options dict.
        Output ONLY the JSON, nothing else.
        """
        # --- preamble -------------------------------------------------------
        if example_store and example_store.system_prompt:
            preamble = example_store.system_prompt
        else:
            preamble = (
                "You are an assistant for SMLP. Convert the user's description "
                "into a JSON of CLI-style options dictionary. Output ONLY the JSON."
            )

        prompt_parts = [preamble, "", "Examples:"]

        # --- few-shot examples ----------------------------------------------
        for i, example in enumerate(selected_examples, 1):
            # Try to recover tag from the example store for the label line
            tag = ""
            if example_store:
                for entry in example_store._data.get("examples", []):
                    if entry["user_text"] == example.user_text:
                        tag = entry.get("tag", "")
                        break
            label = f"#{i} {tag}".rstrip()
            prompt_parts.append(label)
            prompt_parts.append("Input:")
            prompt_parts.append(f'"{example.user_text}"')
            prompt_parts.append("Output:")
            prompt_parts.append(json.dumps(example.smlp_command, indent=2))
            prompt_parts.append("")

        # --- closing instruction --------------------------------------------
        prompt_parts.append(
            "Now convert the following description into a JSON CLI options "
            "dictionary. Output ONLY the JSON, nothing else."
        )
        return "\n".join(prompt_parts)
    
    def save_pool(self, filepath: str = "./smlp_rl_example_pool.pkl"):
        """Save example pool to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.example_pool, f)
    
    def load_pool(self, filepath: str = "./smlp_rl_example_pool.pkl"):
        """Load example pool from disk"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    self.example_pool = pickle.load(f)
                print(f"Loaded {len(self.example_pool)} examples from pool")
            except Exception as e:
                print(f"Error loading example pool: {e}")


class RLTrainer:
    """
    Orchestrates the RL training process for SMLP Agent
    
    Workflow:
    1. Collect user feedback
    2. Update prompt optimizer with rewards
    3. Generate new few-shot prompts
    4. Optionally trigger model fine-tuning
    """
    
    def __init__(self,
                 feedback_collector: FeedbackCollector,
                 prompt_optimizer: PromptOptimizer,
                 fine_tune_threshold: int = 100):
        """
        Args:
            feedback_collector: FeedbackCollector instance
            prompt_optimizer: PromptOptimizer instance
            fine_tune_threshold: Number of examples before triggering fine-tuning
        """
        self.feedback_collector = feedback_collector
        self.prompt_optimizer = prompt_optimizer
        self.fine_tune_threshold = fine_tune_threshold
        
        self.training_history = []
        self.current_examples = []
    
    def process_feedback(self,
                        user_query: str,
                        generated_command: Dict,
                        corrected_command: Dict,
                        execution_success: bool = False) -> Dict:
        """
        Process a single feedback instance
        
        Returns:
            Dictionary with feedback stats and updated prompt
        """
        # Collect feedback and compute reward
        feedback_entry = self.feedback_collector.collect_feedback(
            user_query=user_query,
            generated_command=generated_command,
            corrected_command=corrected_command,
            execution_success=execution_success
        )
        
        # Update example scores if we used few-shot examples
        if self.current_examples:
            self.prompt_optimizer.update_example_scores(
                self.current_examples,
                feedback_entry.reward
            )
        
        # Add corrected command as new example if high quality (threshold 0.6
        # so realistic partial-match feedback still grows the pool)
        if feedback_entry.reward >= 0.6:
            self.prompt_optimizer.add_example(
                user_text=user_query,
                smlp_command=corrected_command
            )
        
        # Check if we should trigger fine-tuning
        should_fine_tune = (
            len(self.feedback_collector.feedback_buffer) % 
            self.fine_tune_threshold == 0
        )
        
        # Generate new prompt AFTER pool has been updated
        self.current_examples = self.prompt_optimizer.select_examples(user_query)
        new_prompt = self.prompt_optimizer.generate_prompt(self.current_examples)
        
        # Log training step — pool size recorded AFTER add_example
        training_step = {
            'timestamp': datetime.datetime.now().isoformat(),
            'reward': feedback_entry.reward,
            'should_fine_tune': should_fine_tune,
            'num_examples_in_pool': len(self.prompt_optimizer.example_pool)  # now accurate
        }
        self.training_history.append(training_step)
        
        return {
            'feedback': feedback_entry.to_dict(),
            'new_prompt': new_prompt,
            'should_fine_tune': should_fine_tune,
            'stats': training_step
        }
    
    def get_current_prompt(self, query: str = None) -> str:
        """Get the current optimized prompt for a query, using ExampleStore
        system preamble if available."""
        self.current_examples = self.prompt_optimizer.select_examples(query)
        return self.prompt_optimizer.generate_prompt(
            self.current_examples,
            example_store=self.prompt_optimizer.example_store
        )
    
    def get_training_stats(self) -> Dict:
        """Get statistics about the training process"""
        if not self.training_history:
            return {
                'total_feedback': 0,
                'avg_reward': 0.0,
                'recent_avg_reward': 0.0,
                'num_examples': len(self.prompt_optimizer.example_pool),
                'total_selections': self.prompt_optimizer.total_selections
            }
        
        recent_history = self.training_history[-100:]  # Last 100 steps
        
        return {
            'total_feedback': len(self.training_history),
            'avg_reward': np.mean([s['reward'] for s in self.training_history]),
            'recent_avg_reward': np.mean([s['reward'] for s in recent_history]),
            'num_examples': len(self.prompt_optimizer.example_pool),
            'total_selections': self.prompt_optimizer.total_selections
        }
    
    def save_checkpoint(self, checkpoint_dir: str = "./smlp_rl_checkpoints"):
        """Save RL training state"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save prompt optimizer pool
        self.prompt_optimizer.save_pool(
            os.path.join(checkpoint_dir, f"example_pool_{timestamp}.pkl")
        )
        
        # Save training history
        with open(os.path.join(checkpoint_dir, f"training_history_{timestamp}.json"), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str = "./smlp_rl_checkpoints"):
        """Load latest RL training state"""
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory {checkpoint_dir} not found")
            return
        
        # Find latest checkpoint files
        pool_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("example_pool_")]
        history_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("training_history_")]
        
        if pool_files:
            latest_pool = sorted(pool_files)[-1]
            self.prompt_optimizer.load_pool(os.path.join(checkpoint_dir, latest_pool))
        
        if history_files:
            latest_history = sorted(history_files)[-1]
            with open(os.path.join(checkpoint_dir, latest_history), 'r') as f:
                self.training_history = json.load(f)
        
        print(f"Loaded checkpoint from {checkpoint_dir}")


# Utility functions for integration with existing SMLP Agent

def create_rl_enhanced_agent(base_agent, load_checkpoint: bool = True):
    """
    Enhance an existing SMLP Agent with RL capabilities
    
    Args:
        base_agent: Existing SmlpAgent instance
        load_checkpoint: Whether to load previous RL state
        
    Returns:
        Enhanced agent with RL capabilities
    """
    # Initialize RL components
    feedback_collector = FeedbackCollector()
    prompt_optimizer = PromptOptimizer()
    rl_trainer = RLTrainer(feedback_collector, prompt_optimizer)
    
    # Load previous state if requested
    if load_checkpoint:
        rl_trainer.load_checkpoint()
    
    # Attach RL components to agent
    base_agent.rl_feedback_collector = feedback_collector
    base_agent.rl_prompt_optimizer = prompt_optimizer
    base_agent.rl_trainer = rl_trainer
    
    return base_agent


def extract_feedback_from_correction(original_dict: Dict, 
                                     corrected_dict: Dict) -> Tuple[Dict, bool]:
    """
    Extract feedback information from user correction
    
    Args:
        original_dict: LLM-generated command
        corrected_dict: User-corrected command
        
    Returns:
        Tuple of (correction_summary, is_perfect_match)
    """
    all_keys = set(original_dict.keys()) | set(corrected_dict.keys())
    
    changes = {
        'added': {},
        'removed': {},
        'modified': {}
    }
    
    for key in all_keys:
        if key not in original_dict:
            changes['added'][key] = corrected_dict[key]
        elif key not in corrected_dict:
            changes['removed'][key] = original_dict[key]
        elif original_dict[key] != corrected_dict[key]:
            changes['modified'][key] = {
                'from': original_dict[key],
                'to': corrected_dict[key]
            }
    
    is_perfect = not any(changes.values())
    
    return changes, is_perfect


if __name__ == "__main__":
    # Example usage
    print("=== SMLP Agent RL Enhancement Module ===\n")
    
    # Initialize components
    feedback_collector = FeedbackCollector()
    prompt_optimizer = PromptOptimizer()
    rl_trainer = RLTrainer(feedback_collector, prompt_optimizer)
    
    # Simulate some feedback
    print("Simulating feedback collection...\n")
    
    test_cases = [
        {
            'query': 'run dataset analysis on sales data with neural network',
            'generated': {
                'analytics_mode': 'dataset',
                'model_name': 'nn',
                'data_file': 'sales.csv'
            },
            'corrected': {
                'analytics_mode': 'dataset',
                'model_name': 'nn',
                'data_file': 'sales_data.csv',
                'log_files_prefix': 'sales_analysis'
            },
            'success': True
        },
        {
            'query': 'verify model coverage for autonomous vehicle spec',
            'generated': {
                'analytics_mode': 'verify',
                'spec_file': 'av_spec.py'
            },
            'corrected': {
                'analytics_mode': 'verify',
                'spec_file': 'av_spec.py',
                'log_files_prefix': 'av_verification'
            },
            'success': True
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Processing feedback {i}...")
        result = rl_trainer.process_feedback(
            user_query=test['query'],
            generated_command=test['generated'],
            corrected_command=test['corrected'],
            execution_success=test['success']
        )
        print(f"  Reward: {result['feedback']['reward']:.3f}")
        print(f"  Examples in pool: {result['stats']['num_examples_in_pool']}")
        print()
    
    # Get training stats
    stats = rl_trainer.get_training_stats()
    print("Training Statistics:")
    print(f"  Total feedback: {stats['total_feedback']}")
    print(f"  Average reward: {stats['avg_reward']:.3f}")
    print(f"  Examples in pool: {stats['num_examples']}")
    print()
    
    # Get current optimized prompt
    print("Current optimized prompt:")
    print("=" * 60)
    prompt = rl_trainer.get_current_prompt("run dataset analysis")
    print(prompt)
    print("=" * 60)
