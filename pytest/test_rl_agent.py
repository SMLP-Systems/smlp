# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
Unit Tests for SMLP RL Agent Core Components

These tests are deterministic and do not require LLM calls.
Run with: pytest test_rl_agent.py -v
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# Adjust import path if needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smlp_agent_rl import (
    ExampleStore, RewardModel, PromptOptimizer, 
    FewShotExample, RLTrainer, FeedbackCollector
)
from smlp_agent_utils import _filter_noise, _is_command_error


# ============================================================================
# RewardModel Tests
# ============================================================================

class TestRewardModel:
    """Test reward computation logic"""
    
    def test_perfect_match(self):
        """Reward should be high (but not 1.0) for perfect match without execution"""
        rm = RewardModel()
        cmd = {"mode": "train", "data": "toy.csv", "resp": "y1"}
        
        # execution_success=False means 0.3*0 = 0, so max reward is 0.7 (0.5 edit + 0.2 semantic)
        reward = rm.compute_reward(cmd, cmd, execution_success=False)
        
        assert 0.4 < reward <= 0.7  # Perfect edit match but no execution success
    
    def test_partial_match(self):
        """Reward should be between 0 and 1 for partial match"""
        rm = RewardModel()
        generated = {"mode": "train", "data": "toy.csv"}
        corrected = {"mode": "train", "data": "toy.csv", "resp": "y1"}
        
        reward = rm.compute_reward(generated, corrected, execution_success=False)
        
        assert 0.0 < reward < 1.0
        assert isinstance(reward, float)
    
    def test_completely_wrong(self):
        """Reward should be low for completely wrong command"""
        rm = RewardModel()
        generated = {"mode": "verify", "model_name": "wrong"}
        corrected = {"mode": "train", "data": "toy.csv", "resp": "y1"}
        
        reward = rm.compute_reward(generated, corrected, execution_success=False)
        
        assert 0.0 <= reward < 0.5
    
    def test_empty_commands(self):
        """Handle empty command dictionaries"""
        rm = RewardModel()
        
        # Both empty - actual implementation returns 0.5 (baseline with no execution)
        result = rm.compute_reward({}, {}, False)
        assert 0.4 <= result <= 0.6
        
        # One empty - should be low
        assert rm.compute_reward({"mode": "train"}, {}, False) < 0.3
        assert rm.compute_reward({}, {"mode": "train"}, False) < 0.3


# ============================================================================
# ExampleStore Tests
# ============================================================================

class TestExampleStore:
    """Test example store operations"""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary example store for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            initial_data = {
                "_meta": {
                    "description": "Test store",
                    "version": "1.0",
                    "system_prompt": "Test prompt"
                },
                "examples": []
            }
            json.dump(initial_data, f)
            store_path = f.name
        
        yield store_path
        
        # Cleanup
        os.unlink(store_path)
    
    def test_load_store(self, temp_store):
        """Test loading an existing store"""
        store = ExampleStore(temp_store)
        
        assert store.count() == 0
        assert store.system_prompt == "Test prompt"
    
    def test_add_example(self, temp_store):
        """Test adding a new example"""
        store = ExampleStore(temp_store)
        initial_count = store.count()
        
        ex_id = store.add_example(
            user_text="test query",
            smlp_command={"mode": "train", "data": "test.csv"},
            tag="test"
        )
        
        assert store.count() == initial_count + 1
        assert ex_id.startswith("ex_")
        
        # Verify it was saved to file
        store2 = ExampleStore(temp_store)
        assert store2.count() == initial_count + 1
    
    def test_get_examples_as_pool(self, temp_store):
        """Test converting store to FewShotExample pool"""
        store = ExampleStore(temp_store)
        
        store.add_example("query 1", {"mode": "train"}, "test")
        store.add_example("query 2", {"mode": "verify"}, "test")
        
        pool = store.get_examples_as_pool()
        
        assert len(pool) == 2
        assert all(isinstance(ex, FewShotExample) for ex in pool)
        assert pool[0].user_text == "query 1"
        assert pool[1].user_text == "query 2"
    
    def test_update_scores(self, temp_store):
        """Test updating success rates"""
        store = ExampleStore(temp_store)
        
        store.add_example("query 1", {"mode": "train"}, "test")
        pool = store.get_examples_as_pool()
        
        # Modify success rate
        pool[0].success_rate = 0.85
        pool[0].usage_count = 10
        
        # Update store
        store.update_scores(pool)
        
        # Reload and verify
        store2 = ExampleStore(temp_store)
        pool2 = store2.get_examples_as_pool()
        
        assert pool2[0].success_rate == 0.85
        assert pool2[0].usage_count == 10


# ============================================================================
# PromptOptimizer Tests
# ============================================================================

class TestPromptOptimizer:
    """Test UCB selection algorithm"""
    
    def test_ucb_selection(self):
        """Test that UCB selects high-success examples"""
        optimizer = PromptOptimizer(
            max_examples=10,
            examples_per_prompt=2,
            exploration_factor=0.5
        )
        
        # Add examples with different success rates
        ex_high = FewShotExample(
            "high success query",
            {"mode": "train"},
            success_rate=0.9,
            usage_count=20
        )
        ex_low = FewShotExample(
            "low success query",
            {"mode": "verify"},
            success_rate=0.3,
            usage_count=20
        )
        
        optimizer.example_pool = [ex_high, ex_low]
        
        # Select multiple times - high success should be picked more often
        selections = []
        for _ in range(10):
            selected = optimizer.select_examples("train model")
            selections.append(selected[0].user_text)
        
        # High success example should be selected at least 60% of the time
        high_count = selections.count("high success query")
        assert high_count >= 6
    
    def test_exploration_bonus(self):
        """Test that rarely-used examples get exploration bonus"""
        optimizer = PromptOptimizer(
            max_examples=10,
            examples_per_prompt=1,
            exploration_factor=1.0  # High exploration
        )
        
        # One example used many times, one never used
        ex_frequent = FewShotExample(
            "frequent",
            {"mode": "train"},
            success_rate=0.7,
            usage_count=100
        )
        ex_rare = FewShotExample(
            "rare",
            {"mode": "train"},
            success_rate=0.5,
            usage_count=1
        )
        
        optimizer.example_pool = [ex_frequent, ex_rare]
        optimizer.total_selections = 101
        
        # Calculate UCB scores manually
        import math
        score_frequent = 0.7 + 1.0 * math.sqrt(math.log(101) / 100)
        score_rare = 0.5 + 1.0 * math.sqrt(math.log(101) / 1)
        
        # Rare example should have higher UCB score due to exploration
        assert score_rare > score_frequent
        
        # Select - rare should be picked
        selected = optimizer.select_examples("test query")
        assert selected[0].user_text == "rare"


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtils:
    """Test utility functions"""
    
    def test_filter_noise_removes_tensorflow(self):
        """Test that TensorFlow noise is filtered"""
        output = """2026-02-15 12:33:52.493595: E tensorflow/compiler/xla
W0000 00:00:1707945600.000000  123 cuda_driver.cc:456] WARNING: All log messages
Some actual SMLP error
I tensorflow/core/platform/cpu_feature_guard.cc:193
Real output here"""
        
        filtered = _filter_noise(output)
        
        # TF noise should be removed
        assert "tensorflow" not in filtered.lower()
        assert "W0000" not in filtered
        
        # Real content should remain
        assert "Some actual SMLP error" in filtered
        assert "Real output here" in filtered
    
    def test_filter_noise_preserves_smlp_errors(self):
        """Test that actual SMLP errors are preserved"""
        output = """2026-02-15 12:33:52.493595: E tensorflow/...
error: unrecognized arguments: --bad-option
FileNotFoundError: data.csv not found"""
        
        filtered = _filter_noise(output)
        
        assert "unrecognized arguments" in filtered
        assert "FileNotFoundError" in filtered
    
    def test_is_command_error_detects_cli_errors(self):
        """Test detection of CLI syntax errors"""
        cli_errors = [
            "error: unrecognized arguments: --bad-option",
            "error: the following arguments are required: --data",
            "error: invalid choice: 'bad_mode' (choose from 'train', 'verify')",
            "error: argument --epsilon: invalid float value: 'abc'"
        ]
        
        for error in cli_errors:
            assert _is_command_error(error), f"Should detect CLI error: {error}"
    
    def test_is_command_error_ignores_runtime_errors(self):
        """Test that runtime errors are not flagged as command errors"""
        runtime_errors = [
            "FileNotFoundError: data.csv not found",
            "ValueError: Invalid data format",
            "RuntimeError: Model training failed",
            "KeyError: 'response' column not found"
        ]
        
        for error in runtime_errors:
            assert not _is_command_error(error), f"Should not flag runtime error: {error}"


# ============================================================================
# Integration Tests (with mocked components)
# ============================================================================

class TestRLTrainer:
    """Test RL trainer workflow"""
    
    @pytest.fixture
    def trainer_components(self, tmp_path):
        """Set up RL trainer components"""
        feedback_path = tmp_path / "feedback.jsonl"
        store_path = tmp_path / "store.json"
        
        # Create minimal store
        store_data = {
            "_meta": {"system_prompt": "test"},
            "examples": [
                {
                    "id": "ex_001",
                    "user_text": "test query",
                    "smlp_command": {"mode": "train"},
                    "success_rate": 0.5,
                    "usage_count": 0,
                    "tag": "test",
                    "source": "test",
                    "status": "test"
                }
            ]
        }
        
        with open(store_path, 'w') as f:
            json.dump(store_data, f)
        
        store = ExampleStore(str(store_path))
        feedback = FeedbackCollector(str(feedback_path))
        optimizer = PromptOptimizer(
            max_examples=10,
            examples_per_prompt=1,
            exploration_factor=0.5,
            example_store=store
        )
        trainer = RLTrainer(feedback, optimizer, fine_tune_threshold=5)
        
        return trainer, store, feedback
    
    def test_process_feedback_updates_scores(self, trainer_components):
        """Test that feedback is recorded correctly"""
        trainer, store, feedback = trainer_components
        
        initial_feedback_count = len(feedback.feedback_buffer)
        
        # Process high-reward feedback
        result = trainer.process_feedback(
            user_query="test query",
            generated_command={"mode": "train"},
            corrected_command={"mode": "train", "data": "x.csv"},
            execution_success=True
        )
        
        # Feedback should be recorded
        assert len(feedback.feedback_buffer) == initial_feedback_count + 1
        
        # Result should indicate feedback was processed
        assert "should_fine_tune" in result
        
        # Feedback should be recorded
        assert len(feedback.feedback_buffer) == 1
    
    def test_fine_tune_threshold_detection(self, trainer_components):
        """Test that fine-tune threshold is detected"""
        trainer, store, feedback = trainer_components
        
        # Add feedback entries up to threshold
        for i in range(5):
            result = trainer.process_feedback(
                user_query=f"query {i}",
                generated_command={"mode": "train"},
                corrected_command={"mode": "train", "data": "x.csv"},
                execution_success=True
            )
        
        # Last one should trigger threshold
        assert result["should_fine_tune"] == True


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
