#!/usr/bin/env python3
"""
Complete End-to-End Example: SMLP Agent with RL Enhancement

This example demonstrates the entire workflow from initial setup
through feedback collection to fine-tuning.

Run this script to see the RL system in action!
"""

import json
import time
from smlp_agent_rl import (
    FeedbackCollector,
    PromptOptimizer,
    RLTrainer,
    RewardModel
)

print("="*70)
print("SMLP Agent - Reinforcement Learning Enhancement Demo")
print("="*70)
print()

# ============================================================================
# PART 1: Initialize RL Components
# ============================================================================
print("PART 1: Initializing RL Components")
print("-" * 70)

feedback_collector = FeedbackCollector(storage_path="./demo_feedback.jsonl")
prompt_optimizer = PromptOptimizer(
    max_examples=100,
    examples_per_prompt=5,
    exploration_factor=0.5
)
rl_trainer = RLTrainer(feedback_collector, prompt_optimizer)

print("✓ FeedbackCollector initialized")
print("✓ PromptOptimizer initialized")
print("✓ RLTrainer initialized")
print()

# ============================================================================
# PART 2: Simulate User Interactions
# ============================================================================
print("PART 2: Simulating User Interactions")
print("-" * 70)

# Sample interactions with the agent
interactions = [
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
        'query': 'verify autonomous vehicle specification',
        'generated': {
            'analytics_mode': 'verify',
            'spec_file': 'av_spec.py'
        },
        'corrected': {
            'analytics_mode': 'verify',
            'spec_file': 'av_spec.py',
            'log_files_prefix': 'av_verification',
            'model_name': 'dummy'
        },
        'success': True
    },
    {
        'query': 'optimize hyperparameters for the model',
        'generated': {
            'analytics_mode': 'optimize'
        },
        'corrected': {
            'analytics_mode': 'optimize',
            'model_name': 'nn',
            'log_files_prefix': 'hyperopt',
            'optimization_iterations': '100'
        },
        'success': True
    },
    {
        'query': 'explain model predictions on test data',
        'generated': {
            'analytics_mode': 'explain',
            'model_file': 'model.pkl'
        },
        'corrected': {
            'analytics_mode': 'explain',
            'model_file': 'trained_model.pkl',
            'data_file': 'test_data.csv',
            'log_files_prefix': 'explanation'
        },
        'success': True
    },
    {
        'query': 'run RAG analysis on documentation',
        'generated': {
            'analytics_mode': 'rag',
            'rag_text': 'docs.pdf'
        },
        'corrected': {
            'analytics_mode': 'rag',
            'rag_text': '../docs/manual.pdf',
            'questions': 'What are the main features?',
            'rag_type': 'lc',
            'rag_train': 'True',
            'log_files_prefix': 'rag_analysis'
        },
        'success': True
    }
]

print(f"Processing {len(interactions)} user interactions...\n")

for i, interaction in enumerate(interactions, 1):
    print(f"Interaction {i}:")
    print(f"  Query: {interaction['query']}")
    
    # Process feedback
    result = rl_trainer.process_feedback(
        user_query=interaction['query'],
        generated_command=interaction['generated'],
        corrected_command=interaction['corrected'],
        execution_success=interaction['success']
    )
    
    print(f"  Reward: {result['feedback']['reward']:.3f}")
    print(f"  Examples in pool: {result['stats']['num_examples_in_pool']}")
    print()
    
    time.sleep(0.5)  # Simulate time between interactions

# ============================================================================
# PART 3: Analyze Training Progress
# ============================================================================
print("PART 3: Training Progress Analysis")
print("-" * 70)

stats = rl_trainer.get_training_stats()

print(f"Total Feedback Collected: {stats['total_feedback']}")
print(f"Average Reward: {stats['avg_reward']:.3f}")
print(f"Recent Average Reward: {stats['recent_avg_reward']:.3f}")
print(f"Examples in Pool: {stats['num_examples']}")
print(f"Total Selections: {stats['total_selections']}")
print()

# Show improvement
if stats['total_feedback'] > 0:
    improvement = stats['recent_avg_reward'] - stats['avg_reward']
    print(f"Improvement Trend: {improvement:+.3f}")
    if improvement > 0:
        print("✓ System is improving!")
    else:
        print("⚠ System needs more training")
print()

# ============================================================================
# PART 4: Examine Few-Shot Prompt
# ============================================================================
print("PART 4: Current Few-Shot Prompt")
print("-" * 70)

current_prompt = rl_trainer.get_current_prompt("run dataset analysis")
print(current_prompt)
print()

# ============================================================================
# PART 5: Example Pool Analysis
# ============================================================================
print("PART 5: Example Pool Analysis")
print("-" * 70)

pool = prompt_optimizer.example_pool
print(f"Total Examples: {len(pool)}")
print(f"Max Capacity: {prompt_optimizer.max_examples}")
print()

if pool:
    # Sort by success rate
    sorted_pool = sorted(pool, key=lambda x: x.success_rate, reverse=True)
    
    print("Top 3 Performing Examples:")
    for i, example in enumerate(sorted_pool[:3], 1):
        print(f"\n{i}. Success Rate: {example.success_rate:.3f}")
        print(f"   Usage Count: {example.usage_count}")
        print(f"   Query: {example.user_text[:60]}...")
        print(f"   Command: {json.dumps(example.smlp_command)[:80]}...")
print()

# ============================================================================
# PART 6: High-Quality Examples
# ============================================================================
print("PART 6: High-Quality Examples (Reward >= 0.8)")
print("-" * 70)

high_quality = feedback_collector.get_high_quality_examples(min_reward=0.8, n=10)

print(f"Found {len(high_quality)} high-quality examples\n")

for i, example in enumerate(high_quality[:3], 1):
    print(f"Example {i}:")
    print(f"  Query: {example.user_query}")
    print(f"  Reward: {example.reward:.3f}")
    print(f"  Corrected Command: {json.dumps(example.corrected_command, indent=2)}")
    print()

# ============================================================================
# PART 7: UCB Selection Demonstration
# ============================================================================
print("PART 7: UCB Example Selection")
print("-" * 70)

print("Selecting examples for next prompt using UCB algorithm...")
selected = prompt_optimizer.select_examples("run dataset analysis")

print(f"Selected {len(selected)} examples:\n")

for i, example in enumerate(selected, 1):
    print(f"{i}. Success Rate: {example.success_rate:.3f}, "
          f"Usage Count: {example.usage_count}")
    print(f"   Query: {example.user_text[:60]}...")
print()

# ============================================================================
# PART 8: Simulate More Learning
# ============================================================================
print("PART 8: Simulating Additional Learning")
print("-" * 70)

print("Adding 5 more interactions to demonstrate learning...\n")

additional_interactions = [
    {
        'query': 'analyze customer churn data',
        'generated': {
            'analytics_mode': 'dataset',
            'data_file': 'churn.csv'
        },
        'corrected': {
            'analytics_mode': 'dataset',
            'data_file': 'customer_churn.csv',
            'model_name': 'nn',
            'log_files_prefix': 'churn_analysis'
        },
        'success': True
    },
    {
        'query': 'verify safety properties of the controller',
        'generated': {
            'analytics_mode': 'verify',
            'spec_file': 'controller.py'
        },
        'corrected': {
            'analytics_mode': 'verify',
            'spec_file': 'controller_spec.py',
            'model_name': 'dummy',
            'log_files_prefix': 'safety_verification'
        },
        'success': True
    },
    {
        'query': 'run sentiment analysis on reviews',
        'generated': {
            'analytics_mode': 'dataset',
            'data_file': 'reviews.csv',
            'model_name': 'nn'
        },
        'corrected': {
            'analytics_mode': 'dataset',
            'data_file': 'product_reviews.csv',
            'model_name': 'nn',
            'log_files_prefix': 'sentiment_analysis'
        },
        'success': True
    },
    {
        'query': 'optimize neural network architecture',
        'generated': {
            'analytics_mode': 'optimize',
            'model_name': 'nn'
        },
        'corrected': {
            'analytics_mode': 'optimize',
            'model_name': 'nn',
            'optimization_iterations': '50',
            'log_files_prefix': 'nn_optimization'
        },
        'success': True
    },
    {
        'query': 'explain fraud detection model decisions',
        'generated': {
            'analytics_mode': 'explain'
        },
        'corrected': {
            'analytics_mode': 'explain',
            'model_file': 'fraud_model.pkl',
            'data_file': 'transactions.csv',
            'log_files_prefix': 'fraud_explanation'
        },
        'success': True
    }
]

rewards_before = []
rewards_after = []

for interaction in additional_interactions:
    result = rl_trainer.process_feedback(
        user_query=interaction['query'],
        generated_command=interaction['generated'],
        corrected_command=interaction['corrected'],
        execution_success=interaction['success']
    )
    rewards_after.append(result['feedback']['reward'])

# Get updated stats
new_stats = rl_trainer.get_training_stats()

print(f"Updated Statistics:")
print(f"  Total Feedback: {stats['total_feedback']} → {new_stats['total_feedback']}")
print(f"  Avg Reward: {stats['avg_reward']:.3f} → {new_stats['avg_reward']:.3f}")
print(f"  Recent Reward: {stats['recent_avg_reward']:.3f} → {new_stats['recent_avg_reward']:.3f}")
print(f"  Pool Size: {stats['num_examples']} → {new_stats['num_examples']}")
print()

# ============================================================================
# PART 9: Save Checkpoint
# ============================================================================
print("PART 9: Saving Checkpoint")
print("-" * 70)

rl_trainer.save_checkpoint("./demo_checkpoints")
print("✓ Checkpoint saved to ./demo_checkpoints/")
print()

# ============================================================================
# PART 10: Fine-Tuning Readiness Check
# ============================================================================
print("PART 10: Fine-Tuning Readiness Check")
print("-" * 70)

total_feedback = new_stats['total_feedback']
avg_reward = new_stats['avg_reward']
high_quality_count = len(feedback_collector.get_high_quality_examples(0.8))

print(f"Total Feedback: {total_feedback} (threshold: 100)")
print(f"Average Reward: {avg_reward:.3f} (threshold: 0.7)")
print(f"High-Quality Examples: {high_quality_count} (minimum: 50)")
print()

if total_feedback >= 100 and avg_reward >= 0.7 and high_quality_count >= 50:
    print("✓ Ready for fine-tuning!")
    print("\nTo fine-tune the model, run:")
    print("  from smlp_agent_rl_finetune import fine_tune_from_feedback")
    print("  result = fine_tune_from_feedback(feedback_collector)")
else:
    print("⚠ Not ready for fine-tuning yet")
    print(f"  Need {max(0, 100 - total_feedback)} more feedback instances")
    print(f"  Need {max(0, 50 - high_quality_count)} more high-quality examples")

print()

# ============================================================================
# PART 11: Demonstration Summary
# ============================================================================
print("="*70)
print("DEMONSTRATION SUMMARY")
print("="*70)

print("""
This demo showed the complete RL enhancement workflow:

1. ✓ Initialized RL components (FeedbackCollector, PromptOptimizer, RLTrainer)
2. ✓ Collected user feedback on generated commands
3. ✓ Computed rewards based on corrections
4. ✓ Updated example pool with high-quality examples
5. ✓ Selected examples using UCB algorithm
6. ✓ Generated optimized few-shot prompts
7. ✓ Tracked training progress and statistics
8. ✓ Saved checkpoints for persistence
9. ✓ Evaluated fine-tuning readiness

The system demonstrated continuous learning through:
- Dynamic example selection
- Reward-based optimization
- Progressive improvement of prompts

Next Steps:
1. Integrate with your SMLP Agent
2. Deploy with web UI for user feedback
3. Monitor training statistics
4. Fine-tune model when ready

For integration, see: INTEGRATION_GUIDE.md
For full documentation, see: SMLP_RL_README.md
""")

print("="*70)
print("Demo complete! Files created:")
print("  - demo_feedback.jsonl (feedback storage)")
print("  - demo_checkpoints/ (RL checkpoints)")
print("="*70)
