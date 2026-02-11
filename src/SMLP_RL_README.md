# SMLP Agent Reinforcement Learning Enhancement

## Overview

This RL enhancement improves the SMLP Agent's text-to-command conversion through user feedback and dynamic few-shot prompt optimization. The system learns from user corrections to generate better SMLP commands over time.

## Architecture

```
User Query → RL-Optimized Prompt → LLM → SMLP Command → Execution
                ↑                                           ↓
                |                                           |
                └───────── User Correction ─────────────────┘
                                  ↓
                          Feedback Collector
                                  ↓
                          Reward Computation
                                  ↓
                          Prompt Optimizer (UCB)
                                  ↓
                    Updated Few-Shot Examples
```

## Key Components

### 1. FeedbackCollector
- **Purpose**: Collects and stores user corrections
- **Storage**: JSONL file (`smlp_rl_feedback.jsonl`)
- **Reward Computation**: 
  - Edit distance (50% weight)
  - Execution success (30% weight)
  - Semantic similarity (20% weight)

### 2. PromptOptimizer
- **Purpose**: Optimizes few-shot example selection
- **Algorithm**: Upper Confidence Bound (UCB) multi-armed bandit
- **Pool Size**: Up to 100 examples
- **Selection**: 5 examples per prompt
- **Strategy**: Balance exploitation (high success rate) vs exploration (untested examples)

### 3. RewardModel
- **Purpose**: Evaluates quality of generated commands
- **Metrics**:
  - Parameter-level accuracy
  - Execution success
  - Edit distance from correct command
- **Output**: Reward score [0, 1]

### 4. RLTrainer
- **Purpose**: Orchestrates the RL training process
- **Workflow**:
  1. Collect feedback
  2. Update example scores
  3. Add high-quality examples to pool
  4. Generate new prompts
  5. Trigger fine-tuning (every 100 examples)

### 5. OllamaFineTuner
- **Purpose**: Fine-tunes local Ollama models
- **Methods**:
  - Custom system prompt creation
  - Training data export
  - Model creation via Ollama CLI
  - Script generation for full fine-tuning

## Installation

```bash
# Install required packages
pip install numpy requests openai python-dotenv

# Ensure Ollama is running
ollama serve  # In separate terminal
```

## Quick Start

### 1. Basic Integration

```python
from smlp_agent import SmlpAgent
from smlp_agent_rl_integration import enhance_agent_with_rl

# Create base agent
base_agent = SmlpAgent()

# Enhance with RL
rl_agent = enhance_agent_with_rl(base_agent, load_checkpoint=True)

# Run command with RL optimization
result = rl_agent.run_text_command("run dataset analysis on sales data")
```

### 2. With Feedback Collection

```python
# User provides query
query = "run dataset analysis on sales data"

# Generate command
result = rl_agent.run_text_command(query)
generated = result['generated_command']

# User reviews and corrects
user_correction = {
    'analytics_mode': 'dataset',
    'data_file': 'sales_data.csv',
    'model_name': 'nn',
    'log_files_prefix': 'sales_analysis'
}

# Submit feedback
feedback = rl_agent.provide_feedback(
    user_query=query,
    generated_command=generated,
    corrected_command=user_correction,
    execution_success=True
)

print(f"Reward: {feedback['reward']:.3f}")
```

### 3. Using the API

```bash
# Start the server
python api_smlp_agent.py  # Your existing API

# Run command with feedback
curl -X POST http://localhost:8000/agent/text_with_feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "run dataset analysis",
    "user_correction": {
      "analytics_mode": "dataset",
      "log_files_prefix": "analysis"
    },
    "execution_success": true
  }'

# Get training statistics
curl http://localhost:8000/agent/training_stats

# View current prompt
curl http://localhost:8000/agent/current_prompt
```

## How It Works

### The RL Learning Loop

1. **Initial State**: System uses static few-shot prompt
2. **User Interaction**: 
   - User provides query
   - LLM generates command using current prompt
   - Command is executed
3. **Feedback Collection**:
   - User reviews command
   - User provides corrections (if needed)
   - System computes reward
4. **Learning**:
   - Update success rates of examples used in prompt
   - Add high-quality corrections to example pool
   - Prune low-performing examples
5. **Optimization**:
   - Select better examples for next prompt using UCB
   - Periodically fine-tune model

### UCB (Upper Confidence Bound) Algorithm

The system uses UCB to select few-shot examples:

```
UCB Score = Success Rate + C * sqrt(ln(Total Selections) / Example Usage Count)
           └─ Exploitation ─┘   └─────────── Exploration ──────────────────┘
```

- **Exploitation**: Prefer examples with high success rates
- **Exploration**: Try examples that haven't been used much
- **Balance**: C (exploration factor) = 0.5

### Reward Computation

```
Total Reward = 0.5 * Edit Reward + 0.3 * Success Reward + 0.2 * Semantic Reward

Edit Reward = # Matching Parameters / Total Parameters
Success Reward = 1.0 if execution succeeded, else 0.0
Semantic Reward = (optional) embedding similarity
```

## Fine-Tuning

### Trigger Conditions

Fine-tuning is suggested when:
- Every 100 feedback instances collected
- Average reward drops below threshold
- Manual trigger via API

### Fine-Tuning Process

```python
from smlp_agent_rl_finetune import fine_tune_from_feedback

# Fine-tune model
result = fine_tune_from_feedback(
    feedback_collector=rl_agent.feedback_collector,
    base_model="mistral",
    min_reward=0.8,
    model_name="smlp_agent_v1"
)

# Use fine-tuned model
rl_agent.base_agent.llm.model_name = "smlp_agent_v1"
```

### Full Fine-Tuning (with transformers)

```bash
# Generate fine-tuning script
python smlp_agent_rl_finetune.py

# Run the generated script
pip install transformers datasets peft accelerate
python fine_tune_smlp_agent_[timestamp].py
```

## API Reference

### Endpoints

#### POST /agent/text_with_feedback
Run command with RL optimization and optional feedback.

**Request:**
```json
{
  "query": "run dataset analysis",
  "user_correction": {
    "analytics_mode": "dataset",
    "log_files_prefix": "analysis"
  },
  "execution_success": true
}
```

**Response:**
```json
{
  "status": "success",
  "result": "...",
  "generated_command": {...},
  "feedback": {
    "reward": 0.85,
    "stats": {...},
    "should_fine_tune": false
  }
}
```

#### POST /agent/feedback
Submit feedback for a previous command.

#### GET /agent/training_stats
Get current training statistics.

**Response:**
```json
{
  "total_feedback": 150,
  "avg_reward": 0.73,
  "recent_avg_reward": 0.81,
  "num_examples": 45,
  "total_selections": 320
}
```

#### GET /agent/current_prompt
Get the current RL-optimized prompt.

#### GET /agent/feedback_history?n=100
Get recent feedback history.

#### GET /agent/high_quality_examples?min_reward=0.8&n=50
Get high-quality training examples.

#### POST /agent/save_checkpoint
Save current RL state.

#### GET /agent/example_pool
View example pool statistics.

## Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=your_key_here  # If using OpenAI
```

### RL Parameters

```python
# In code
rl_agent = RLEnhancedSmlpAgent(
    base_agent,
    load_checkpoint=True
)

# Configure prompt optimizer
rl_agent.prompt_optimizer.max_examples = 100
rl_agent.prompt_optimizer.examples_per_prompt = 5
rl_agent.prompt_optimizer.exploration_factor = 0.5

# Configure reward model
rl_agent.feedback_collector.reward_model.edit_weight = 0.5
rl_agent.feedback_collector.reward_model.success_weight = 0.3
rl_agent.feedback_collector.reward_model.semantic_weight = 0.2

# Configure training
rl_agent.rl_trainer.fine_tune_threshold = 100
```

## File Structure

```
smlp/
├── smlp_agent.py                      # Original agent (your code)
├── smlp_agent_rl.py                   # Core RL components
├── smlp_agent_rl_integration.py       # Integration wrapper
├── smlp_agent_rl_finetune.py         # Fine-tuning module
├── api_smlp_agent_rl.py              # API endpoints
├── smlp_rl_feedback.jsonl            # Feedback storage
├── smlp_rl_example_pool.pkl          # Example pool
└── smlp_rl_checkpoints/              # Training checkpoints
    ├── example_pool_20260211_143022.pkl
    └── training_history_20260211_143022.json
```

## Best Practices

### 1. Feedback Quality
- Provide complete corrections (all parameters)
- Mark execution success accurately
- Add metadata for context when useful

### 2. Training
- Start with at least 50 feedback examples
- Aim for 80%+ reward threshold for fine-tuning
- Save checkpoints regularly
- Monitor training statistics

### 3. Prompt Optimization
- Let the system explore initially (first 100 examples)
- Adjust exploration factor if needed
- Prune underperforming examples periodically

### 4. Fine-Tuning
- Wait for 100+ high-quality examples
- Test fine-tuned model before deploying
- Keep base model as fallback

## Monitoring

### Training Progress

```python
# Get statistics
stats = rl_agent.get_training_stats()

print(f"Total feedback: {stats['total_feedback']}")
print(f"Average reward: {stats['avg_reward']:.3f}")
print(f"Recent reward: {stats['recent_avg_reward']:.3f}")
print(f"Improvement: {stats['recent_avg_reward'] - stats['avg_reward']:.3f}")
```

### Example Pool Health

```python
# View pool statistics
pool = rl_agent.prompt_optimizer.example_pool

print(f"Pool size: {len(pool)}/{rl_agent.prompt_optimizer.max_examples}")
print(f"Avg success rate: {np.mean([ex.success_rate for ex in pool]):.3f}")

# Top performing examples
top = sorted(pool, key=lambda x: x.success_rate, reverse=True)[:5]
for ex in top:
    print(f"  {ex.success_rate:.3f}: {ex.user_text[:50]}...")
```

## Troubleshooting

### Issue: Low reward scores
**Solution**: 
- Check if corrections are complete
- Verify parameter names match SMLP CLI
- Review reward weights

### Issue: Prompt not improving
**Solution**:
- Increase exploration factor
- Check example pool diversity
- Collect more feedback

### Issue: Fine-tuning fails
**Solution**:
- Verify Ollama is running
- Check training data quality
- Use external fine-tuning script

### Issue: Memory usage high
**Solution**:
- Reduce feedback buffer size
- Prune example pool more aggressively
- Clear old checkpoints

## Advanced Usage

### Custom Reward Function

```python
from smlp_agent_rl import RewardModel

class CustomRewardModel(RewardModel):
    def compute_reward(self, generated, corrected, execution_success, semantic_sim=0.0):
        # Custom logic here
        base_reward = super().compute_reward(
            generated, corrected, execution_success, semantic_sim
        )
        
        # Add custom penalties/bonuses
        if 'log_files_prefix' not in generated:
            base_reward *= 0.9  # Penalty for missing log prefix
        
        return base_reward

# Use custom reward model
rl_agent.feedback_collector.reward_model = CustomRewardModel()
```

### Semantic Similarity (Optional)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(query1, query2):
    emb1 = model.encode([query1])
    emb2 = model.encode([query2])
    return cosine_similarity(emb1, emb2)[0][0]

# Add to feedback collection
feedback_entry = rl_agent.provide_feedback(
    user_query=query,
    generated_command=generated,
    corrected_command=corrected,
    execution_success=True
)
```

## Performance Metrics

Expected improvements after collecting 100+ feedback instances:

- **Reward Score**: 0.6 → 0.85+
- **Parameter Accuracy**: 70% → 90%+
- **Execution Success**: 60% → 85%+
- **User Corrections**: 50% of commands → 15% of commands

## Future Enhancements

1. **Advanced RL Algorithms**
   - PPO (Proximal Policy Optimization)
   - DQN (Deep Q-Network)
   - A3C (Asynchronous Advantage Actor-Critic)

2. **Embedding-Based Retrieval**
   - Semantic similarity for example selection
   - Query clustering
   - Context-aware prompting

3. **Active Learning**
   - Identify uncertain cases
   - Request targeted feedback
   - Adaptive sampling

4. **Multi-Model Ensemble**
   - Combine multiple fine-tuned models
   - Voting mechanisms
   - Confidence-based selection

## Contributing

When extending the RL system:

1. Maintain backward compatibility
2. Add tests for new components
3. Document configuration options
4. Update this README

## References

- Multi-Armed Bandits: UCB algorithm
- Reinforcement Learning: Reward shaping
- Few-Shot Learning: Dynamic example selection
- Fine-Tuning: LoRA and parameter-efficient methods

## License

SPDX-License-Identifier: Apache-2.0
