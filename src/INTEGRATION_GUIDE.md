# Integration Guide: Adding RL to Your Existing SMLP Agent

This guide shows exactly how to integrate the RL enhancement into your existing SMLP Agent codebase.

## Step 1: Update Your API File (api_smlp_agent.py)

Add these imports at the top:

```python
from smlp_agent_rl_integration import enhance_agent_with_rl, RLEnhancedSmlpAgent
from api_smlp_agent_rl import setup_rl_endpoints
```

Modify your agent initialization:

```python
# OLD CODE:
# agent = SmlpAgent()

# NEW CODE:
base_agent = SmlpAgent()
agent = enhance_agent_with_rl(base_agent, load_checkpoint=True)
```

Add RL endpoints to your FastAPI app:

```python
# After creating your FastAPI app
app = FastAPI()

# ... your existing endpoints ...

# Add RL endpoints
setup_rl_endpoints(app, agent)
```

## Step 2: Update Your SmlpAgent Class (smlp_agent.py)

### Option A: Minimal Changes (Recommended)

Keep your existing `SmlpAgent` class unchanged. The RL enhancement wraps around it.

No modifications needed! The `RLEnhancedSmlpAgent` wrapper handles everything.

### Option B: Direct Integration

If you prefer to integrate RL directly into `SmlpAgent`, add these methods:

```python
class SmlpAgent:
    def __init__(self, llm_interpreter=None, spec_generator=None, executor=None, doc_qa=None):
        # ... existing init code ...
        
        # Add RL components (optional)
        self._init_rl_components()
    
    def _init_rl_components(self):
        """Initialize RL components"""
        try:
            from smlp_agent_rl import FeedbackCollector, PromptOptimizer, RLTrainer
            
            self.rl_feedback_collector = FeedbackCollector()
            self.rl_prompt_optimizer = PromptOptimizer()
            self.rl_trainer = RLTrainer(
                self.rl_feedback_collector,
                self.rl_prompt_optimizer
            )
            
            # Try to load checkpoint
            self.rl_trainer.load_checkpoint()
            
            self.rl_enabled = True
            print("RL components initialized successfully")
        except Exception as e:
            print(f"RL components not available: {e}")
            self.rl_enabled = False
    
    def run_text_command(self, user_input: str):
        """Enhanced version with optional RL optimization"""
        
        # Use RL-optimized prompt if available
        if hasattr(self, 'rl_enabled') and self.rl_enabled:
            optimized_prompt = self.rl_trainer.get_current_prompt(user_input)
            self.llm.load_prompt(optimized_prompt)
        
        # ... rest of existing code ...
        
        return result
    
    def collect_feedback(self, 
                        user_query: str,
                        generated_command: dict,
                        corrected_command: dict,
                        execution_success: bool = False):
        """Collect user feedback for RL training"""
        
        if not hasattr(self, 'rl_enabled') or not self.rl_enabled:
            print("RL not enabled, feedback not collected")
            return None
        
        feedback_result = self.rl_trainer.process_feedback(
            user_query=user_query,
            generated_command=generated_command,
            corrected_command=corrected_command,
            execution_success=execution_success
        )
        
        return feedback_result
```

## Step 3: Update API Endpoints

### Modify existing `/agent/text` endpoint:

```python
# OLD CODE:
@app.post("/agent/text")
async def run_text(request: dict):
    query = request.get("query", "")
    result = agent.run_text_command(query)
    return {"result": result}

# NEW CODE (if using wrapper):
@app.post("/agent/text")
async def run_text(request: dict):
    query = request.get("query", "")
    result = agent.run_text_command(query)  # Uses RL-optimized prompt automatically
    return {"result": result}

# ALTERNATIVE (if you want feedback in same call):
@app.post("/agent/text")
async def run_text(request: dict):
    query = request.get("query", "")
    correction = request.get("correction", None)
    execution_success = request.get("execution_success", False)
    
    result = agent.run_text_command_with_feedback(
        user_input=query,
        user_correction=correction,
        execution_success=execution_success
    )
    
    return result
```

## Step 4: File Organization

Add these new files to your project:

```
smlp/
├── smlp_agent.py                      # Your existing file (minimal/no changes)
├── smlp_agent_rl.py                   # NEW: Core RL components
├── smlp_agent_rl_integration.py       # NEW: Integration wrapper
├── smlp_agent_rl_finetune.py         # NEW: Fine-tuning utilities
├── api_smlp_agent.py                  # Modified: Add RL endpoint setup
├── api_smlp_agent_rl.py              # NEW: RL-specific endpoints
├── smlp_agent_web_ui.html            # NEW: Web interface
└── SMLP_RL_README.md                 # NEW: Documentation
```

## Step 5: Testing the Integration

### Test 1: Basic Command Generation

```python
# test_rl_integration.py
from smlp_agent import SmlpAgent
from smlp_agent_rl_integration import enhance_agent_with_rl

# Create and enhance agent
base_agent = SmlpAgent()
rl_agent = enhance_agent_with_rl(base_agent)

# Test query
result = rl_agent.run_text_command("run dataset analysis on sales data")
print("Generated command:", result)
```

### Test 2: Feedback Collection

```python
# Simulate user correction
generated = {"analytics_mode": "dataset"}
corrected = {
    "analytics_mode": "dataset",
    "data_file": "sales.csv",
    "log_files_prefix": "sales_analysis"
}

feedback = rl_agent.provide_feedback(
    user_query="run dataset analysis on sales data",
    generated_command=generated,
    corrected_command=corrected,
    execution_success=True
)

print(f"Reward: {feedback['reward']:.3f}")
```

### Test 3: API Endpoints

```bash
# Start your server
python api_smlp_agent.py

# Test text command
curl -X POST http://localhost:8000/agent/text \
  -H "Content-Type: application/json" \
  -d '{"query": "run dataset analysis"}'

# Test with feedback
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

# Check stats
curl http://localhost:8000/agent/training_stats
```

## Step 6: Migration Path (Gradual Rollout)

### Phase 1: Parallel Testing (Week 1-2)
- Keep existing agent running
- Add RL agent as separate endpoint
- Collect feedback without affecting production

```python
# Keep both versions
agent_v1 = SmlpAgent()  # Original
agent_v2 = enhance_agent_with_rl(SmlpAgent())  # RL-enhanced

@app.post("/agent/text")  # Original endpoint
async def run_text(request: dict):
    return agent_v1.run_text_command(request["query"])

@app.post("/agent/text/rl")  # New RL endpoint
async def run_text_rl(request: dict):
    return agent_v2.run_text_command(request["query"])
```

### Phase 2: A/B Testing (Week 3-4)
- Route 50% of traffic to RL agent
- Compare performance metrics
- Collect user feedback

```python
import random

@app.post("/agent/text")
async def run_text(request: dict):
    if random.random() < 0.5:
        result = agent_v2.run_text_command(request["query"])
        result['version'] = 'rl'
    else:
        result = agent_v1.run_text_command(request["query"])
        result['version'] = 'original'
    return result
```

### Phase 3: Full Rollout (Week 5+)
- Replace original with RL-enhanced version
- Monitor performance
- Continue collecting feedback

## Step 7: Monitoring and Maintenance

### Daily Tasks
```bash
# Check training stats
curl http://localhost:8000/agent/training_stats

# View recent feedback
curl http://localhost:8000/agent/feedback_history?n=10
```

### Weekly Tasks
```python
# Generate training report
stats = rl_agent.get_training_stats()
print(f"This week's stats:")
print(f"  Total feedback: {stats['total_feedback']}")
print(f"  Average reward: {stats['avg_reward']:.3f}")
print(f"  Improvement: {stats['recent_avg_reward'] - stats['avg_reward']:.3f}")

# Save checkpoint
rl_agent.save_checkpoint()
```

### Monthly Tasks
```python
# Check if fine-tuning is needed
if stats['total_feedback'] >= 100 and stats['avg_reward'] > 0.8:
    from smlp_agent_rl_finetune import fine_tune_from_feedback
    
    result = fine_tune_from_feedback(
        feedback_collector=rl_agent.feedback_collector,
        min_reward=0.8
    )
    print(f"Fine-tuned model: {result['model_name']}")
```

## Common Issues and Solutions

### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'smlp_agent_rl'
```

**Solution**: Ensure all new files are in the same directory as `smlp_agent.py`, or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/smlp"
```

### Issue 2: Checkpoint Not Loading
```
Checkpoint directory ./smlp_rl_checkpoints not found
```

**Solution**: Checkpoints are created after first feedback. To initialize:
```python
rl_agent.save_checkpoint()  # Creates initial checkpoint
```

### Issue 3: Low Reward Scores
```
Average reward stuck at 0.3-0.4
```

**Solution**: 
1. Check if corrections are complete (all parameters)
2. Verify execution_success flag is accurate
3. Adjust reward weights:
```python
rl_agent.feedback_collector.reward_model.edit_weight = 0.6
rl_agent.feedback_collector.reward_model.success_weight = 0.4
```

### Issue 4: Memory Usage
```
Process using too much memory
```

**Solution**: Reduce buffer sizes:
```python
rl_agent.feedback_collector.feedback_buffer = deque(maxlen=500)  # Was 1000
rl_agent.prompt_optimizer.max_examples = 50  # Was 100
```

## Example: Complete Modified api_smlp_agent.py

```python
# api_smlp_agent.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from smlp_agent import SmlpAgent
from smlp_agent_rl_integration import enhance_agent_with_rl
from api_smlp_agent_rl import setup_rl_endpoints

app = FastAPI()

# Initialize agent with RL enhancement
base_agent = SmlpAgent()
agent = enhance_agent_with_rl(base_agent, load_checkpoint=True)

# Add RL endpoints
setup_rl_endpoints(app, agent)

# Existing endpoints (now RL-enhanced automatically)
class TextRequest(BaseModel):
    query: str

@app.post("/agent/text")
async def run_text(request: TextRequest):
    """Original endpoint, now with RL optimization"""
    try:
        result = agent.run_text_command(request.query)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ... rest of your existing endpoints ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Next Steps

1. **Week 1**: Integrate RL components, test locally
2. **Week 2**: Deploy with parallel endpoints, collect initial feedback
3. **Week 3**: Start A/B testing
4. **Week 4**: Analyze results, adjust parameters
5. **Month 2**: Full rollout, schedule first fine-tuning

## Support

If you encounter issues:

1. Check the logs: `smlp_agent_log.jsonl` and `smlp_rl_feedback.jsonl`
2. Verify Ollama is running: `curl http://localhost:11434`
3. Test API endpoints individually
4. Review training stats for anomalies

## Summary

The RL enhancement integrates seamlessly with minimal changes to your existing code:

✅ **Minimal changes to existing code**
✅ **Backward compatible**
✅ **Gradual rollout possible**
✅ **Easy to monitor and debug**
✅ **Automatic improvement over time**

Key files to modify:
- `api_smlp_agent.py` - Add RL endpoint setup (3 lines)
- Optional: `smlp_agent.py` - Add RL initialization (if direct integration preferred)

Everything else is handled by the new RL modules!
