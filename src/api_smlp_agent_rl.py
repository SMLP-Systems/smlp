# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
FastAPI endpoints for RL-enhanced SMLP Agent

Adds endpoints for:
1. Collecting user feedback/corrections
2. Viewing training statistics
3. Managing RL checkpoints
4. Viewing high-quality examples
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import json

# These would be imported from your existing API file
# from api_smlp_agent import agent  # Your existing agent

from smlp_agent_rl_integration import (
    RLEnhancedSmlpAgent,
    compare_commands,
    suggest_corrections
)


# Request/Response models
class FeedbackRequest(BaseModel):
    """Request model for submitting feedback"""
    user_query: str
    generated_command: Dict
    corrected_command: Dict
    execution_success: bool = False
    metadata: Optional[Dict] = None


class TextCommandRequest(BaseModel):
    """Request model for text command with optional feedback"""
    query: str
    user_correction: Optional[Dict] = None
    execution_success: Optional[bool] = None


class TrainingStatsResponse(BaseModel):
    """Response model for training statistics"""
    total_feedback: int
    avg_reward: float
    recent_avg_reward: float
    num_examples: int
    total_selections: int


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    reward: float
    should_fine_tune: bool
    stats: Dict
    message: str


class CommandComparisonResponse(BaseModel):
    """Response model for command comparison"""
    matching: Dict
    added: Dict
    removed: Dict
    modified: Dict
    is_perfect: bool
    accuracy: float


# RL-enhanced API endpoints
def setup_rl_endpoints(app: FastAPI, rl_agent: RLEnhancedSmlpAgent):
    """
    Add RL-related endpoints to existing FastAPI app
    
    Args:
        app: FastAPI application instance
        rl_agent: RLEnhancedSmlpAgent instance
    """
    
    @app.post("/agent/text_with_feedback")
    async def run_text_command_with_feedback(request: TextCommandRequest):
        """
        Run text command with RL optimization and optional feedback collection
        
        This endpoint:
        1. Uses RL-optimized prompt
        2. Generates SMLP command
        3. Executes command
        4. Optionally collects feedback if user_correction is provided
        """
        try:
            result = rl_agent.run_text_command_with_feedback(
                user_input=request.query,
                user_correction=request.user_correction,
                execution_success=request.execution_success
            )
            
            response = {
                "status": "success",
                "result": result['result'],
                "generated_command": result['generated_command']
            }
            
            # Add feedback stats if correction was provided
            if 'feedback_stats' in result:
                response['feedback'] = {
                    'reward': result['reward'],
                    'stats': result['feedback_stats'],
                    'should_fine_tune': result['should_fine_tune']
                }
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/agent/feedback")
    async def submit_feedback(request: FeedbackRequest):
        """
        Submit feedback for a previously generated command
        
        This allows users to provide corrections asynchronously
        (e.g., after reviewing the execution results)
        """
        try:
            feedback_result = rl_agent.provide_feedback(
                user_query=request.user_query,
                generated_command=request.generated_command,
                corrected_command=request.corrected_command,
                execution_success=request.execution_success
            )
            
            return FeedbackResponse(
                reward=feedback_result['reward'],
                should_fine_tune=feedback_result['should_fine_tune'],
                stats=feedback_result['stats'],
                message="Feedback collected successfully"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/agent/training_stats")
    async def get_training_stats():
        """
        Get current RL training statistics
        """
        try:
            stats = rl_agent.get_training_stats()
            return TrainingStatsResponse(**stats)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/agent/current_prompt")
    async def get_current_prompt():
        """
        Get the current RL-optimized few-shot prompt
        """
        try:
            prompt = rl_agent.get_current_prompt()
            return {
                "prompt": prompt,
                "num_examples": len(rl_agent.prompt_optimizer.example_pool),
                "examples_in_prompt": rl_agent.prompt_optimizer.examples_per_prompt
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/agent/feedback_history")
    async def get_feedback_history(n: int = 100):
        """
        Get recent feedback history
        
        Args:
            n: Number of recent entries to return (default: 100)
        """
        try:
            history = rl_agent.get_feedback_history(n)
            return {
                "count": len(history),
                "feedback": history
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/agent/high_quality_examples")
    async def get_high_quality_examples(min_reward: float = 0.8, n: int = 50):
        """
        Get high-quality training examples
        
        Args:
            min_reward: Minimum reward threshold (default: 0.8)
            n: Maximum number of examples (default: 50)
        """
        try:
            examples = rl_agent.get_high_quality_examples(min_reward, n)
            return {
                "count": len(examples),
                "examples": examples,
                "min_reward": min_reward
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/agent/compare_commands")
    async def compare_commands_endpoint(
        generated: Dict,
        corrected: Dict
    ):
        """
        Compare generated and corrected commands
        
        Returns detailed analysis of differences
        """
        try:
            comparison = compare_commands(generated, corrected)
            return CommandComparisonResponse(**comparison)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/agent/save_checkpoint")
    async def save_checkpoint():
        """
        Manually trigger checkpoint save
        """
        try:
            rl_agent.save_checkpoint()
            return {
                "status": "success",
                "message": "RL checkpoint saved successfully"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/agent/example_pool")
    async def get_example_pool():
        """
        Get information about the current example pool
        """
        try:
            pool = rl_agent.prompt_optimizer.example_pool
            
            pool_info = {
                "total_examples": len(pool),
                "max_capacity": rl_agent.prompt_optimizer.max_examples,
                "examples": [
                    {
                        "user_text": ex.user_text,
                        "smlp_command": ex.smlp_command,
                        "success_rate": ex.success_rate,
                        "usage_count": ex.usage_count,
                        "last_used": ex.last_used
                    }
                    for ex in sorted(pool, key=lambda x: x.success_rate, reverse=True)[:20]
                ]
            }
            
            return pool_info
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Example usage
if __name__ == "__main__":
    print("=== SMLP Agent RL API Endpoints ===\n")
    
    print("Available endpoints:")
    print("  POST /agent/text_with_feedback")
    print("    - Run command with RL optimization + optional feedback")
    print()
    print("  POST /agent/feedback")
    print("    - Submit feedback for a previous command")
    print()
    print("  GET /agent/training_stats")
    print("    - Get RL training statistics")
    print()
    print("  GET /agent/current_prompt")
    print("    - Get current RL-optimized prompt")
    print()
    print("  GET /agent/feedback_history?n=100")
    print("    - Get recent feedback history")
    print()
    print("  GET /agent/high_quality_examples?min_reward=0.8&n=50")
    print("    - Get high-quality training examples")
    print()
    print("  POST /agent/compare_commands")
    print("    - Compare generated vs corrected commands")
    print()
    print("  POST /agent/save_checkpoint")
    print("    - Manually save RL checkpoint")
    print()
    print("  GET /agent/example_pool")
    print("    - View example pool statistics")
    
    print("\n" + "="*60)
    print("Example curl commands:")
    print("="*60)
    print("""
# Run command with feedback
curl -X POST http://127.0.0.1:8000/agent/text_with_feedback \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "run dataset analysis on sales data",
    "user_correction": {
      "analytics_mode": "dataset",
      "data_file": "sales_data.csv",
      "model_name": "nn",
      "log_files_prefix": "sales_analysis"
    },
    "execution_success": true
  }'

# Submit feedback
curl -X POST http://127.0.0.1:8000/agent/feedback \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_query": "run verification",
    "generated_command": {"analytics_mode": "verify"},
    "corrected_command": {"analytics_mode": "verify", "log_files_prefix": "verify_run"},
    "execution_success": true
  }'

# Get training stats
curl http://127.0.0.1:8000/agent/training_stats

# Get current prompt
curl http://127.0.0.1:8000/agent/current_prompt

# Get feedback history
curl http://127.0.0.1:8000/agent/feedback_history?n=50

# Save checkpoint
curl -X POST http://127.0.0.1:8000/agent/save_checkpoint
    """)
