# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
Integration module for RL-enhanced SMLP Agent

This module provides wrapper functions and modifications to integrate
the RL enhancement with the existing SmlpAgent class.
"""

import json
from typing import Dict, Optional
from smlp_agent_rl import (
    FeedbackCollector,
    PromptOptimizer,
    RLTrainer,
    extract_feedback_from_correction
)


class RLEnhancedSmlpAgent:
    """
    Wrapper class that adds RL capabilities to the existing SmlpAgent
    
    Usage:
        # Create base agent
        base_agent = SmlpAgent(...)
        
        # Enhance with RL
        rl_agent = RLEnhancedSmlpAgent(base_agent)
        
        # Use as normal, with feedback collection
        result = rl_agent.run_text_command_with_feedback(
            user_input="run dataset analysis",
            user_correction={"analytics_mode": "dataset", ...}
        )
    """
    
    def __init__(self, base_agent, load_checkpoint: bool = True):
        """
        Initialize RL-enhanced agent
        
        Args:
            base_agent: Existing SmlpAgent instance
            load_checkpoint: Whether to load previous RL training state
        """
        self.base_agent = base_agent
        
        # Initialize RL components
        self.feedback_collector = FeedbackCollector()
        self.prompt_optimizer = PromptOptimizer(
            max_examples=100,
            examples_per_prompt=5,
            exploration_factor=0.5
        )
        self.rl_trainer = RLTrainer(
            self.feedback_collector,
            self.prompt_optimizer,
            fine_tune_threshold=100
        )
        
        # Load checkpoint if requested
        if load_checkpoint:
            try:
                self.rl_trainer.load_checkpoint()
                print("RL checkpoint loaded successfully")
            except Exception as e:
                print(f"Could not load RL checkpoint: {e}")
        
        # Track current examples being used
        self.current_selected_examples = []
    
    def run_text_command_with_feedback(self,
                                      user_input: str,
                                      user_correction: Optional[Dict] = None,
                                      execution_success: Optional[bool] = None) -> Dict:
        """
        Run text command with RL-enhanced prompt and optional feedback collection
        
        Args:
            user_input: Natural language query
            user_correction: Optional user-corrected command dict
            execution_success: Optional execution success flag
            
        Returns:
            Dictionary containing:
                - result: SMLP execution result
                - generated_command: LLM-generated command
                - feedback_stats: RL feedback statistics (if correction provided)
        """
        # Get RL-optimized prompt
        optimized_prompt = self.rl_trainer.get_current_prompt(user_input)
        
        # Load optimized prompt into LLM
        self.base_agent.llm.load_prompt(optimized_prompt)
        
        # Run the command through base agent
        print("Running SMLP with RL-optimized prompt...")
        result = self.base_agent.run_text_command(user_input)
        
        # Get the generated command (need to intercept this from llm)
        llm_params_dict = self.base_agent.llm.plan_from_text(user_input)
        
        response = {
            'result': result,
            'generated_command': llm_params_dict
        }
        
        # If user provides correction, collect feedback
        if user_correction is not None:
            feedback_result = self.rl_trainer.process_feedback(
                user_query=user_input,
                generated_command=llm_params_dict,
                corrected_command=user_correction,
                execution_success=execution_success if execution_success is not None else False
            )
            
            response['feedback_stats'] = feedback_result['stats']
            response['reward'] = feedback_result['feedback']['reward']
            response['should_fine_tune'] = feedback_result['should_fine_tune']
            
            # Auto-save checkpoint periodically
            if len(self.feedback_collector.feedback_buffer) % 50 == 0:
                self.rl_trainer.save_checkpoint()
                print("RL checkpoint saved")
        
        return response
    
    def run_text_command(self, user_input: str):
        """
        Run text command with RL-optimized prompt (no feedback)
        
        Maintains compatibility with existing API
        """
        # Get RL-optimized prompt
        optimized_prompt = self.rl_trainer.get_current_prompt(user_input)
        
        # Load optimized prompt into LLM
        self.base_agent.llm.load_prompt(optimized_prompt)
        
        # Run the command through base agent
        return self.base_agent.run_text_command(user_input)
    
    def provide_feedback(self,
                        user_query: str,
                        generated_command: Dict,
                        corrected_command: Dict,
                        execution_success: bool = False) -> Dict:
        """
        Provide feedback on a previously generated command
        
        This allows asynchronous feedback collection (e.g., after user reviews output)
        
        Returns:
            Dictionary with feedback statistics
        """
        feedback_result = self.rl_trainer.process_feedback(
            user_query=user_query,
            generated_command=generated_command,
            corrected_command=corrected_command,
            execution_success=execution_success
        )
        
        return {
            'reward': feedback_result['feedback']['reward'],
            'stats': feedback_result['stats'],
            'should_fine_tune': feedback_result['should_fine_tune']
        }
    
    def get_training_stats(self) -> Dict:
        """Get RL training statistics"""
        return self.rl_trainer.get_training_stats()
    
    def get_current_prompt(self) -> str:
        """Get the current RL-optimized prompt"""
        return self.rl_trainer.get_current_prompt()
    
    def save_checkpoint(self):
        """Manually save RL checkpoint"""
        self.rl_trainer.save_checkpoint()
    
    def get_feedback_history(self, n: int = 100) -> list:
        """Get recent feedback history"""
        entries = self.feedback_collector.get_recent_feedback(n)
        return [entry.to_dict() for entry in entries]
    
    def get_high_quality_examples(self, min_reward: float = 0.8, n: int = 50) -> list:
        """Get high-quality training examples"""
        entries = self.feedback_collector.get_high_quality_examples(min_reward, n)
        return [entry.to_dict() for entry in entries]


def enhance_agent_with_rl(base_agent, load_checkpoint: bool = True):
    """
    Factory function to enhance existing SmlpAgent with RL
    
    Args:
        base_agent: Existing SmlpAgent instance
        load_checkpoint: Whether to load previous RL state
        
    Returns:
        RLEnhancedSmlpAgent instance
    """
    return RLEnhancedSmlpAgent(base_agent, load_checkpoint)


# Additional utility functions for working with corrections

def compare_commands(generated: Dict, corrected: Dict) -> Dict:
    """
    Compare generated and corrected commands and return detailed diff
    
    Returns:
        Dictionary with:
            - matching: dict of matching parameters
            - added: dict of parameters user added
            - removed: dict of parameters user removed
            - modified: dict of parameters user changed
    """
    changes, is_perfect = extract_feedback_from_correction(generated, corrected)
    
    all_keys = set(generated.keys()) | set(corrected.keys())
    matching = {k: generated[k] for k in all_keys 
                if k in generated and k in corrected and generated[k] == corrected[k]}
    
    return {
        'matching': matching,
        'added': changes['added'],
        'removed': changes['removed'],
        'modified': changes['modified'],
        'is_perfect': is_perfect,
        'accuracy': len(matching) / len(all_keys) if all_keys else 1.0
    }


def suggest_corrections(generated: Dict, 
                       smlp_default_params: Dict,
                       common_errors: Optional[Dict] = None) -> Dict:
    """
    Suggest possible corrections based on common patterns
    
    Args:
        generated: LLM-generated command
        smlp_default_params: Default SMLP parameters
        common_errors: Optional dict of common error patterns
        
    Returns:
        Dictionary of suggested corrections
    """
    suggestions = {}
    
    # Check for missing commonly-used parameters
    common_params = ['log_files_prefix', 'analytics_mode', 'model_name']
    for param in common_params:
        if param not in generated and param in smlp_default_params:
            suggestions[param] = {
                'reason': 'Commonly used parameter missing',
                'suggested_value': smlp_default_params[param]
            }
    
    # Check for parameter value issues
    if 'analytics_mode' in generated:
        valid_modes = ['dataset', 'verify', 'explain', 'optimize', 'rag']
        if generated['analytics_mode'] not in valid_modes:
            suggestions['analytics_mode'] = {
                'reason': 'Invalid analytics mode',
                'suggested_value': 'dataset',
                'valid_options': valid_modes
            }
    
    return suggestions


if __name__ == "__main__":
    print("=== RL-Enhanced SMLP Agent Integration ===\n")
    
    # Example: How to use with existing SmlpAgent
    print("Usage example:")
    print("""
    # Import existing agent
    from smlp_agent import SmlpAgent
    from smlp_agent_rl_integration import enhance_agent_with_rl
    
    # Create base agent
    base_agent = SmlpAgent()
    
    # Enhance with RL
    rl_agent = enhance_agent_with_rl(base_agent, load_checkpoint=True)
    
    # Run command with RL optimization
    result = rl_agent.run_text_command("run dataset analysis on sales data")
    
    # Provide feedback if user makes corrections
    user_correction = {
        'analytics_mode': 'dataset',
        'data_file': 'sales_data.csv',
        'model_name': 'nn',
        'log_files_prefix': 'sales_analysis'
    }
    
    feedback = rl_agent.provide_feedback(
        user_query="run dataset analysis on sales data",
        generated_command=result['generated_command'],
        corrected_command=user_correction,
        execution_success=True
    )
    
    print(f"Feedback reward: {feedback['reward']:.3f}")
    
    # Get training statistics
    stats = rl_agent.get_training_stats()
    print(f"Total feedback collected: {stats['total_feedback']}")
    print(f"Average reward: {stats['avg_reward']:.3f}")
    
    # Save checkpoint
    rl_agent.save_checkpoint()
    """)
    
    print("\n" + "="*60)
    print("Integration module loaded successfully!")
    print("="*60)
