# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
Ollama Model Fine-Tuning Module for SMLP Agent

This module handles fine-tuning of local Ollama models using
collected feedback data. It prepares training data and executes
the fine-tuning process.

Fine-tuning approach:
1. Collect high-quality examples from feedback
2. Format as training data for Ollama
3. Create a Modelfile for fine-tuning
4. Fine-tune using Ollama CLI
5. Test the fine-tuned model
"""

import json
import os
import subprocess
import datetime
from typing import List, Dict, Optional
from pathlib import Path


class OllamaFineTuner:
    """
    Handles fine-tuning of Ollama models using feedback data
    """
    
    def __init__(self, 
                 base_model: str = "mistral",
                 output_dir: str = "./smlp_fine_tuned_models"):
        """
        Args:
            base_model: Base Ollama model to fine-tune
            output_dir: Directory to store fine-tuned models and artifacts
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_training_data(self, 
                            feedback_entries: List[Dict],
                            min_reward: float = 0.8) -> List[Dict]:
        """
        Prepare training data from feedback entries
        
        Args:
            feedback_entries: List of feedback entry dictionaries
            min_reward: Minimum reward threshold for inclusion
            
        Returns:
            List of training examples in chat format
        """
        training_data = []
        
        for entry in feedback_entries:
            # Only use high-quality examples
            if entry['reward'] < min_reward:
                continue
            
            # Format as chat-style training example
            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": entry['user_query']
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(entry['corrected_command'], indent=2)
                    }
                ]
            }
            
            training_data.append(example)
        
        return training_data
    
    def save_training_data(self, 
                          training_data: List[Dict],
                          filename: str = None) -> Path:
        """
        Save training data to JSONL file
        
        Args:
            training_data: List of training examples
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.jsonl"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"Training data saved to {filepath}")
        print(f"Total examples: {len(training_data)}")
        
        return filepath
    
    def create_modelfile(self, 
                        system_prompt: str,
                        model_name: str = None) -> Path:
        """
        Create Ollama Modelfile for fine-tuning
        
        Args:
            system_prompt: System prompt for the model
            model_name: Optional custom model name
            
        Returns:
            Path to created Modelfile
        """
        if model_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"smlp_agent_{timestamp}"
        
        modelfile_content = f"""FROM {self.base_model}

# Set the system prompt
SYSTEM \"\"\"
{system_prompt}
\"\"\"

# Set parameters for better code generation
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40

# Set stop sequences
PARAMETER stop "User:"
PARAMETER stop "Assistant:"
"""
        
        filepath = self.output_dir / f"Modelfile_{model_name}"
        
        with open(filepath, 'w') as f:
            f.write(modelfile_content)
        
        print(f"Modelfile created at {filepath}")
        return filepath
    
    def fine_tune_model(self,
                       training_data_path: Path,
                       model_name: str,
                       system_prompt: str,
                       epochs: int = 3,
                       batch_size: int = 4) -> Dict:
        """
        Fine-tune Ollama model using prepared data
        
        Note: This is a simplified version. Actual Ollama fine-tuning
        may require different approaches depending on the Ollama version.
        
        Args:
            training_data_path: Path to training data JSONL
            model_name: Name for fine-tuned model
            system_prompt: System prompt for the model
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with fine-tuning results
        """
        # Create Modelfile
        modelfile_path = self.create_modelfile(system_prompt, model_name)
        
        # Create the model using Ollama CLI
        # This creates a new model based on the Modelfile
        try:
            print(f"\nCreating model '{model_name}' from {modelfile_path}...")
            
            create_cmd = [
                "ollama", "create", model_name,
                "-f", str(modelfile_path)
            ]
            
            result = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"Model created successfully!")
            print(f"stdout: {result.stdout}")
            
            # Note: Actual fine-tuning on training data would require
            # additional steps or tools. Ollama doesn't natively support
            # supervised fine-tuning via CLI (as of current versions).
            # 
            # Options for actual fine-tuning:
            # 1. Use ollama-python library with custom training loop
            # 2. Export to GGUF format, fine-tune with llama.cpp, re-import
            # 3. Use third-party tools like axolotl or unsloth
            # 4. Wait for native Ollama fine-tuning support
            
            return {
                "status": "success",
                "model_name": model_name,
                "base_model": self.base_model,
                "modelfile": str(modelfile_path),
                "message": "Model created with custom system prompt. "
                          "Note: Full supervised fine-tuning requires additional tools."
            }
            
        except subprocess.CalledProcessError as e:
            print(f"Error creating model: {e}")
            print(f"stderr: {e.stderr}")
            return {
                "status": "error",
                "error": str(e),
                "stderr": e.stderr
            }
    
    def export_training_script(self, 
                              training_data_path: Path,
                              model_name: str) -> Path:
        """
        Export a Python script for fine-tuning using external tools
        
        This creates a script that can be run with tools like:
        - Hugging Face transformers
        - axolotl
        - unsloth
        
        Args:
            training_data_path: Path to training data
            model_name: Name for the model
            
        Returns:
            Path to generated script
        """
        script_content = f'''#!/usr/bin/env python3
"""
SMLP Agent Fine-Tuning Script

This script fine-tunes a model for SMLP command generation.
Requires: transformers, datasets, peft, accelerate

Install: pip install transformers datasets peft accelerate

Usage:
    python fine_tune_smlp.py
"""

import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
MODEL_NAME = "{self.base_model}"
OUTPUT_DIR = "./smlp_fine_tuned_{model_name}"
TRAINING_DATA = "{training_data_path}"

def load_training_data(filepath):
    """Load training data from JSONL"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_prompt(example):
    """Format training example as prompt"""
    messages = example['messages']
    user_msg = messages[0]['content']
    assistant_msg = messages[1]['content']
    
    return {{
        'text': f"User: {{user_msg}}\\nAssistant: {{assistant_msg}}"
    }}

def main():
    print("Loading training data...")
    raw_data = load_training_data(TRAINING_DATA)
    formatted_data = [format_prompt(ex) for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)
    
    print(f"Loaded {{len(dataset)}} training examples")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto"
    )
    
    # Prepare for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=['text']
    )
    
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=50,
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Training complete! Model saved to {{OUTPUT_DIR}}")
    
    # Save the final model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
'''
        
        script_path = self.output_dir / f"fine_tune_{model_name}.py"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"Fine-tuning script created at {script_path}")
        print(f"To run: python {script_path}")
        
        return script_path
    
    def test_model(self, model_name: str, test_queries: List[str]) -> List[Dict]:
        """
        Test a fine-tuned model with sample queries
        
        Args:
            model_name: Name of model to test
            test_queries: List of test queries
            
        Returns:
            List of test results
        """
        results = []
        
        for query in test_queries:
            try:
                # Run inference using Ollama
                cmd = [
                    "ollama", "run", model_name,
                    query
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                results.append({
                    "query": query,
                    "response": result.stdout,
                    "success": result.returncode == 0
                })
                
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        return results


# Convenience function for the full fine-tuning workflow
def fine_tune_from_feedback(feedback_collector,
                           base_model: str = "mistral",
                           min_reward: float = 0.8,
                           model_name: str = None) -> Dict:
    """
    Complete workflow: feedback → training data → fine-tuned model
    
    Args:
        feedback_collector: FeedbackCollector instance with data
        base_model: Base Ollama model
        min_reward: Minimum reward for training examples
        model_name: Optional custom model name
        
    Returns:
        Dictionary with fine-tuning results
    """
    # Initialize fine-tuner
    fine_tuner = OllamaFineTuner(base_model=base_model)
    
    # Get high-quality feedback
    high_quality = feedback_collector.get_high_quality_examples(
        min_reward=min_reward,
        n=1000
    )
    
    if len(high_quality) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(high_quality)} high-quality examples. Need at least 10.",
            "min_reward": min_reward
        }
    
    print(f"Found {len(high_quality)} high-quality training examples")
    
    # Prepare training data
    training_data = fine_tuner.prepare_training_data(
        [entry.to_dict() for entry in high_quality],
        min_reward=min_reward
    )
    
    # Save training data
    data_path = fine_tuner.save_training_data(training_data)
    
    # Create system prompt
    system_prompt = """You are an AI assistant that converts natural language descriptions into SMLP command-line parameter dictionaries.

Given a user's request, output ONLY a JSON dictionary with SMLP parameters. Do not include explanations or markdown formatting.

Example:
User: "run dataset analysis on sales data with neural network"
Output: {"analytics_mode": "dataset", "model_name": "nn", "data_file": "sales_data.csv"}"""
    
    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"smlp_agent_{timestamp}"
    
    # Fine-tune (or create customized model)
    result = fine_tuner.fine_tune_model(
        training_data_path=data_path,
        model_name=model_name,
        system_prompt=system_prompt
    )
    
    # Also export Python script for full fine-tuning
    script_path = fine_tuner.export_training_script(data_path, model_name)
    
    result['training_data_path'] = str(data_path)
    result['fine_tune_script'] = str(script_path)
    result['num_training_examples'] = len(training_data)
    
    return result


if __name__ == "__main__":
    print("=== Ollama Fine-Tuning Module ===\n")
    
    # Example usage
    print("Example workflow:")
    print("""
from smlp_agent_rl import FeedbackCollector
from smlp_agent_rl_finetune import fine_tune_from_feedback

# Collect feedback (this happens during normal usage)
feedback_collector = FeedbackCollector()

# ... after collecting sufficient feedback ...

# Fine-tune model
result = fine_tune_from_feedback(
    feedback_collector=feedback_collector,
    base_model="mistral",
    min_reward=0.8,
    model_name="smlp_agent_v1"
)

print(f"Status: {result['status']}")
print(f"Model: {result['model_name']}")
print(f"Training examples: {result['num_training_examples']}")

# The model can now be used with:
# ollama run smlp_agent_v1 "your query here"
    """)
    
    print("\n" + "="*60)
    print("Note: Full supervised fine-tuning requires additional tools.")
    print("See the generated Python script for transformers-based training.")
    print("="*60)
