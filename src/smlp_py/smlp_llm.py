import os
import torch
from typing import Callable, Any, Union, Tuple, Optional
import json

from smlp_py.smlp_utils import str_to_bool
from smlp_py.smlp_judge import SmlpScratchJudge

# Ensures determinizm in training -- at least, reduces randomness in training: CPU-level
# ops and PyTorch internals may still introduce non-determinism unless you go further 
# (e.g., torch.use_deterministic_algorithms(True) — which may crash unsupported ops).
#random.seed(42)
#np.random.seed(42)
#torch.manual_seed(42)

'''
To achieve completely reproducible output:
1. Save your finetuned model (including tokenizer):
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")
2. Load it later with:
model = AutoModelForCausalLM.from_pretrained("my_model")
tokenizer = AutoTokenizer.from_pretrained("my_model")
3. Use do_sample=False to ensure no randomness in inference:
output = model.generate(input_ids, do_sample=False, ...)
'''

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    RagTokenizer,
    RagRetriever,
    RagTokenForGeneration,
    RagSequenceForGeneration,
    RagModel,
    Seq2SeqTrainer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from datasets import Dataset, DatasetDict

from tokenizers import ByteLevelBPETokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from smlp_py.transformer import BigramLanguageModel, Transformer
        
from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset

from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        src = inputs["input_ids"]
        tgt = inputs["labels"]

        # Replace -100 (ignore_index) in target with pad token (or 0)
        tgt = tgt.masked_fill(tgt == -100, 0)

        logits = model(src, tgt)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tgt[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

        if return_outputs:
            return loss, logits
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        src = inputs["input_ids"]
        tgt = inputs["labels"]

        # Replace -100 (ignore_index) in target with 0 for embedding compatibility
        tgt = tgt.masked_fill(tgt == -100, 0)

        with torch.no_grad():
            logits = model(src, tgt)

        if prediction_loss_only:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tgt[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            return (loss, None, None)

        return (None, logits, tgt)

class SmlpLlm:
    def __init__(self):
        self._llm_logger = None
        self._DEF_LLM_MODEL_CLASS = "gpt2"
        self._DEF_LLM_TRAINED_MODEL_PATH = None
        self._DEF_LLM_EPOCHS = 3
        self._DEF_LLM_BATCH_SIZE = 4
        self._DEF_LLM_BLOCK_SIZE = 128
        self._DEF_LLM_VOCAB_SIZE = 50000
        self._DEF_LLM_OUTPUT_DIR = None
        self._DEF_LLM_TRAIN = True
        self._DEF_LLM_GENERATE = True
        self._DEF_LLM_GENERATE_PROMPT = "Once upon a time"

        self.judge = SmlpScratchJudge()
        
        ''' TODO !!! add parameters to control LLM training
        | Name                          | Description                                                              |
        | ----------------------------- | ------------------------------------------------------------------------ |
        | `learning_rate`               | Step size in optimizer. Typical: `2e-5` to `5e-4`.                       |
        | `num_train_epochs`            | Number of full dataset passes. Typical: `3–10`.                          |
        | `per_device_train_batch_size` | Batch size per GPU/CPU.                                                  |
        | `gradient_accumulation_steps` | Accumulates gradients over multiple steps to simulate larger batch size. |
        | `lr_scheduler_type`           | E.g. `"linear"`, `"cosine"`, `"constant"`.                               |
        | `warmup_steps`                | Gradual LR increase at start. Typically 5–10% of total steps.            |
        | `weight_decay`                | Prevents overfitting. Usually `0.01` or `0.1`.                           |
        | `logging_steps`               | Print loss every N steps.                                                |
        | `save_steps`                  | Save checkpoints every N steps.                                          |
        | `eval_strategy`               | `steps` or `epoch`.                                                      |

        '''
        self.llm_params_dict = {
            "llm_model_class": {
                "abbr": "llm_model_class", "default": self._DEF_LLM_MODEL_CLASS, "type": str,
                "help": "The name of the LLM model architecture to use for training from scratch.\n"
                        "Supported values: 'gpt2', 'transformer', and 'bigram' (experimental)."
            },
            "llm_trained_model_path": {
                "abbr": "llm_trained_model_path", "default": self._DEF_LLM_TRAINED_MODEL_PATH, "type": str,
                "help": (
                    "Directory where the ftrained LLM model will be saved. "
                    "Includes model weights, tokenizer, config, and training artifacts."
                )
            },
            "llm_epochs": {
                "abbr": "llm_epochs", "default": self._DEF_LLM_EPOCHS, "type": int,
                "help": "Number of epochs for scratch training of the language model."
            },
            "llm_batch_size": {
                "abbr": "llm_batch_size", "default": self._DEF_LLM_BATCH_SIZE, "type": int,
                "help": "Training batch size (per device)."
            },
            "llm_block_size": {
                "abbr": "llm_block_size", "default": self._DEF_LLM_BLOCK_SIZE, "type": int,
                "help": "Block size for input tokenization and chunking."
            },
            "llm_vocab_size": {
                "abbr": "llm_vocab_size", "default": self._DEF_LLM_VOCAB_SIZE, "type": int,
                "help": "Vocabulary size for tokenizer training (if a tokenizer is trained from scratch)."
            },
            "llm_output_dir": {
                "abbr": "llm_output_dir", "default": self._DEF_LLM_OUTPUT_DIR, "type": str,
                "help": "Directory to save the trained model and tokenizer."
            },
            "llm_train": {
                "abbr": "llm_train", "default": self._DEF_LLM_TRAIN, "type": str_to_bool,
                "help": "Whether to run scratch model training (True/False)."
            },
            "llm_generate": {
                "abbr": "llm_generate", "default": self._DEF_LLM_GENERATE, "type": str_to_bool,
                "help": "Whether to run generation after training using the trained model."
            },
            "llm_prompt": {
                "abbr": "llm_prompt", "default": self._DEF_LLM_GENERATE_PROMPT, "type": str,
                "help": "Prompt to use for generation (used only if llm_generate=True)."
            },
            "llm_temperature": {
                "abbr": "llm_temperature", "default": 1.0, "type": float,
                "help": (
                    "Temperature for text generation sampling. "
                    "Must be a strictly positive float (e.g., 1.0 for moderate randomness, 0.7 for more focus). "
                    "Ignored if sampling is disabled. "
                    "Set close to 0 (e.g., 1e-5) for near-deterministic sampling."
                )
            },
            "llm_top_k": {
                "abbr": "llm_top_k", "default": 50, "type": int,
                "help": (
                    "Top-k sampling for text generation. Chooses from the top-k most likely next tokens.\n"
                    "Set to 0 to disable top-k filtering. Works only when sampling is enabled."
                )
            },
            "llm_do_sample": {
                "abbr": "llm_sample", "default": True, "type": bool,
                "help": (
                    "Enables random sampling from token probabilities (vs. greedy decoding). "
                    "Must be True to use temperature or top_k. Set to False for deterministic generation."
                )
            },
            "llm_max_new_tokens": {
                "abbr": "llm_max_new", "default": 128, "type": int,
                "help": "Maximum number of tokens to generate after the input prompt."
            },
            "llm_stop_token": {
                "abbr": "llm_stop", "default": None, "type": str,
                "help": "Optional stop sequence. If present in the output, generation will terminate early."
            },
        }

    def set_logger(self, logger):
        self._llm_logger = logger
        self.judge.set_logger(logger)

    def set_report_file_prefix(self, prefix):
        self.report_file_prefix = prefix
        self.judge.set_report_file_prefix(prefix)
    
    @property
    def generated_text_file_name(self):
        return self.report_file_prefix + '_llm_generated.txt'
    
    def prepare_tokenizer(self, dataset):
        if self.tokenizer:
            self._llm_logger.info("Using provided tokenizer")
            return self.tokenizer

        if self.model_class == "gpt2":
            self._llm_logger.info("Loading GPT2 tokenizer")
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            self._llm_logger.info("Training tokenizer from scratch")
            tmp_path = "./tmp_vocab"
            os.makedirs(tmp_path, exist_ok=True)

            text_file = os.path.join(tmp_path, "text.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                for line in dataset["train"]["text"]:
                    f.write((line or "") + "\n")

            tokenizer_trainer = ByteLevelBPETokenizer()
            tokenizer_trainer.train(files=text_file, vocab_size=self.vocab_size, min_frequency=2)
            tokenizer_trainer.save_model(tmp_path)

            tokenizer = GPT2TokenizerFast(
                vocab_file=os.path.join(tmp_path, "vocab.json"),
                merges_file=os.path.join(tmp_path, "merges.txt"),
            )
            tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
            import shutil

            # Clean up temporary tokenizer training directory
            shutil.rmtree(tmp_path, ignore_errors=True)

            
        tokenizer.save_pretrained(self.output_dir)
        self.tokenizer = tokenizer
        return tokenizer

    def prepare_dataset(self, texts):
        dataset = Dataset.from_dict({"text": texts})
        dataset = DatasetDict({
            "train": dataset,
            "test": dataset.select(range(min(10, len(dataset))))
        })

        tokenizer = self.prepare_tokenizer(dataset)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length",
                             max_length=self.block_size, return_attention_mask=True)

        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        def group_texts(examples):
            concatenated = sum(examples["input_ids"], [])
            total_length = (len(concatenated) // self.block_size) * self.block_size
            result = {
                "input_ids": [concatenated[i:i + self.block_size]
                              for i in range(0, total_length, self.block_size)]
            }
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized = tokenized.remove_columns(
            [col for col in tokenized["train"].column_names if col != "input_ids"]
        )
        lm_dataset = tokenized.map(group_texts, batched=True)
        lm_dataset.set_format(type="torch")

        return lm_dataset["train"], lm_dataset["test"]

    def build_model(self):
        if self.model_class == "bigram":
            return BigramLanguageModel(vocab_size=self.vocab_size, n_embed=8, block_size=8) #, vocab_size=100
            #return BigramLanguageModel(vocab_size=self.vocab_size, n_embed=128, block_size=self.block_size)
        elif self.model_class == "transformer":
            return Transformer(
                src_vocab_size=self.vocab_size,
                tgt_vocab_size=self.vocab_size,
                d_model=256,
                num_heads=8,
                num_layers=6,
                d_ff=256,
                max_seq_length=self.block_size,
                dropout=0.1
            )
        elif self.model_class == "gpt2":
            config = GPT2Config(
                #vocab_size=self.vocab_size, # setting this often implies smaller size than the number of tokens
                n_positions=self.block_size,
                n_embd=256,
                n_layer=4,
                n_head=4
            )
            return GPT2LMHeadModel(config)
        else:
            raise ValueError(f"Unsupported model class: {self.model_class}")

    def train(self, texts, epochs=3, batch_size=4):
        train_data, eval_data = self.prepare_dataset(texts)
        model = self.build_model()
        self.model = model

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10000000,
            save_total_limit=1,
            load_best_model_at_end=False,
            remove_unused_columns=False,
            report_to="none"
        )

        if self.model_class.lower() == "gpt2":
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=eval_data,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
        else:
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=eval_data,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self._llm_logger.info(f"Training complete. Model saved to {self.output_dir}")
        
    def load_model(self, path):
        if self.model_class.lower() == "gpt2":
            # Specifically for G{T-2 compatible model could use the commented out rows.
            # But definitions of self.tokenizer and self.model that are used are more general
            #from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            #self.model = GPT2LMHeadModel.from_pretrained(path)
            #self.tokenizer = GPT2TokenizerFast.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        elif self.model_class.lower() == "transformer":
            # load model
            from safetensors.torch import load_model as load_safetensors
            self.model = Transformer(
                src_vocab_size=self.vocab_size,
                tgt_vocab_size=self.vocab_size,
                d_model=256,
                num_heads=8,
                num_layers=6,
                d_ff=256,
                max_seq_length=self.block_size,
                dropout=0.1
            )
            #self.model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
            model_path = os.path.join(path, "model.safetensors")
            load_safetensors(self.model, model_path)  # replaces torch.load + load_state_dict
            self.model.eval()
            
            # load tokenizer
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"))
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.model_class.lower() == "bigram":
            # load l
            from safetensors.torch import load_model as load_safetensors
            self.model = BigramLanguageModel(vocab_size=self.vocab_size)
            #self.model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
            model_path = os.path.join(path, "model.safetensors")
            load_safetensors(self.model, model_path)
            self.model.eval()
            
            # load tokenizer
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"))
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError(f"Unsupported model_class: {self.model_class}")
    
    
    def generate(self, prompt: Union[str, torch.Tensor], max_new_tokens: int=50, do_sample: bool=False,
        temperature: float=1.0, top_k: int= 0, stop_token: Optional[str]=None, decode: bool=True,
    ) -> Union[str, torch.Tensor]:
        """
        Generate output given a prompt, supports GPT2, Transformer, and Bigram models.

        Args:
            prompt (str or torch.Tensor): Input prompt as string or token ids.
            max_new_tokens (int): Number of tokens to generate (excluding prompt).
            do_sample (bool): Whether to sample (True) or greedy decode (False).
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            stop_token (str, optional): If given, truncate output at this substring.
            decode (bool): Return decoded text if True, else return token ids.

        Returns:
            str or torch.Tensor: Generated text or token ids.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Prepare prompt tokens tensor
        if isinstance(prompt, str):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_tokens = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"]
        else:
            prompt_tokens = prompt.to(device)
            assert isinstance(prompt_tokens, torch.Tensor)

        # Case 1: GPT2 (HuggingFace model) supports .generate natively
        if self.model_class.lower() == "gpt2":
            assert isinstance(prompt, str)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            outputs = self.model.generate(
                input_ids=prompt_tokens,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k if do_sample else 0,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            output_tokens = outputs[0]

        # Case 2: Custom Transformer (encoder-decoder)
        elif self.model_class.lower() == "transformer":
            input_ids=prompt_tokens
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, max_new_tokens=20)
                #output_text = self.model.decode(output_ids[0].tolist())
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                #print('output_text', output_text)
                        
            src = prompt_tokens
            #print("Vocab sample:", list(self.tokenizer.get_vocab().keys()))
            # Start with BOS or prompt tokens as tgt; if no BOS defined, start empty
            start_token_id = (
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else
                self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else
                0
            )
            
            if self.tokenizer.eos_token is None:
                # Try setting eos_token to '<|endoftext|>' (common GPT2 EOS) or your model's EOS token string
                self.tokenizer.eos_token = self.tokenizer.pad_token or ""
            if self.tokenizer.eos_token_id is None:
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
            if "<|endoftext|>" in self.tokenizer.get_vocab():
                self.tokenizer.eos_token = "<|endoftext|>"
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            else:
                self._llm_logger.warning("Missing <|endoftext|> in vocab!")

            #print("Tokenizer eos_token:", self.tokenizer.eos_token)
            #print("Tokenizer eos_token_id:", self.tokenizer.eos_token_id)
            #print("Token ID 19217:", self.tokenizer.decode([19217]))
            
            generated_ids = torch.full((src.shape[0], 1), start_token_id, device=device, dtype=torch.long)
            #print('max_new_tokens =', max_new_tokens)
            tokens = self.tokenizer("This is a sample sentence to test block size.", return_tensors=None)["input_ids"]
            #print('first 128 tokens', tokens[:128])  # inspect the token chunk
            #print('first 128 tokens decoded', self.tokenizer.decode(tokens[:128]))  # see how it maps back to text

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = self.model(src, generated_ids)  # forward requires src and tgt
                logits = logits[:, -1, :]  # last token logits
                probs = torch.softmax(logits / temperature, dim=-1)
                
                if do_sample:
                    if top_k > 0:
                        topk_probs, topk_indices = torch.topk(probs, top_k)
                        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                        next_token = topk_indices.gather(
                            -1,
                            torch.multinomial(topk_probs, num_samples=1)
                        )
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            output_tokens = generated_ids[0]

        # Case 3: BigramLanguageModel (simple decoder-only model)
        elif self.model_class.lower() == "bigram":
            idx_cond = prompt_tokens
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = self.model(idx_cond)
                logits = logits[:, -1, :]  # last token logits
                probs = torch.softmax(logits / temperature, dim=-1)

                if do_sample:
                    if top_k > 0:
                        topk_probs, topk_indices = torch.topk(probs, top_k)
                        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                        next_token = topk_indices.gather(
                            -1,
                            torch.multinomial(topk_probs, num_samples=1)
                        )
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                idx_cond = torch.cat([idx_cond, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            output_tokens = idx_cond[0]

        else:
            raise ValueError(f"Unsupported model_class: {self.model_class}")
        
        if decode:
            generated_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            if stop_token is not None and stop_token in generated_text:
                generated_text = generated_text.split(stop_token)[0]
            self._llm_logger.info(f"Generated text: {generated_text}")
            self._llm_logger.info(f"Saving generated text into {self.generated_text_file_name}")
            with open(self.generated_text_file_name, "w") as file:
                file.write(generated_text)
            return generated_text
        else:
            return output_tokens
        

    def smlp_llm(self, llm_text:str=None, llm_model_class=None, llm_trained_model_path=None, llm_prompt:str=None, 
                llm_train:bool=True, llm_generate:bool=True, llm_vocab_size=None, llm_block_size=None, llm_epochs=None,
                llm_batch_size=None, llm_quality_method=None, llm_judge_model=None, llm_judge_max_examples=None,
                llm_judge_prompt=None):

        # Load text
        with open(llm_text, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"No non-empty lines found in file: {llm_text}")

        self.model_class = llm_model_class.lower()
        self.vocab_size = llm_vocab_size
        self.block_size = llm_block_size
        self.output_dir = llm_trained_model_path
        self.tokenizer = None
        self.model = None

        # Training
        if llm_train:
            self.train(
                texts=lines,
                epochs=llm_epochs,
                batch_size=llm_batch_size,
            )

        # Judge after training
        if llm_train and llm_quality_method == 'judge':
            self.load_model(llm_trained_model_path)

            # Simple heuristic: generate continuations for training text
            generated = []

            for text in lines[: llm_judge_max_examples]:
                gen = self.generate(prompt=text)
                generated.append(gen)
            
            judge_results = self.judge.run(
                training_texts=lines,
                generated_texts=generated,
                llm_judge_model=llm_judge_model,
                llm_judge_max_examples=llm_judge_max_examples,
                llm_judge_prompt=llm_judge_prompt,
                train_vs_gen=True
            )
            
        # Generation
        if llm_generate:
            self.load_model(llm_trained_model_path)
            result = self.generate(prompt=llm_prompt)
            self._llm_logger.info(f"Generated: {result}")

            # Judge after generation
            if llm_quality_method == 'judge':
                gen_eval = self.judge.run( #evaluate_generation(
                    training_texts=llm_prompt,
                    generated_texts=result,
                    llm_judge_model=llm_judge_model,
                    llm_judge_max_examples=llm_judge_max_examples,
                    llm_judge_prompt=llm_judge_prompt,
                    train_vs_gen=False
                )
