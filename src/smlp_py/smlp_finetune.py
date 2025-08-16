# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

'''
model_name = "TinyLlama/TinyLlama-1.1B-Chat"
Size: 1.1B parameters
Architecture: LLaMA-like
Optimized for chat and fine-tuning
Works well with the guanaco dataset


model_name = "google/flan-t5-base"
Size: ~250M
Not a decoder-only model (it's an encoder-decoder)
Easy to fine-tune on instruction datasets
Lightweight, CPU-friendly


model = "Locutusque/TinyMistral-248M"
Size: 248M
Very fast and light
Based on Mistral architecture (chat-capable, compact)
'''

import os, sys
import torch
import importlib
from datasets import load_dataset
from datasets import Dataset as HFDataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSeq2SeqLM,
    pipeline, 
    AutoModelForQuestionAnswering,
    PreTrainedTokenizer,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator,
    AutoConfig,
    #Trainer
)
# impporting Trainer directly fromm transformers does not work properly, at least with transformers=4.52.4
from transformers.trainer import Trainer


from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
#import openai

from smlp_py.smlp_utils import str_to_bool


# This class is introduced to be able to define custom function compute_loss 
# that overrides compute_loss defined within Trainer class in transformers
class QAWithCustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            start_positions=inputs["start_positions"],
            end_positions=inputs["end_positions"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class SimpleTokenizer:
    def __init__(self, encode_fn, decode_fn):
        self.encode = encode_fn
        self.decode = decode_fn

'''
# The model that we want to re-train (finetune) from Hugging Face Hub
model_name = "NousResearch/Llama-2-7b-chat-hf"
model_name = "meta-llama/Meta-Llama-3-8B" 
model_name = "tiiuae/falcon-rw-1b" 
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "Locutusque/TinyMistral-248M"

# The training (fine-tuning) data to use
dataset_name = "mlabonne/guanaco-llama2-1k"
'''



# Core class
class SmlpFinetune:
    def __init__(self):
        # Constants (can also be class attributes if you prefer)
        self._DEF_FINETUNE_BASE_MODEL_NAME = None #"tiiuae/falcon-rw-1b"
        self._DEF_FINETUNE_DATASET_NAME = None #"lmsys/chatbot_arena_conversations"
        self._DEF_FINETUNE_TRAINED_MODEL_PATH = None
        self._DEF_FINETUNE_NUM_EPOCHS = 3
        self._DEF_FINETUNE_BATCH_SIZE = 4
        self._DEF_FINETUNE_USE_4BIT = True
        self._DEF_FINETUNE_LORA_R = 8
        self._DEF_FINETUNE_LORA_ALPHA = 16
        self._DEF_FINETUNE_LORA_DROPOUT = 0.05
        self._DEF_FINETUNE_FP16 = False
        self._DEF_FINETUNE_BF16 = False
        self._DEF_FINETUNE_MAX_SEQ_LENGTH = 512
        self._DEF_FINETUNE_SAVE_STEPS = 0
        self._DEF_FINETUNE_SAVE_STRATEGY = 'no' # 'batch'
        self._DEF_FINETUNE_TRAIN = True
        self._DEF_FINETUNE_EVAL = True
        self._DEF_FINETUNE_PROMPT = None
        self._DEF_FINETUNE_CONTEXT = None
        self._DEF_FINETUNE_TASK_TYPE = "text-generation"
        self._DEF_FINETUNE_MAX_NEW_TOKENS = 128
        self._DEF_FINETUNE_SAMPLE = False
        self._DEF_FINETUNE_TEMPERATURE = 1e-5
        self._DEF_FINETUNE_NUM_BEAMS = 1
        self._DEF_FINETUNE_TOP_K = 50
        self._DEF_FINETUNE_TOP_P = 0.95
        self._DEF_FINETUNE_REPETITION_PENALTY = 1.2
        
        
        # CLI-compatible config dictionary
        self.finetune_config_dict = {
            "finetune_base_model_name": {
                "abbr": "finetune_base_model_name", "default": self._DEF_FINETUNE_BASE_MODEL_NAME, "type": str,
                "help": (
                    "Name or local path of the pre-trained base model to fine-tune. "
                    "Can be any HuggingFace-supported model (e.g., 'bert-base-uncased', 'google/flan-t5-small', "
                    "'TinyLlama/TinyLlama-1.1B-Chat'). Ensure the model type matches the selected task "
                    "(decoder-only for text-generation, encoder-decoder for summarization/QA, etc.)."
                )
            },
            "finetune_trained_model_path": {
                "abbr": "finetune_trained_model_path", "default": self._DEF_FINETUNE_TRAINED_MODEL_PATH, "type": str,
                "help": (
                    "Directory where the fine-tuned model will be saved. "
                    "Includes model weights, tokenizer, config, and training artifacts."
                )
            },
            "finetune_train": {
                "abbr": "finetune_train", "default": self._DEF_FINETUNE_TRAIN, "type": str_to_bool,
                "help": (
                    "Whether to run model fine-tuning. "
                    "Set to False to skip training and only run inference using an already fine-tuned model."
                )
            },
            "finetune_eval": {
                "abbr": "finetune_eval", "default": self._DEF_FINETUNE_EVAL, "type": str_to_bool,
                "help": (
                    "Whether to evaluate the model after fine-tuning using the given prompt and/or context. "
                    "If True, the model will generate an answer or output for the test input provided."
                )
            },
            "finetune_num_train_epochs": {
                "abbr": "finetune_epochs", "default": self._DEF_FINETUNE_NUM_EPOCHS, "type": int,
                "help": (
                    "Number of epochs (full passes over the dataset) to train the model. "
                    "Increase for more training, but beware of overfitting on small datasets."
                )
            },
            "finetune_per_device_train_batch_size": {
                "abbr": "finetune_batch", "default": self._DEF_FINETUNE_BATCH_SIZE, "type": int,
                "help": (
                    "Batch size per training device (CPU or GPU). "
                    "Total effective batch size = batch size * gradient_accumulation_steps * num_devices."
                )
            },
            "finetune_use_4bit": {
                "abbr": "finetune_4bit", "default": self._DEF_FINETUNE_USE_4BIT, "type": str_to_bool,
                "help": (
                    "Enable 4-bit QLoRA quantization to reduce memory usage. "
                    "Requires bitsandbytes library and GPU support. Ignored on CPU."
                )
            },
            "finetune_lora_r": {
                "abbr": "finetune_r", "default": self._DEF_FINETUNE_LORA_R, "type": int,
                "help": (
                    "Rank of the LoRA low-rank matrices. "
                    "Smaller values reduce memory and computation; larger values increase capacity."
                )
            },
            "finetune_lora_alpha": {
                "abbr": "finetune_alpha", "default": self._DEF_FINETUNE_LORA_ALPHA, "type": int,
                "help": (
                    "LoRA scaling factor (alpha). "
                    "Controls how strongly the LoRA updates influence the base model weights."
                )
            },
            "finetune_lora_dropout": {
                "abbr": "finetune_dropout", "default": self._DEF_FINETUNE_LORA_DROPOUT, "type": float,
                "help": (
                    "Dropout probability applied to the LoRA adapter during training. "
                    "Helps regularize the adaptation; set to 0 for deterministic training."
                )
            },
            "finetune_fp16": {
                "abbr": "finetune_fp16", "default": self._DEF_FINETUNE_FP16, "type": str_to_bool,
                "help": (
                    "Enable mixed precision (FP16) training. "
                    "Recommended for compatible GPUs to speed up training and reduce memory usage."
                )
            },
            "finetune_bf16": {
                "abbr": "finetune_bf16", "default": self._DEF_FINETUNE_BF16, "type": str_to_bool,
                "help": (
                    "Enable bfloat16 (BF16) precision training. "
                    "Use only on GPUs that support BF16 (e.g., A100, RTX 30xx+). "
                    "Cannot be used with FP16 simultaneously."
                )
            },
            "finetune_max_seq_length": {
                "abbr": "finetune_seq", "default": self._DEF_FINETUNE_MAX_SEQ_LENGTH, "type": int,
                "help": (
                    "Maximum tokenized sequence length during training. Longer sequences will be truncated. "
                    "Must match model architecture limits (e.g., 512 for BERT, 2048+ for GPT models)."
                )
            },
            "finetune_save_steps": {
                "abbr": "ft_save_steps", "default": self._DEF_FINETUNE_SAVE_STEPS,  "type": int,
                "help": (
                    "Save a model checkpoint every N steps during training. "
                    "Each checkpoint includes the model weights, optimizer, and scheduler states. "
                    "Set to 0 to disable step-based saving. Ignored if `finetune_save_strategy` is 'no'. "
                )
            },
            "finetune_save_strategy": {
                "abbr": "ft_save_strategy", "default": self._DEF_FINETUNE_SAVE_STRATEGY, "type": str,
                "help": (
                    "Strategy for when to save model checkpoints. Options:\n"
                    " - 'no': Never save automatically (manual saving only).\n"
                    " - 'steps': Save every N steps (see --finetune_save_steps).\n"
                    " - 'epoch': Save at the end of each epoch. "
                )
            },
            # generation parameters (optional test after training)
            "finetune_prompt": {
                "abbr": "finetune_prompt", "default": self._DEF_FINETUNE_PROMPT, "type": str,
                "help": (
                    "Prompt text used during post-finetuning generation of fine-tuned model output. "
                    "Applicable to all supported fine-tuning tasks: 'text-generation', 'qa', and 'summarization'. "
                    "For text-generation and summarization tasks, the prompt is passed directly to the model. "
                    "For QA tasks with encoder-decoder models (e.g., T5), the prompt should be a full "
                    "'question: ... context: ...' string. "
                    "For extractive QA models (e.g., BERT), use this in conjunction with --finetune_context."
                )
            },
            "finetune_context": {
                "abbr": "finetune_context", "default": self._DEF_FINETUNE_CONTEXT, "type": str,
                "help": (
                    "Context text to accompany the question during post-finetuning generation for QA tasks. "
                    "This is required only when using decoder-only or extractive models like BERT for QA. "
                    "If using encoder-decoder models (e.g., T5), context is typically embedded in the prompt itself. "
                    "This option is ignored for text-generation and summarization tasks."
                )
            },
            "finetune_task_type": {
                "abbr": "finetune_task", "default": self._DEF_FINETUNE_TASK_TYPE,  "type": str,
                "help": (
                    "Task type to specify the objective of fine-tuning. "
                    "Supported values: 'text-generation', 'qa' (question answering), and 'summarization'. "
                    "This affects dataset formatting, model type selection, and evaluation logic."
                )
            },
            "finetune_max_new_tokens": {"abbr": "finetune_max_tokens", "default": self._DEF_FINETUNE_MAX_NEW_TOKENS, "type": int,
                "help": (
                    "Maximum number of tokens to generate during post-training evaluation. "
                    f"Controls output length for all fine-tuning tasks. "
                )
            },
            "finetune_sample": {
                "abbr": "finetune_sample", "default": self._DEF_FINETUNE_SAMPLE, "type": str_to_bool,
                "help": (
                    "Whether to use sampling during text generation after fine-tuning. "
                    "Set to False for deterministic greedy decoding. "
                    "Set to True to enable sampling (temperature, top-k/top-p) for diverse outputs."
                )
            },
            "finetune_temperature": {
                "abbr": "finetune_temperature", "default": self._DEF_FINETUNE_TEMPERATURE, "type": float,
                "help": (
                    "Temperature for text generation sampling. "
                    "Must be a strictly positive float (e.g., 1.0 for moderate randomness, 0.7 for more focus). "
                    "Ignored if --finetune_sample is False. "
                    "Set close to 0 (e.g., 1e-5) for near-deterministic sampling."
                )
            },
            "finetune_num_beams": {
                "abbr": "ft_beams", "default": self._DEF_FINETUNE_NUM_BEAMS, "type": int,
                "help": (
                    "Number of beams for beam search during generation. "
                    "Use 1 to disable beam search and prefer greedy decoding. "
                    "Higher values (e.g., 4 or 8) explore more paths for potentially better outputs but are slower."
                )
            },
            "finetune_top_k": {
                "abbr": "ft_top_k", "default": self._DEF_FINETUNE_TOP_K, "type": int,
                "help": (
                    "Top-k sampling for text generation. Chooses from the top-k most likely next tokens.\n"
                    "Set to 0 to disable top-k filtering (default: 50). Works only when sampling is enabled."
                )
            },
            "finetune_top_p": {
                "abbr": "ft_top_p", "default": self._DEF_FINETUNE_TOP_P, "type": float,
                "help": (
                    "Top-p (nucleus) sampling for text generation. Chooses from the smallest set of tokens "
                    "whose cumulative probability exceeds p. Set to 1.0 to disable (default: 0.95).\n"
                    "Effective only when sampling is enabled (see --finetune_sample)."
                )
            },
            "finetune_repetition_penalty": {
                "abbr": "ft_repeat_penalty", "default": self._DEF_FINETUNE_REPETITION_PENALTY, "type": float,
                "help": (
                    "Penalty applied to repeated tokens during generation. Values > 1 discourage repetition.\n"
                    "Set to 1.0 to disable the penalty (default: 1.2). Applies to all generation tasks."
                )
            },
        }
        
        self._finetune_logger = None
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._finetune_logger = logger 
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
    
    @property
    def generated_text_file_name(self):
        return self.report_file_prefix + '_finetuned_generated.txt'
        
    # Utility functions
    def get_lora_target_modules(self, model):
        common_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        found_modules = set()
        for name, _ in model.named_modules():
            for target in common_modules:
                if target in name:
                    found_modules.add(target)
        return list(found_modules)
    
    # not used
    def is_quantization_compatible(self, model_name):
        supported = ["llama", "mistral", "falcon", "bloom", "gpt", "opt"]
        model_name_lower = model_name.lower()
        return any(s in model_name_lower for s in supported)
    
    # not used
    def is_decoder_only(self, model):
        return (
            hasattr(model.config, "is_encoder_decoder") and not model.config.is_encoder_decoder
        ) or (
            hasattr(model, "transformer") and hasattr(model.transformer, "wte")
        ) or isinstance(model, AutoModelForCausalLM)
    
    
    def load_tokenizer(self, model_path: str) -> PreTrainedTokenizer:
        if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
            try:
                return GPT2TokenizerFast.from_pretrained(model_path)
            except Exception:
                return AutoTokenizer.from_pretrained(model_path)
        else:
            return AutoTokenizer.from_pretrained(model_path)

    '''
    | Feature                                   | Present? | Comments                                                 |
    | ----------------------------------------- | -------- | -------------------------------------------------------- |
    | Handles string and in-session model types | Yes      | Uses `hasattr(model, "generate")`                        |
    | Auto tokenizer loading with fallback      | Yes      | Tries `GPT2TokenizerFast`, falls back to `AutoTokenizer` |
    | QA and summarization support              | Yes      | With automatic pipeline selection                        |
    | Proper tokenizer padding, EOS handling    | Yes      | Defaults correctly if missing                            |
    | Generation kwargs customizable            | Yes      | Merges defaults and user args safely                     |
    | Device compatibility                      | Yes      | Uses `model.device` or defaults to CPU                   |
    | Output printing with prompts              | Yes      | Great for debugging or CLI use                           |
    | Avoids crash on missing pad/eos token     | Yes      | Adds fallback logic                                      |

    '''
    
    '''
    Covers OpenAI support via plan_from_text() if needed
    Supports vocab/decode_fn for in-session models
    Integrates HF models with inference
    Supports QA / summarization / text-gen / custom decode
    Does better device handling, padding, etc.
    '''
    '''
    task_type='text-generation' → uses AutoModelForCausalLM (e.g., GPT2, LLaMA, Mistral).
    task_type='summarization' → uses AutoModelForSeq2SeqLM (e.g., t5-small, bart-large).
    task_type='qa' → uses AutoModelForQuestionAnswering (e.g., bert-base-uncased, distilbert, roberta).
    It also:
    Autodetects task when task_type="auto" (based on model.config).
    Supports Mistral, LLaMA, Falcon, etc., if available on HuggingFace.
    Does not support OpenAI models unless you explicitly add an OpenAI case (can be done via API call).
    Does not support RAG pre-trained or RAG-trained models.
    Supports models trained from scratch Yes, but with this requirement:
    -- must pass the model and tokenizer explicitly, and
    -- The model must implement .generate() and .decode() (SMLP in-session Bigram/Transformer models do this).
    '''
    
    def load_finetune_dataset(self, finetune_dataset_name):
        if finetune_dataset_name.endswith(".json"):
            self._finetune_logger.info(f"Loading finetune dataset {finetune_dataset_name}")
            if not os.path.isfile(finetune_dataset_name):
                raise FileNotFoundError(f"File not found: {finetune_dataset_name}")
            return load_dataset("json", data_files=finetune_dataset_name, split="train")

        elif finetune_dataset_name.endswith(".txt"):
            if not os.path.isfile(finetune_dataset_name):
                raise FileNotFoundError(f"File not found: {finetune_dataset_name}")
            with open(finetune_dataset_name, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return HFDataset.from_dict({"text": lines})

        elif finetune_dataset_name.endswith(".pdf"):
            import fitz  # PyMuPDF
            if not os.path.isfile(finetune_dataset_name):
                raise FileNotFoundError(f"File not found: {finetune_dataset_name}")
            doc = fitz.open(finetune_dataset_name)
            lines = [page.get_text() for page in doc]
            flat_lines = [line.strip() for text in lines for line in text.split("\n") if line.strip()]
            return HFDataset.from_dict({"text": flat_lines})

        else:
            # Assume it's a HuggingFace dataset name like "ag_news"
            return load_dataset(finetune_dataset_name, split="train")

        
    def load_model_for_task(self, task_type, model_name, use_cpu=False, use_4bit=False):
        from transformers import (
            AutoConfig, AutoModelForCausalLM,
            AutoModelForQuestionAnswering,
            AutoModelForSeq2SeqLM,
        )
        import torch

        config = AutoConfig.from_pretrained(model_name)

        if task_type == "text-generation":
            if use_4bit and not use_cpu:
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float32,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=False,
                    )
                except ImportError:
                    raise ImportError("bitsandbytes not found. Install or set --finetune_use_4bit f.")
            else:
                bnb_config = None

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if not use_cpu else None,
                quantization_config=bnb_config
            )
            model.config.use_cache = False
            model.config.pretraining_tp = 1
            return model

        elif task_type == "qa":
            if config.is_encoder_decoder:
                return AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                return AutoModelForQuestionAnswering.from_pretrained(model_name)

        elif task_type == "summarization":
            if not config.is_encoder_decoder:
                raise ValueError("Summarization requires encoder-decoder model (e.g. T5, BART).")
            return AutoModelForSeq2SeqLM.from_pretrained(model_name)

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    # Ready datasets that can be used are "wikitext" or "ag_news"
    # Currrntly supports decoder-only causal language models, which are used for text 
    # generation -- such as TinyLlama, LLaMA, Mistral, Falcon, GPT2, etc.
    # Does not support encoder-decoder (e.g., T5, BART) used in summarization, and
    # Does not support QA-specific architectures like BERT, RoBERTa, etc. 
    # To support summarization and QA, need to 
    # --use a different trainer or modify flow to support Seq2Seq or encoder-only models.
    # --use AutoModelForSeq2SeqLM for summarization or AutoModelForQuestionAnswering for QA.
    '''
    During training, the following is saved in output_dir:
    Saving checkpoints
        Models are saved at each save_steps interval (e.g., ./outputs/checkpoint-500/)
        These checkpoints include:
        Trained model weights (pytorch_model.bin or adapter_model.bin if using PEFT)
        Tokenizer files (optional)
        Optimizer state (optimizer.pt, etc.)
        Trainer state (trainer_state.json)
    After training is finished, the final model is saved to output_dir/.
        If you're using LoRA/QLoRA (via peft), only the adapter weights are saved (unless you merge manually).
        Tokenizer is also saved using: tokenizer.save_pretrained(output_dir)
        If report_to="tensorboard", logs like loss and learning rate go into output_dir/runs/
    The file structure:
    outputs/
    ├── config.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── adapter_model.bin (if using LoRA)
    ├── runs/                     <- TensorBoard logs
    ├── checkpoint-500/          <- Intermediate checkpoint
    │   ├── adapter_model.bin
    │   └── trainer_state.json
    └── trainer_state.json       <- Final state
    '''
    '''
    | Task              | Dataset Columns       | Example                            |
    | ----------------- | --------------------- | ---------------------------------- |
    | `text-generation` | `"text"`              | `"The moon is made of cheese."`    |
    | `summarization`   | `"input"`, `"output"` | `"Long doc..." → "Short summary"`  |
    | `qa`              | `"input"`, `"output"` | `"question: X context: Y"` → `"A"` |
    '''
    '''
    Recommanded models for question-answering task
    | Model Name                    | Size  | Type                 | Pros                                    | Load with               |
    | ----------------------------- | ----- | -------------------- | ----------------------------------------| ----------------------- |
    | `google/flan-t5-small`        | \~80M | T5 (encoder-decoder) | Very fast, multilingual, well-supported | `AutoModelForSeq2SeqLM` |
    | `t5-small`                    | \~60M | T5                   | Vanilla baseline, stable                | `AutoModelForSeq2SeqLM` |
    | `sshleifer/tiny-mbart`        | \~25M | mBART                | Good for experimentation, very tiny     | `AutoModelForSeq2SeqLM` |
    | `philschmid/bart-tiny-random` | \~25M | BART                 | Super small for testing                 | `AutoModelForSeq2SeqLM` |

    '''
    def finetune(self,
        finetune_base_model_name,
        finetune_trained_model_path,
        finetune_dataset_name,
        finetune_task_type="text-generation",
        finetune_num_train_epochs=3,
        finetune_per_device_train_batch_size=4,
        finetune_use_4bit=False,
        finetune_lora_r=8,
        finetune_lora_alpha=16,
        finetune_lora_dropout=0.1,
        finetune_fp16=False,
        finetune_bf16=False,
        finetune_max_seq_length=512,
        finetune_save_steps=0,
        finetune_save_strategy='no',
        finetune_use_cpu=False,
        finetune_bnb_4bit_quant_type="nf4",
        finetune_bnb_4bit_compute_dtype=torch.float32,
        finetune_use_nested_quant=False,
        finetune_per_device_eval_batch_size=4,
        finetune_gradient_accumulation_steps=1,
        finetune_learning_rate=5e-5,
        finetune_weight_decay=0.0,
        finetune_max_grad_norm=1.0,
        finetune_max_steps=-1,
        finetune_logging_steps=50,
        finetune_warmup_ratio=0.03,
        finetune_lr_scheduler_type="linear",
        finetune_group_by_length=True,
        finetune_packing=False):

        self._finetune_logger.info(f"Starting fine-tuning: model={finetune_base_model_name}, task={finetune_task_type}")

        # --- Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(finetune_base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        tokenizer.padding_side = "right"

        # --- Dataset ---
        dataset = self.load_finetune_dataset(finetune_dataset_name)
        self._finetune_logger.info("Dataset loaded.")

        config = AutoConfig.from_pretrained(finetune_base_model_name)
        is_encoder_decoder = getattr(config, "is_encoder_decoder", False)

        if finetune_task_type == "text-generation":
            def preprocess(example):
                return tokenizer(example["text"], truncation=True, padding="max_length", max_length=finetune_max_seq_length)
            
            dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
        elif finetune_task_type == "qa":
            if is_encoder_decoder:
                # Generative QA (T5-style): expects input/output
                required_keys = {"input", "output"}
                if not required_keys.issubset(dataset.column_names):
                    raise ValueError(f"Dataset for encoder-decoder QA must contain columns: {required_keys}")

                def preprocess(example):
                    model_inputs = tokenizer(
                        example["input"], 
                        truncation=True, 
                        padding="max_length", 
                        max_length=finetune_max_seq_length
                    )
                    labels = tokenizer(
                        example["output"], 
                        truncation=True, 
                        padding="max_length", 
                        max_length=finetune_max_seq_length
                    )
                    model_inputs["labels"] = labels["input_ids"]
                    return model_inputs
                
                dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
            else:
                # Extractive QA (BERT-style): expects question/context/answer
                required_keys = {"question", "context", "answer"}
                if not required_keys.issubset(dataset.column_names):
                    raise ValueError(f"Dataset for extractive QA must contain columns: {required_keys}")
                
                def preprocess(example):
                    question = example["question"]
                    context = example["context"]
                    answer = example["answer"]

                    inputs = tokenizer(
                        question,
                        context,
                        truncation=True,
                        padding="max_length",
                        max_length=finetune_max_seq_length,
                        return_offsets_mapping=True
                    )

                    offset_mapping = inputs.pop("offset_mapping")
                    start_char = context.find(answer)
                    end_char = start_char + len(answer)

                    start_token, end_token = 0, 0
                    for idx, (start, end) in enumerate(offset_mapping):
                        if start <= start_char < end:
                            start_token = idx
                        if start < end_char <= end:
                            end_token = idx

                    inputs["start_positions"] = start_token
                    inputs["end_positions"] = end_token
                    return inputs

                dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
        elif finetune_task_type == "summarization":
            required_keys = {"input", "output"}
            if not required_keys.issubset(dataset.column_names):
                raise ValueError(f"Dataset for summarization must contain columns: {required_keys}")

            def preprocess(example):
                model_inputs = tokenizer(
                    example["input"], 
                    truncation=True, 
                    padding="max_length", 
                    max_length=finetune_max_seq_length
                )
                labels = tokenizer(
                    example["output"], 
                    truncation=True, 
                    padding="max_length", 
                    max_length=finetune_max_seq_length
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
        else:
            raise ValueError(f"Unsupported task: {finetune_task_type}")

        # --- Quantization ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=finetune_use_4bit,
            bnb_4bit_quant_type=finetune_bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=finetune_bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=finetune_use_nested_quant
        ) if finetune_use_4bit else None

        # --- Model and Trainer ---
        model = self.load_model_for_task(
            finetune_task_type,
            finetune_base_model_name,
            use_cpu=finetune_use_cpu,
            use_4bit=finetune_use_4bit
        )

        if finetune_task_type == "text-generation":
            if finetune_use_cpu or not finetune_use_4bit:
                bnb_config = None
            else:
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float32,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=False
                    )
                except ImportError:
                    raise ImportError(
                        "4-bit quantization requested, but bitsandbytes is not available or misconfigured. "
                        "Either install it properly or set `--finetune_use_4bit f`."
                    )

            try:
                target_modules = self.get_lora_target_modules(model)
                peft_config = LoraConfig(
                    target_modules=target_modules,
                    lora_alpha=finetune_lora_alpha,
                    lora_dropout=finetune_lora_dropout,
                    r=finetune_lora_r,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
            except ValueError as e:
                self._finetune_logger.warning(f"LoRA not applied: {e}")
                peft_config = None

            training_args = SFTConfig(
                output_dir=finetune_trained_model_path,
                num_train_epochs=finetune_num_train_epochs,
                per_device_train_batch_size=finetune_per_device_train_batch_size,
                per_device_eval_batch_size=finetune_per_device_eval_batch_size,
                gradient_accumulation_steps=finetune_gradient_accumulation_steps,
                learning_rate=finetune_learning_rate,
                weight_decay=finetune_weight_decay,
                fp16=finetune_fp16,
                bf16=finetune_bf16,
                max_grad_norm=finetune_max_grad_norm,
                max_steps=finetune_max_steps,
                save_steps=finetune_save_steps,
                save_strategy=finetune_save_strategy,
                logging_steps=finetune_logging_steps,
                warmup_ratio=finetune_warmup_ratio,
                group_by_length=finetune_group_by_length,
                lr_scheduler_type=finetune_lr_scheduler_type,
                report_to='none',
                packing=finetune_packing,
                max_seq_length=finetune_max_seq_length
            )
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                args=training_args,
                peft_config=peft_config,
                processing_class=tokenizer
            )
        else:  # summarization or QA
            if finetune_task_type == "qa" and not model.config.is_encoder_decoder:
                # Extractive QA
                def compute_loss_for_qa(model, inputs, return_outputs=False):
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        start_positions=inputs["start_positions"],
                        end_positions=inputs["end_positions"],
                    )
                    loss = outputs.loss
                    return (loss, outputs) if return_outputs else loss

                training_args = TrainingArguments(
                    output_dir=finetune_trained_model_path,
                    num_train_epochs=finetune_num_train_epochs,
                    per_device_train_batch_size=finetune_per_device_train_batch_size,
                    per_device_eval_batch_size=finetune_per_device_eval_batch_size,
                    logging_steps=finetune_logging_steps,
                    save_steps=finetune_save_steps,
                    save_strategy=finetune_save_strategy,
                    learning_rate=finetune_learning_rate,
                    weight_decay=finetune_weight_decay,
                    fp16=finetune_fp16,
                    bf16=finetune_bf16,
                    warmup_ratio=finetune_warmup_ratio,
                    report_to='none',
                    remove_unused_columns=False,
                )

                trainer = QAWithCustomLossTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                    #compute_loss=compute_loss_for_qa,
                )
            else:
                training_args = Seq2SeqTrainingArguments(
                    output_dir=finetune_trained_model_path,
                    num_train_epochs=finetune_num_train_epochs,
                    per_device_train_batch_size=finetune_per_device_train_batch_size,
                    per_device_eval_batch_size=finetune_per_device_eval_batch_size,
                    gradient_accumulation_steps=finetune_gradient_accumulation_steps,
                    learning_rate=finetune_learning_rate,
                    weight_decay=finetune_weight_decay,
                    fp16=finetune_fp16,
                    bf16=finetune_bf16,
                    max_grad_norm=finetune_max_grad_norm,
                    max_steps=finetune_max_steps,
                    save_steps=finetune_save_steps,
                    save_strategy=finetune_save_strategy,
                    logging_steps=finetune_logging_steps,
                    warmup_ratio=finetune_warmup_ratio,
                    lr_scheduler_type=finetune_lr_scheduler_type,
                    report_to='none',
                    predict_with_generate=True,
                    remove_unused_columns=False,
                )

                trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                    data_collator=DataCollatorForSeq2Seq(tokenizer)
                )

        self._finetune_logger.info("Training started...")
        trainer.train()
        trainer.save_model(finetune_trained_model_path)
        tokenizer.save_pretrained(finetune_trained_model_path)
        self._finetune_logger.info(f"Training finished. Model saved to {finetune_trained_model_path}")

    
    def llm_generate(self, model, tokenizer=None, prompt=None, context='', task_type=None,
            max_new_tokens=128, temperature=0.7, generation_kwargs=None, question=None,
            finetune_sample=True, finetune_temperature=0.7, finetune_num_beams=1, finetune_top_k=50,
            finetune_top_p=0.95, finetune_repetition_penalty=1.2
    ):
        """
        Generate output from a language model for various tasks.

        Supports:
        - Custom session-trained models with `.generate()` and `.decode()`
        - HuggingFace models for text-generation, QA, summarization
        - Auto task-type detection for HF models
        """

        generation_kwargs = generation_kwargs or {}

        # --- Step 1: In-session custom model (Bigram, Transformer, etc.)
        if hasattr(model, "decode") and hasattr(model, "generate"):
            input_ids = tokenizer(prompt) if callable(tokenizer) else tokenizer.encode(prompt)
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids).unsqueeze(0)
            output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
            output_text = model.decode(output_ids[0].tolist())

            self._finetune_logger.info(f"Prompt/Input: {prompt}")
            self._finetune_logger.info(f"Model Output: {output_text}")
            return output_text

        # --- Step 2: If model is path string, load it from HF
        elif isinstance(model, str):
            model_id_or_path = model

            if tokenizer is None:
                tokenizer = self.load_tokenizer(model_id_or_path)

            # Task-specific handling
            if task_type == "qa":
                used_question = question or prompt
                device_id = 0 if torch.cuda.is_available() else -1
                model_config = AutoConfig.from_pretrained(model_id_or_path)

                if model_config.is_encoder_decoder:
                    # Step 2A: Generative QA (T5, BART, etc.)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_path)
                    prompt = f"question: {used_question} context: {context}"
                    pipeline_task = "text2text-generation"
                    input_text = prompt
                else:
                    # Step 2B: Extractive QA (BERT-style)
                    model = AutoModelForQuestionAnswering.from_pretrained(model_id_or_path)
                    pipeline_task = "question-answering"
                    input_text = {"question": used_question, "context": context}

                qa_pipeline = pipeline(pipeline_task, model=model, tokenizer=tokenizer, device=device_id)
                output = qa_pipeline(input_text)

                if pipeline_task == "text2text-generation":
                    output_text = output[0]["generated_text"]
                else:
                    output_text = output["answer"]

                self._finetune_logger.info(f"Question: {used_question}")
                self._finetune_logger.info(f"Context: {context}")
                self._finetune_logger.info(f"Model Output: {output_text}")
                return output_text

            elif task_type == "summarization":
                # Step 2C: Summarization (Seq2Seq)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_path)
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                )
                input_len = len(tokenizer(prompt)["input_ids"])
                generation_kwargs.setdefault("max_length", max(30, int(input_len * 0.7)))
                generation_kwargs.setdefault("min_length", max(10, int(input_len * 0.3)))
                generation_kwargs.setdefault("do_sample", finetune_sample)
                generation_kwargs.setdefault("temperature", finetune_temperature)
                generation_kwargs.setdefault("num_beams", finetune_num_beams)
                generation_kwargs.setdefault("top_k", finetune_top_k)
                generation_kwargs.setdefault("top_p", finetune_top_p)
                generation_kwargs.setdefault("repetition_penalty", finetune_repetition_penalty)
                generation_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
                generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

                output_text = summarizer(prompt, **generation_kwargs)[0]["summary_text"]

                self._finetune_logger.info(f"Article (Input): {prompt}")
                self._finetune_logger.info(f"Summary (Output): {output_text}")
                return output_text

            elif task_type == "text-generation":
                # Step 2D: Decoder-only model (GPT, TinyLlama, etc.)
                model = AutoModelForCausalLM.from_pretrained(model_id_or_path)

                # Final fallback (HF model already loaded, task: text-generation)
                if tokenizer is None:
                    raise ValueError("Tokenizer must be provided for non-string models")

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(model.device if hasattr(model, "device") else "cpu")
                attention_mask = inputs["attention_mask"].to(model.device if hasattr(model, "device") else "cpu")

                generation_kwargs.setdefault("max_new_tokens", max_new_tokens)
                generation_kwargs.setdefault("temperature", finetune_temperature)
                generation_kwargs.setdefault("do_sample", finetune_sample)
                generation_kwargs.setdefault("num_beams", finetune_num_beams)
                generation_kwargs.setdefault("top_k", finetune_top_k)
                generation_kwargs.setdefault("top_p", finetune_top_p)
                generation_kwargs.setdefault("repetition_penalty", finetune_repetition_penalty)
                generation_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
                generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs
                    )

                output_text = tokenizer.decode(output[0], skip_special_tokens=True)

                self._finetune_logger.info(f"Prompt/Input: {prompt}")
                self._finetune_logger.info(f"Model Output: {output_text}")
                return output_text
        
        # --- Step 3: If none of the above matched
        else:
            raise ValueError(f"[llm_generate] Unsupported model/task combination: {type(model)}, task_type={task_type}")
    
    def generate(self, finetune_trained_model_path=None, finetune_prompt=None, finetune_context=None, 
            finetune_max_new_tokens=None, finetune_task_type=None, finetune_sample=None, finetune_temperature=None, 
            finetune_num_beams=None, finetune_top_k=None, finetune_top_p=None, finetune_repetition_penalty=None):
        output_text = self.llm_generate(
            model=finetune_trained_model_path,
            prompt=finetune_prompt,
            context=finetune_context,
            task_type=finetune_task_type,
            max_new_tokens=finetune_max_new_tokens,
            finetune_sample=finetune_sample,
            finetune_temperature=finetune_temperature,
            finetune_num_beams=finetune_num_beams,
            finetune_top_k=finetune_top_k, 
            finetune_top_p=finetune_top_p, 
            finetune_repetition_penalty=finetune_repetition_penalty
            #use_peft=True
        )

        #self._finetune_logger.info(f"Prompt: {finetune_prompt}")
        #self._finetune_logger.info(f"Model Output: {output_text}")
        
        self._finetune_logger.info(f"Saving generated text into {self.generated_text_file_name}")
        with open(self.generated_text_file_name, "w") as file:
            file.write(output_text)
        
        return output_text

    # Small models that can be used without a restriction:
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0, tiiuae/falcon-7b-instruct, facebook/opt-1.3b
    # Small model that requires HuggungFace access token : mistralai/Mistral-7B-Instruct-v0.1
    def smlp_finetune(self, finetune_base_model_name=None, finetune_trained_model_path=None, finetune_dataset_name=None, 
            finetune_prompt=None, finetune_context=None, finetune_task_type=None, finetune_max_new_tokens=None,
            finetune_train=None, finetune_eval=None, finetune_num_train_epochs=None, finetune_per_device_train_batch_size=None, 
            finetune_use_4bit=None, finetune_lora_r=None, finetune_lora_alpha=None, finetune_lora_dropout=None,
            finetune_fp16=None, finetune_bf16=None, finetune_max_seq_length=None, finetune_save_steps=None, 
            finetune_save_strategy=None, finetune_sample=None, finetune_temperature=None, finetune_num_beams=None,
            finetune_top_k=None, finetune_top_p=None, finetune_repetition_penalty=None):
        
        # perform sanity checks that model type 
        SUPPORTED_TASKS = {'text-generation', 'qa', 'summarization'}
        if finetune_task_type not in SUPPORTED_TASKS:
            raise ValueError(f"[ERROR] Invalid finetuning task: '{finetune_task_type}'. Must be one of {SUPPORTED_TASKS}")

        # Auto-detect architecture
        if finetune_base_model_name is None:
            assert not finetune_train, "finetune_base_model_name must be provided to perform finetuning"
        else:
            config = AutoConfig.from_pretrained(finetune_base_model_name)
            is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
            model_type = getattr(config, "model_type", "")

            if finetune_task_type == "text-generation" and is_encoder_decoder:
                raise ValueError("[ERROR] 'text-generation' requires a decoder-only model (e.g., GPT, LLaMA, TinyLlama).")

            if finetune_task_type == "summarization" and not is_encoder_decoder:
                raise ValueError("[ERROR] 'summarization' requires an encoder-decoder model (e.g., T5, BART).")

            if finetune_task_type == "qa":
                if is_encoder_decoder:
                    # Generative QA
                    if model_type not in ["t5", "bart", "mbart", "marian"]:
                        raise ValueError(f"[ERROR] Generative QA requires a supported encoder-decoder model, got: {model_type}")
                else:
                    # Extractive QA
                    if model_type not in ["bert", "distilbert", "roberta", "electra", "albert"]:
                        raise ValueError(f"[ERROR] Extractive QA requires a BERT-style encoder model, got: {model_type}")

        if finetune_train:
            self._finetune_logger.info("Running fine-tuning.")
            self.finetune(finetune_base_model_name=finetune_base_model_name, finetune_trained_model_path=finetune_trained_model_path, 
                finetune_dataset_name=finetune_dataset_name, finetune_task_type=finetune_task_type, 
                finetune_num_train_epochs=finetune_num_train_epochs, 
                finetune_per_device_train_batch_size=finetune_per_device_train_batch_size, finetune_use_4bit=finetune_use_4bit, 
                finetune_lora_r=finetune_lora_r, finetune_lora_alpha=finetune_lora_alpha, finetune_lora_dropout=finetune_lora_dropout,
                finetune_fp16=finetune_fp16, finetune_bf16=finetune_bf16, finetune_max_seq_length=finetune_max_seq_length, 
                finetune_save_steps=finetune_save_steps, finetune_save_strategy=finetune_save_strategy)
        
        if finetune_eval:
            self._finetune_logger.info("Running model generation (inference).")
            self.generate(finetune_trained_model_path=finetune_trained_model_path, finetune_prompt=finetune_prompt, 
                finetune_context=finetune_context, finetune_max_new_tokens=finetune_max_new_tokens, 
                finetune_task_type=finetune_task_type, finetune_sample=finetune_sample, 
                finetune_temperature=finetune_temperature, finetune_num_beams=finetune_num_beams,
                finetune_top_k=finetune_top_k, finetune_top_p=finetune_top_p, 
                finetune_repetition_penalty=finetune_repetition_penalty)
        