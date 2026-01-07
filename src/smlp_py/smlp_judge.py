import os
import json
import logging
import re
import torch
from typing import List, Dict, Any, Optional, Literal
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from openai import OpenAI
from dataclasses import dataclass



CONTEXT_GROUNDED_JUDGE_PROMPT = """
You are evaluating an LLM model output in {{evaluation_mode}} mode.

Context definition:
{{context_definition}}

Evaluation criteria:
- Groundedness: {{groundedness_definition}}
- Hallucination: {{hallucination_definition}}
"""

CONTEXT_GROUNDED_JUDGE_PROMPT_RAG = """
You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task is to evaluate the answer ONLY using the provided context.

Question:
{question}

Retrieved context:
{context}

Model answer:
{answer}

Evaluation rules:
1. The answer must be fully supported by the context.
2. Do NOT use external knowledge.
3. If any part of the answer is not supported by the context, mark hallucination = true.
4. If the answer is partially supported, grounded = false.
5. Score from 1 (poor) to 5 (excellent).

Respond ONLY in valid JSON with the following fields:
{{
  "grounded": boolean,
  "hallucination": boolean,
  "score": integer,
  "explanation": string
}}

Example valid response:
{{
  "grounded": true,
  "hallucination": false,
  "score": 5,
  "explanation": "The answer is fully supported by the provided context."
}}
"""

LLM_QUALITY_JUDGE_PROMPT = CONTEXT_GROUNDED_JUDGE_PROMPT = """
You are an impartial evaluator for an LLM system.

Evaluation mode: {evaluation_mode}

Task:
Evaluate the model answer based ONLY on the provided inputs.

Question:
{question}

Context:
{context}

Model answer:
{answer}

Evaluation rules:
1. If context is provided and non-empty:
   - The answer must be fully supported by the context.
   - Do NOT use external knowledge.
   - If any part of the answer is not supported by the context, set hallucination = true.
   - grounded = true ONLY if the answer is fully supported.
2. If context is empty or "N/A":
   - Evaluate internal consistency, plausibility, and relevance to the question.
   - hallucination should reflect logical inconsistency or fabricated facts.
   - grounded should be null.
3. Score from 1 (poor) to 5 (excellent).

Respond ONLY in valid JSON with the following fields:
{{
  "grounded": boolean | null,
  "hallucination": boolean,
  "score": integer,
  "explanation": string
}}
"""

LLM_QUALITY_JUDGE_PROMPT = LLM_QUALITY_JUDGE_PROMPT = """
You are an impartial evaluator for an LLM system.

Evaluation mode: {evaluation_mode}

Question: {question}

Context: {context}

Model answer: {answer}

EVALUATION CRITERIA:

1. GROUNDEDNESS (is the answer supported by context?):
   - TRUE: Every claim in the answer is directly supported by the context
   - FALSE: Some claims are unsupported, incomplete, vague, or use external knowledge
   - null: ONLY use if context is empty/N/A

2. HALLUCINATION (does the answer contain fabricated information?):
   - TRUE: The answer contains ANY fabricated facts, wrong names, or invented details
   - FALSE: All stated facts come from the context (even if incomplete)
   - Note: An incomplete answer is NOT a hallucination
   - Note: Missing information is NOT a hallucination

3. SCORING (1-5):
   - 5: Perfect answer - fully grounded, no hallucinations, complete
   - 4: Good answer - mostly grounded, minor incompleteness
   - 3: Adequate answer - partially grounded or incomplete, but no hallucinations
   - 2: Poor answer - mostly ungrounded OR contains hallucinations
   - 1: Failed answer - completely ungrounded AND/OR multiple hallucinations

IMPORTANT DISTINCTIONS:
- Incomplete ≠ Hallucinated (missing info is not fabrication)
- Vague ≠ Hallucinated (imprecise is not false)
- Wrong/fabricated facts = Hallucination (e.g., wrong names, invented details)

CRITICAL: Read the ENTIRE answer carefully before making your judgment.

Respond ONLY with valid JSON (no markdown, no explanations outside JSON):
{{
  "grounded": true/false/null,
  "hallucination": true/false,
  "score": 1-5,
  "explanation": "Detailed explanation of your evaluation"
}}
"""

@dataclass
class JudgeInput:
    """
    Canonical input to an LLM judge.
    All evaluation modes must map their data to this structure.
    --RAG: context = retrieved documents
    --Finetuning (QA / summarization): context = reference output or input+output
    --Scratvh training: context = original training line.
    
    As a result, the groundedness in jusge evaluation means:
    --RAG: Answer must be supported only by retrieved docs
    --Finetuning: Answer must match training target (or paraphrase it)
    --Scratch: Answer should resemble distribution of training text, not invent tokens
    """
    evaluation_mode: Literal["rag", "finetune", "scratch"]
    question: str
    answer: str
    context: Optional[str] = ""
    
    
class LlmBaseJudge(): #ABC
    """
    Abstract base class for all judge backends (HF, OpenAI, etc.).
    """

    def __init__(self):

        self._compute_device = "cpu" # TODO !!! avoid hard coded

        self._DEF_RAG_EVAL = None
        self._DEF_LLM_JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
        self._DEF_LLM_TRAIN_QUALITY_OUTPUT = "llm_training_quality.json"
        self._DEF_LLM_GEN_QUALITY_OUTPUT = "llm_generation_quality.json"
        self._DEF_LLM_QUALITY_METHOD = "judge"
        self._DEF_LLM_JUDGE_MAX_EXAMPLES = 20
        
        # Generation parameters
        self._DEF_LLM_JUDGE_DO_SAMPLE = False
        self._DEF_LLM_JUDGE_TEMPERATURE = 0.1
        self._DEF_LLM_JUDGE_TOP_P = 0.9
        self._DEF_LLM_JUDGE_REPETITION_PENALTY = 1.1
        self._DEF_LLM_JUDGE_MAX_NEW_TOKENS = 512
        self._DEF_LLM_JUDGE_MAX_INPUT_LENGTH = 4096

        # Evaluation parameters
        self._DEF_LLM_JUDGE_RETRY_ATTEMPTS = 2
        self._DEF_LLM_JUDGE_VALIDATE_CONSISTENCY = True
        self._DEF_LLM_JUDGE_STRIP_COT = True
        self._DEF_LLM_JUDGE_DEBUG_LOGGING = False
        
        # TODO !!! will need llm_model_type (e.g. "hf", "openai") -- currently can be inferred 
        # from llm_judge_model but if other llm_quality_methods will be used such as "metric"
        # "embedding", "rule" or "hybrid", then llm_model_type will be required -- ROUGE, BLEU 
        # for "metric", cosine similarity for "embedding", heuristic for "rule", and metric+LLM 
        # for "hybrid:.
        self.llm_quality_params_dict = {
            'llm_quality_method': {
                'abbr': 'llm_quality',
                'default': self._DEF_LLM_QUALITY_METHOD,
                'type': str,
                'help': (
                    'llm evaluation mode. '
                    '"judge" enables LLM-as-a-Judge evaluation, where a language model '
                    'scores generated answers for groundedness, hallucination, and overall quality '
                    'based only on the retrieved context. '
                    'If unset or different from "judge", no LLM-based llm evaluation is performed.'
                ),
            },
            'llm_judge_model': {
                'abbr': 'llm_judge_model',
                'default': self._DEF_LLM_JUDGE_MODEL,
                'type': str,
                'help': (
                    'Judge LLM model name used for llm evaluation. '
                    'Both HuggingFace and OpenAI models are supported.\n\n'
                    'HuggingFace (local) models:\n'
                    '  - Qwen/Qwen2.5-3B-Instruct: recommended minimum for reliable judging, '
                    '    good instruction following, suitable for CPU with adequate RAM.\n'
                    '  - Qwen/Qwen2.5-7B-Instruct: better reasoning and accuracy, '
                    '    recommended if memory allows (requires ~14GB RAM).\n'
                    '  - Qwen/Qwen2.5-1.5B-Instruct (~1.5B params): runs on CPU (slow but usable), '
                    '    but may miss details and produce less reliable judgments.\n'
                    '  - meta-llama/Llama-3.2-3B-Instruct: alternative 3B model with strong performance.\n'
                    '  - google/flan-t5-small / google/flan-t5-base: encoder-decoder models, '
                    '    very CPU-friendly and reliable for structured judging tasks.\n'
                    '  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: useful for regression tests and sanity checks, '
                    '    but not recommended for nuanced evaluation.\n\n'
                    'Models to avoid:\n'
                    '  - distilgpt2, gpt2 and other non-instruction-tuned base models.\n'
                    '  - Embedding-only models.\n'
                    '  These tend to hallucinate scores and explanations.\n\n'
                    'OpenAI models:\n'
                    '  - Examples: gpt-4o-mini, gpt-4o, gpt-4-turbo.\n'
                    '  - OpenAI judge models require a valid and funded OpenAI API account.\n'
                    '  - If no API key or insufficient quota is available, execution will fail.\n\n'
                    'Model backend is inferred automatically: model names starting with "gpt-" '
                    'use the OpenAI backend; all others use HuggingFace.'
                ),
            },
            "llm_judge_prompt": {
                "abbr": "llm_judge_prompt",
                "default": None,
                "type": str,
                "help": (
                    "Path to custom prompt template file for judge evaluation. "
                    "If not provided, uses the built-in prompt template. "
                    "Custom prompts must include placeholders: "
                    "{evaluation_mode}, {question}, {answer}, {context}. "
                    "The prompt should instruct the judge to return JSON with fields: "
                    "grounded, hallucination, score, explanation."
                )
            },
            "llm_judge_max_examples": {
                "abbr": "llm_judge_max_examples",
                "default": self._DEF_LLM_JUDGE_MAX_EXAMPLES,
                "type": int,
                "help": (
                    "Maximum number of samples to evaluate with judge. "
                    "Useful for limiting evaluation time on large datasets. "
                    "Set to None or 0 to evaluate all examples."
                )
            },
            "llm_judge_do_sample": {
                "abbr": "llm_judge_sample",
                "default": False,
                "type": bool,
                "help": (
                    "Enable sampling for judge generation. "
                    "False: use greedy decoding (deterministic, always picks highest probability token). "
                    "True: use sampling with temperature and top_p (adds controlled randomness). "
                    "When False, temperature and top_p are ignored. "
                    "Recommended: False for maximum reproducibility in evaluation."
                )
            },
            "llm_judge_temperature": {
                "abbr": "llm_judge_temp",
                "default": 0.1,
                "type": float,
                "help": (
                    "Temperature for judge model generation. "
                    "Only used when llm_judge_do_sample=True. "
                    "Lower values (0.0-0.3) produce more deterministic judgments. "
                    "Higher values (0.5-1.0) add variability but may reduce reliability. "
                    "Recommended: 0.1 for mostly consistent evaluation with slight randomness. "
                    "Note: Ignored when llm_judge_do_sample=False."
                )
            },
            "llm_judge_top_p": {
                "abbr": "llm_judge_top_p",
                "default": 0.9,
                "type": float,
                "help": (
                    "Top-p (nucleus sampling) parameter for judge generation. "
                    "Only used when llm_judge_do_sample=True. "
                    "Controls diversity by sampling from smallest set of tokens "
                    "whose cumulative probability exceeds p. "
                    "Recommended: 0.9 for balanced sampling. "
                    "Use 1.0 for maximum diversity or lower (0.7-0.85) for more focus. "
                    "Note: Ignored when llm_judge_do_sample=False."
                )
            },
            "llm_judge_repetition_penalty": {
                "abbr": "llm_judge_rep_penalty",
                "default": 1.1,
                "type": float,
                "help": (
                    "Repetition penalty for judge model generation. "
                    "Values > 1.0 discourage repetitive text in judge explanations. "
                    "Recommended: 1.1 to prevent redundant phrasing. "
                    "Use 1.0 to disable, or higher values (1.2-1.5) "
                    "if judge produces very repetitive outputs."
                )
            },
            "llm_judge_max_new_tokens": {
                "abbr": "llm_judge_max_tokens",
                "default": 512,
                "type": int,
                "help": (
                    "Maximum number of new tokens the judge can generate. "
                    "Higher values allow more detailed explanations but increase latency. "
                    "Recommended: 512 for comprehensive explanations. "
                    "Use 256 for shorter explanations, or 1024 if responses are truncated."
                )
            },
            "llm_judge_max_input_length": {
                "abbr": "llm_judge_max_input",
                "default": 4096,
                "type": int,
                "help": (
                    "Maximum input sequence length (in tokens) for judge model. "
                    "Inputs exceeding this length will be truncated. "
                    "Should be set based on judge model's context window: "
                    "Qwen 2.5: 4096-8192, Llama 3.2: 4096, "
                    "FLAN-T5: 512-1024, OpenAI: 8192-128000. "
                    "Increase if judge inputs are being truncated."
                )
            },
            "llm_judge_retry_attempts": {
                "abbr": "llm_judge_retries",
                "default": 2,
                "type": int,
                "help": (
                    "Number of retry attempts if judge fails to produce valid output. "
                    "Useful for handling occasional parsing failures or malformed JSON. "
                    "Recommended: 2 retries (3 total attempts). "
                    "Set to 0 to disable retries, or increase (3-5) "
                    "if judge frequently fails on first attempt."
                )
            },
            "llm_judge_validate_consistency": {
                "abbr": "llm_judge_validate",
                "default": True,
                "type": bool,
                "help": (
                    "Enable logical consistency validation for judge outputs. "
                    "Checks for contradictions like: grounded=True + hallucination=True, "
                    "or score=5 + grounded=False. "
                    "When inconsistencies are detected, adds confidence='low' flag "
                    "and warning to results. "
                    "Recommended: True for quality assurance."
                )
            },
            "llm_judge_strip_cot": {
                "abbr": "llm_judge_strip_cot",
                "default": True,
                "type": bool,
                "help": (
                    "Strip chain-of-thought reasoning from answers before judging. "
                    "Removes <think>...</think> tags and similar patterns. "
                    "Prevents judge confusion from speculative reasoning or meta-commentary. "
                    "Recommended: True to improve judge accuracy. "
                    "Set to False to evaluate the entire response including reasoning."
                )
            },
            "llm_judge_debug_logging": {
                "abbr": "llm_judge_debug",
                "default": False,
                "type": bool,
                "help": (
                    "Enable detailed debug logging for judge evaluation process. "
                    "Logs input sizes, prompt lengths, raw outputs, and parsing details. "
                    "Useful for diagnosing judge failures or understanding behavior. "
                    "Warning: Produces verbose output, recommended only for debugging."
                )
            },
            "llm_judge_load_in_8bit": {
                "abbr": "llm_judge_8bit",
                "default": False,
                "type": bool,
                "help": (
                    "Load judge model in 8-bit precision to reduce memory usage. "
                    "Requires bitsandbytes library. "
                    "Reduces memory by ~50% with minimal quality loss."
                )
            },
            "llm_judge_load_in_4bit": {
                "abbr": "llm_judge_4bit",
                "default": False,
                "type": bool,
                "help": (
                    "Load judge model in 4-bit precision for maximum memory savings. "
                    "Requires bitsandbytes library. "
                    "Reduces memory by ~75% with slight quality loss."
                )
            },
        }
    
    def set_report_file_prefix(self, report_file_prefix):
        """Set prefix for all output report files."""
        self.report_file_prefix = report_file_prefix

    def set_logger(self, logger):
        """Inject logger from SMLP runtime."""
        self._base_judge_logger = logger
    
    def _aggregate(self, results: list[dict]) -> dict:
        """
        Aggregate per-example judge results.
        A result is valid if at least one of: grounded, hallucination, or score is not None.
        """
        n_total = len(results)

        # Include results with at least one useful field. ALternative options are also reasonable, 
        # say when only the score field is required to be not None.
        valid = [
            r for r in results
            if r.get("grounded") is not None
            or r.get("hallucination") is not None
            or r.get("score") is not None
        ]

        if not valid:
            return {
                "summary": {
                    "num_examples": n_total,
                    "num_valid": 0,
                    "grounded_rate": None,
                    "hallucination_rate": None,
                    "avg_score": None,
                    "grounded_count": 0,
                    "hallucination_count": 0,
                    "score_count": 0,
                },
                "details": results,
            }

        # Calculate rates only from non-None values
        grounded_values = [r["grounded"] for r in valid if r.get("grounded") is not None]
        hallucin_values = [r["hallucination"] for r in valid if r.get("hallucination") is not None]
        score_values = [r["score"] for r in valid if r.get("score") is not None]

        aggr_result = {
            "num_examples": n_total,
            "num_valid": len(valid),
            # Rates (None if no data available)
            "grounded_rate": None if len(grounded_values) == 0 else sum(grounded_values) / len(grounded_values),
            "hallucination_rate": None if len(hallucin_values) == 0 else sum(hallucin_values) / len(hallucin_values),
            "avg_score": None if len(score_values) == 0 else sum(score_values) / len(score_values),
            # Counts (how many results contributed to each metric)
            "grounded_count": len(grounded_values),
            "hallucination_count": len(hallucin_values),
            "score_count": len(score_values),
        }

        output = {
            "summary": aggr_result,
            "details": results,
        }

        return output

    def _aggregate(self, results: list[dict]) -> dict:
        """
        Aggregate per-example judge results, handling None values gracefully.
        """
        n_total = len(results)

        # A result is valid if it has at least a score
        # (grounded and hallucination can be None for valid reasons)
        valid = [r for r in results if r.get("score") is not None]

        if not valid:
            return {
                "summary": {
                    "num_examples": n_total,
                    "num_valid": 0,
                    "grounded_rate": None,
                    "hallucination_rate": None,
                    "avg_score": None,
                },
                "details": results,
            }

        # Calculate rates only for non-None values
        grounded_values = [r["grounded"] for r in valid if r.get("grounded") is not None]
        hallucin_values = [r["hallucination"] for r in valid if r.get("hallucination") is not None]
        score_values = [r["score"] for r in valid if r.get("score") is not None]

        # Also track how many results had None values
        grounded_none_count = sum(1 for r in valid if r.get("grounded") is None)
        hallucin_none_count = sum(1 for r in valid if r.get("hallucination") is None)

        aggr_result = {
            "num_examples": n_total,
            "num_valid": len(valid),
            "grounded_rate": None if len(grounded_values) == 0 else sum(grounded_values) / len(grounded_values),
            "grounded_none_count": grounded_none_count,
            "hallucination_rate": None if len(hallucin_values) == 0 else sum(hallucin_values) / len(hallucin_values),
            "hallucination_none_count": hallucin_none_count,
            "avg_score": None if len(score_values) == 0 else sum(score_values) / len(score_values),
        }

        output = {
            "summary": aggr_result,
            "details": results,
        }

        return output
    def _strip_think(self, answer: str) -> str:
        """Remove chain-of-thought reasoning from answer before judging.
        This improves performance of LLM-as-Judge evaluation quality because:
        --The judge is asked to evaluate only the answer
        --But the answer includes:
          speculative reasoning
          references to earlier answers
          meta-commentary
        -- This confuses smaller judge models
        """
        # Strip <think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)

        # Also strip common CoT patterns
        cleaned = re.sub(r"Step-by-step explanation:.*?(?=Answer:|$)", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"Let me think.*?(?=\n\n|Answer:|$)", "", cleaned, flags=re.DOTALL)

        # Remove excessive whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract and parse JSON from judge output with better error handling."""

        # Try markdown code block first
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        brace = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(1))
            except json.JSONDecodeError:
                pass

        # Try to extract field by field as fallback
        try:
            grounded_match = re.search(r'"grounded"\s*:\s*(true|false|null)', text, re.IGNORECASE)
            halluc_match = re.search(r'"hallucination"\s*:\s*(true|false)', text, re.IGNORECASE)
            score_match = re.search(r'"score"\s*:\s*(\d+)', text)
            expl_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', text, re.DOTALL)

            if all([grounded_match, halluc_match, score_match, expl_match]):
                return {
                    "grounded": {"true": True, "false": False, "null": None}[grounded_match.group(1).lower()],
                    "hallucination": halluc_match.group(1).lower() == "true",
                    "score": int(score_match.group(1)),
                    "explanation": expl_match.group(1),
                }
        except Exception:
            pass

        raise ValueError(f"No valid JSON found in judge output. Output preview: {text[:200]}")
    
    def _validate_result(self, result: Dict[str, Any]) -> None:
        """Validate and enrich judge output, fixing None values intelligently."""
        required = {"grounded", "hallucination", "score", "explanation"}
        if not required.issubset(result):
            raise ValueError(f"Missing judge fields: {required - set(result)}")

        # Only fix clear logical impossibilities, not assumptions

        # If hallucinated, it CANNOT be grounded (this is logically certain)
        if result.get("hallucination") is True and result.get("grounded") is None:
            result["grounded"] = False

        # If fully grounded, it CANNOT be hallucinated (also logically certain)
        if result.get("grounded") is True and result.get("hallucination") is None:
            result["hallucination"] = False

        # WARNING: Do NOT assume grounded=False means hallucination=True
        # grounded=False could mean: incomplete, vague, missing info, or hallucinated
        # Only the judge can determine if fabrication occurred

        # Check for logical contradictions
        if result.get("grounded") is True and result.get("hallucination") is True:
            result["confidence"] = "low"
            result["warning"] = "Contradictory: marked as both grounded and hallucinated"
            # Try to fix the contradiction by trusting hallucination flag more
            # (hallucination is usually easier to detect than full groundedness)
            result["grounded"] = False

        # Check for score inconsistencies
        elif result.get("score") == 5 and result.get("grounded") is False:
            result["confidence"] = "low"
            result["warning"] = "Inconsistent: perfect score but not grounded"

        elif result.get("score") == 5 and result.get("hallucination") is True:
            result["confidence"] = "low"
            result["warning"] = "Inconsistent: perfect score but contains hallucinations"

        elif result.get("score") == 1 and result.get("grounded") is True and result.get("hallucination") is False:
            result["confidence"] = "low"
            result["warning"] = "Inconsistent: worst score but marked as grounded and accurate"

        else:
            result["confidence"] = "high"
    
    def _write_llm_quality_report(self, llm_quality_result, training_vs_gen:bool, 
            report_file_prefix:str, logger):
        report_file_suffix = self._DEF_LLM_TRAIN_QUALITY_OUTPUT if training_vs_gen else \
            self._DEF_LLM_GEN_QUALITY_OUTPUT
        llm_quality_output = "_".join([report_file_prefix, report_file_suffix])

        with open(llm_quality_output, "w") as f:
            json.dump(llm_quality_result, f, indent=2)

        training_vs_gen_str = 'training' if training_vs_gen else 'generation'
        logger.info(f"Judge results for ({training_vs_gen_str}): {llm_quality_result}")
        logger.info(f"LLM judge results written to {llm_quality_output}")
    
    def _infer_judge_type(self, model_name: str) -> str:
        """Infer judge backend type from model name."""
        if model_name.startswith(("gpt-", "o4-", "o3-", "chatgpt")):
            return "openai"
        return "hf"

    def _openai_api_key_present(self) -> bool:
        """Check whether OpenAI API key is available."""
        key = os.getenv("OPENAI_API_KEY")
        return key is not None and key.strip() != ""
    
    def _create_llm_judge(
        self, 
        model_name: str, 
        prompt_path: str,
        do_sample: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 512,
        max_input_length: int = 4096,
        debug_logging: bool = False,
        load_in_8bit = False,
        load_in_4bit = False
    ):
        """
        Factory for creating judge backend with all parameters.
        """
        judge_type = self._infer_judge_type(model_name)

        if judge_type == "openai":
            if not self._openai_api_key_present():
                raise RuntimeError(
                    "OpenAIJudge requires a funded OpenAI API account"
                )
            return OpenAIJudge(
                model_name=model_name,
                prompt_path=prompt_path,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                debug_logging=debug_logging,
            )

        return HFJudge(
            model_name=model_name,
            prompt_path=prompt_path,
            device=self._compute_device,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            max_input_length=max_input_length,
            debug_logging=debug_logging,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
    
    def _judge_with_retry(self, judge_llm, inp: JudgeInput,  max_retries: int) -> Dict[str, Any]:
        """Judge with retry logic for parsing failures."""

        for attempt in range(max_retries + 1):
            result = judge_llm.judge(inp)

            # Check if judgment succeeded (at least score should be present)
            if result.get("score") is not None:
                return result

            # Log retry
            logger = getattr(self, '_rag_judge_logger', None) or \
                     getattr(self, '_finetune_judge_logger', None) or \
                     getattr(self, '_llm_judge_logger', None)

            if logger and attempt < max_retries:
                logger.warning(
                    f"Judge attempt {attempt + 1} failed (score=None), retrying..."
                )

        return result  # Return last attempt even if failed
    
class HFJudge(LlmBaseJudge):
    """
    HuggingFace-based judge implementation.
    Supports both causal and seq2seq models.
    """

    def __init__(
        self, 
        model_name: str, 
        prompt_path: str = None, 
        device: str = "cpu",
        do_sample: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 512,
        max_input_length: int = 4096,
        debug_logging: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        super().__init__()

        self.device = device.lower() if isinstance(device, str) else device
        
        # Store generation parameters
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self.debug_logging = debug_logging

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        if getattr(config, "is_encoder_decoder", False):
            self.model_type = "seq2seq"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif config.model_type in {
            "gpt2", "llama", "qwen2", "mistral", "falcon", "phi"
        }:
            self.model_type = "causal"
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(
                f"Unsupported judge model type: "
                f"model_type={config.model_type}, "
                f"is_encoder_decoder={getattr(config, 'is_encoder_decoder', None)}"
            )

        self.model.to(self.device)
        self.model.eval()

        if prompt_path:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        else:
            self.prompt_template = LLM_QUALITY_JUDGE_PROMPT

    def generate(self, prompt: str) -> str:
        """
        Run text generation on the HF model.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        # Log input size if debug enabled
        if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
            self._base_judge_logger.debug(
                f"Judge input tokens: {inputs['input_ids'].shape[1]}/{self.max_input_length}"
            )

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
        }

        # Only add sampling parameters if do_sample is True
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            if self.model_type == "seq2seq":
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_kwargs
                )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                # For causal models, extract only the newly generated tokens
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Log output size if debug enabled
        if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
            self._base_judge_logger.debug(
                f"Judge output length: {len(generated_text)} chars"
            )

        return generated_text

    def judge(self, inp: JudgeInput) -> Dict[str, Any]:
        # Validate inputs
        if not inp.answer or not inp.answer.strip():
            return {
                "grounded": None,
                "hallucination": True,
                "score": 1,
                "explanation": "Empty answer provided",
            }

        # Log input sizes if debug enabled
        if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
            self._base_judge_logger.debug(
                f"Judge input sizes - Q: {len(inp.question)} chars, "
                f"A: {len(inp.answer)} chars, C: {len(inp.context or '')} chars"
            )

        # Strip CoT if enabled (from parent class)
        cleaned_answer = self._strip_think(inp.answer) if hasattr(self, '_strip_think') else inp.answer

        prompt = self.prompt_template.format(
            evaluation_mode=inp.evaluation_mode,
            question=inp.question,
            answer=cleaned_answer,
            context=inp.context or "N/A",
        )

        # Log prompt length if debug enabled
        if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
            self._base_judge_logger.debug(f"Judge prompt length: {len(prompt)} chars")

        try:
            raw = self.generate(prompt)

            # Log raw output if debug enabled
            if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
                self._base_judge_logger.debug(f"Judge raw output: {raw[:500]}")

            result = LlmBaseJudge._extract_json(raw)
            self._validate_result(result)
            return result

        except Exception as e:
            if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
                self._base_judge_logger.error(
                    f"Judge failed: {e}, raw output: {raw[:500] if 'raw' in locals() else 'N/A'}"
                )

            return {
                "grounded": None,
                "hallucination": None,
                "score": None,
                "explanation": f"Judge parsing failed: {e}",
                "raw_output": raw if 'raw' in locals() else None,
            }


class OpenAIJudge(LlmBaseJudge):
    """
    OpenAI-based judge implementation.
    Requires a funded OpenAI account.
    """

    def __init__(
        self, 
        model_name: str, 
        prompt_path: str = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        debug_logging: bool = False,
    ):
        super().__init__()
        self.client = OpenAI()
        self.model_name = model_name
        
        # Store generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.debug_logging = debug_logging

        if prompt_path:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        else:
            self.prompt_template = LLM_QUALITY_JUDGE_PROMPT

    def generate(self, prompt: str) -> str:
        """
        Call OpenAI ChatCompletion API.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluation judge. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

        return response.choices[0].message.content.strip()

    def judge(self, inp: JudgeInput) -> Dict[str, Any]:
        # Validate inputs
        if not inp.answer or not inp.answer.strip():
            return {
                "grounded": None,
                "hallucination": True,
                "score": 1,
                "explanation": "Empty answer provided",
            }

        # Log input sizes if debug enabled
        if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
            self._base_judge_logger.debug(
                f"Judge input sizes - Q: {len(inp.question)} chars, "
                f"A: {len(inp.answer)} chars, C: {len(inp.context or '')} chars"
            )

        # Strip CoT if enabled
        cleaned_answer = self._strip_think(inp.answer) if hasattr(self, '_strip_think') else inp.answer

        prompt = self.prompt_template.format(
            evaluation_mode=inp.evaluation_mode,
            question=inp.question,
            answer=cleaned_answer,
            context=inp.context or "N/A",
        )

        # Log prompt length if debug enabled
        if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
            self._base_judge_logger.debug(f"Judge prompt length: {len(prompt)} chars")

        try:
            raw = self.generate(prompt)

            # Log raw output if debug enabled
            if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
                self._base_judge_logger.debug(f"Judge raw output: {raw[:500]}")

            result = LlmBaseJudge._extract_json(raw)
            self._validate_result(result)
            return result

        except Exception as e:
            if self.debug_logging and hasattr(self, '_base_judge_logger') and self._base_judge_logger:
                self._base_judge_logger.error(
                    f"Judge failed: {e}, raw output: {raw[:500] if 'raw' in locals() else 'N/A'}"
                )

            return {
                "grounded": None,
                "hallucination": None,
                "score": None,
                "explanation": f"Judge parsing failed: {e}",
                "raw_output": raw if 'raw' in locals() else None,
            }

class SmlpRagJudge(LlmBaseJudge):
    """
    Dataset-level LLM-as-a-Judge for RAG evaluation.
    """

    def __init__(self):
        super().__init__()
        self._compute_device = "cpu" # TODO !!! avoid hard coded

    def set_report_file_prefix(self, report_file_prefix):
        """Set prefix for all output report files."""
        self.report_file_prefix = report_file_prefix

    def set_logger(self, logger):
        """Inject logger from SMLP runtime."""
        self._rag_judge_logger = logger

    def run(
        self, 
        rag_outputs,
        rag_retrieved,
        llm_quality_method: str,
        llm_judge_model: str,
        llm_judge_max_examples: int,
        llm_judge_prompt: str,
        llm_judge_do_sample: bool = False,
        llm_judge_temperature: float = 0.1,
        llm_judge_top_p: float = 0.9,
        llm_judge_repetition_penalty: float = 1.1,
        llm_judge_max_new_tokens: int = 512,
        llm_judge_max_input_length: int = 4096,
        llm_judge_retry_attempts: int = 2,
        llm_judge_validate_consistency: bool = True,
        llm_judge_strip_cot: bool = True,
        llm_judge_debug_logging: bool = False,
        llm_judge_load_in_8bit: bool = False,
        llm_judge_load_in_4bit: bool = False
    ):
        """Run LLM judge evaluation on RAG outputs."""
        if llm_quality_method != "judge":
            self._rag_judge_logger.info("LLM judge disabled")
            return None

        self._rag_judge_logger.info(
            f"Running LLM judge using model: {llm_judge_model}"
        )

        # Store strip_cot setting for use in judge
        self._strip_cot_enabled = llm_judge_strip_cot

        judge_llm = self._create_llm_judge(
            model_name=llm_judge_model,
            prompt_path=llm_judge_prompt,
            do_sample=llm_judge_do_sample,
            temperature=llm_judge_temperature,
            top_p=llm_judge_top_p,
            repetition_penalty=llm_judge_repetition_penalty,
            max_new_tokens=llm_judge_max_new_tokens,
            max_input_length=llm_judge_max_input_length,
            debug_logging=llm_judge_debug_logging,
        )

        # Inject logger into judge for debug logging
        if hasattr(judge_llm, '_base_judge_logger'):
            judge_llm._base_judge_logger = self._rag_judge_logger

        results = []
        for item in rag_outputs[:llm_judge_max_examples]:
            inp = JudgeInput(
                evaluation_mode='rag',
                question=item["question"],
                answer=item["answer"],
                context=rag_retrieved
            )

            if llm_judge_debug_logging:
                self._rag_judge_logger.info(f"LLM judge input:\n{inp}")

            # Use retry logic if enabled
            if llm_judge_retry_attempts > 0:
                res = self._judge_with_retry(
                    judge_llm, inp, llm_judge_retry_attempts
                )
            else:
                res = judge_llm.judge(inp)

            # Validate and fix inconsistencies
            if llm_judge_validate_consistency:
                self._validate_result(res)

            results.append(res)

        if llm_judge_debug_logging:
            self._rag_judge_logger.info(f"Raw results: {results}")

        aggr_results = self._aggregate(results)
        self._write_llm_quality_report(
            aggr_results,
            training_vs_gen=True,
            report_file_prefix=self.report_file_prefix,
            logger=self._rag_judge_logger,
        )

        return aggr_results


class SmlpFinetuneJudge(LlmBaseJudge):
    """
    LLM-as-a-Judge for finetuning and inference evaluation.
    Can be used:
      - after finetuning (training data)
      - after inference (new data)
    """

    def __init__(self, compute_device="cpu"):
        super().__init__()
        self._compute_device = 'cpu'
        self._logger = None

    def set_logger(self, logger):
        self._finetune_judge_logger = logger

    def set_report_file_prefix(self, report_file_prefix):
        """Set prefix for all output report files."""
        self.report_file_prefix = report_file_prefix
    
    
    def run(
        self, 
        dataset,
        task_type: str,
        judge_model: str,
        prompt_path: str = None,
        max_examples: int = None,
        train_vs_gen: bool = None,
        llm_judge_do_sample: bool = False,
        llm_judge_temperature: float = 0.1,
        llm_judge_top_p: float = 0.9,
        llm_judge_repetition_penalty: float = 1.1,
        llm_judge_max_new_tokens: int = 512,
        llm_judge_max_input_length: int = 4096,
        llm_judge_retry_attempts: int = 2,
        llm_judge_validate_consistency: bool = True,
        llm_judge_strip_cot: bool = True,
        llm_judge_debug_logging: bool = False,
        llm_judge_load_in_8bit: bool = False,
        llm_judge_load_in_4bit: bool = False
    ) -> dict:
        """
        Run judge over a finetuning dataset.
        """
        # Store strip_cot setting
        self._strip_cot_enabled = llm_judge_strip_cot

        judge_llm = self._create_llm_judge(
            model_name=judge_model,
            prompt_path=prompt_path,
            do_sample=llm_judge_do_sample,
            temperature=llm_judge_temperature,
            top_p=llm_judge_top_p,
            repetition_penalty=llm_judge_repetition_penalty,
            max_new_tokens=llm_judge_max_new_tokens,
            max_input_length=llm_judge_max_input_length,
            debug_logging=llm_judge_debug_logging,
        )

        # Inject logger
        if hasattr(judge_llm, '_base_judge_logger'):
            judge_llm._base_judge_logger = self._finetune_judge_logger

        results = []
        for idx, example in enumerate(dataset):
            if max_examples and idx >= max_examples:
                break

            try:
                inputs = self._example_to_judge_inputs(example, task_type)
                inp = JudgeInput(**inputs)

                if llm_judge_debug_logging:
                    self._finetune_judge_logger.info(f"Judge input {idx}: {inp}")

                # Use retry logic if enabled
                if llm_judge_retry_attempts > 0:
                    res = self._judge_with_retry(
                        judge_llm, inp, llm_judge_retry_attempts
                    )
                else:
                    res = judge_llm.judge(inp)

                # Validate and fix inconsistencies
                if llm_judge_validate_consistency:
                    self._validate_result(res)

            except Exception as e:
                self._finetune_judge_logger.error(f"Judge failed on example {idx}: {e}")
                res = {
                    "grounded": None,
                    "hallucination": None,
                    "score": None,
                    "explanation": f"Judge error: {str(e)}",
                }

            results.append(res)

        aggr_results = self._aggregate(results)
        self._write_llm_quality_report(
            aggr_results,
            train_vs_gen,
            self.report_file_prefix,
            self._finetune_judge_logger
        )

        return aggr_results
    
    def _example_to_judge_inputs(self, example: dict, task_type: str):
        """
        Convert finetuning example into (question, answer, context)
        expected by judge.
        """

        if task_type == "text-generation":
            return {
                "question": "Generate the correct continuation.",
                "answer": example.get("text", ""),
                "context": "",
            }

        elif task_type == "qa":
            if "input" in example and "output" in example:
                return {
                    "question": example["input"],
                    "answer": example["output"],
                    "context": "",
                }
            else:
                return {
                    "question": example["question"],
                    "answer": example["answer"],
                    "context": example["context"],
                }

        elif task_type == "summarization":
            return {
                "question": "Summarize the following text.",
                "answer": example["output"],
                "context": example["input"],
            }

        else:
            raise ValueError(f"Unsupported task_type for judge: {task_type}")

        
class SmlpScratchJudge(LlmBaseJudge):
    """
    Dataset-level LLM-as-a-Judge for LLM evaluation.
    """
    def __init__(self):
        super().__init__()
        self._compute_device = "cpu"

    def set_report_file_prefix(self, report_file_prefix):
        """Set prefix for all output report files."""
        self.report_file_prefix = report_file_prefix

    def set_logger(self, logger):
        """Inject logger from SMLP runtime."""
        self._llm_judge_logger = logger
    
    def run(
        self, 
        training_texts: List[str] | str,
        generated_texts: List[str] | str,
        llm_judge_model: str,
        llm_judge_max_examples: int,
        llm_judge_prompt: str,
        train_vs_gen: bool,
        llm_judge_do_sample: bool = False,
        llm_judge_temperature: float = 0.1,
        llm_judge_top_p: float = 0.9,
        llm_judge_repetition_penalty: float = 1.1,
        llm_judge_max_new_tokens: int = 512,
        llm_judge_max_input_length: int = 4096,
        llm_judge_retry_attempts: int = 2,
        llm_judge_validate_consistency: bool = True,
        llm_judge_strip_cot: bool = True,
        llm_judge_debug_logging: bool = False,
        llm_judge_load_in_8bit: bool = False,
        llm_judge_load_in_4bit: bool = False
    ) -> dict:
        """Run LLM judge evaluation on scratch training outputs."""

        if isinstance(training_texts, str):
            assert isinstance(generated_texts, str)
            training_texts = [training_texts]
            generated_texts = [generated_texts]

        # Store strip_cot setting
        self._strip_cot_enabled = llm_judge_strip_cot

        judge_llm = self._create_llm_judge(
            model_name=llm_judge_model,
            prompt_path=llm_judge_prompt,
            do_sample=llm_judge_do_sample,
            temperature=llm_judge_temperature,
            top_p=llm_judge_top_p,
            repetition_penalty=llm_judge_repetition_penalty,
            max_new_tokens=llm_judge_max_new_tokens,
            max_input_length=llm_judge_max_input_length,
            debug_logging=llm_judge_debug_logging,
            load_in_8bit=llm_judge_load_in_8bit,
            load_in_4bit=llm_judge_load_in_4bit
        )

        # Inject logger
        if hasattr(judge_llm, '_base_judge_logger'):
            judge_llm._base_judge_logger = self._llm_judge_logger

        results = []
        for idx, (src, gen) in enumerate(zip(training_texts, generated_texts)):
            if llm_judge_max_examples and idx >= llm_judge_max_examples:
                break

            inp = JudgeInput(
                evaluation_mode='scratch',
                question="Does the generated text stay faithful to the training text?",
                context=src,
                answer=gen,
            )

            if llm_judge_debug_logging:
                self._llm_judge_logger.info(f"Source {idx}: {src}")
                self._llm_judge_logger.info(f"Generated {idx}: {gen}")

            # Use retry logic if enabled
            if llm_judge_retry_attempts > 0:
                res = self._judge_with_retry(
                    judge_llm, inp, llm_judge_retry_attempts
                )
            else:
                res = judge_llm.judge(inp)

            if llm_judge_debug_logging:
                self._llm_judge_logger.info(f"Judge result {idx}: {res}")

            # Validate and fix inconsistencies
            if llm_judge_validate_consistency:
                self._validate_result(res)

            results.append(res)

        aggr_results = self._aggregate(results)
        self._write_llm_quality_report(
            aggr_results,
            train_vs_gen,
            self.report_file_prefix,
            self._llm_judge_logger
        )

        return aggr_results
""" TEMPORARY !!!!!!!!!!!!!!!!!!!!
6. What I might need next (only if you want improvements)

Not required now, but useful later:

Does generate() accept max_new_tokens?

Can train() expose training loss history?

Should judge results be saved to disk (JSON)?

If you want, next we can:

Add automatic aggregation (mean groundedness / hallucination rate)

Add CSV / JSON export

Align scratch judge prompts with finetune & RAG prompts

You’re very close to a fully unified evaluation framework in SMLP.
"""