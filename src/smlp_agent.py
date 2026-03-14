# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import json
import datetime
import subprocess

from smlp_py.smlp_flows import SmlpFlows
from smlp_py.smlp_utils import (str_to_bool, str_to_float_list, str_to_int_list, str_to_str_list, \
    str_to_str_list_list, str_to_int_tuple, str_to_str_list_pipe)


# This module immplements an SMLP agent. 
# Currently its task is to get text input from user that describes SMLP command in aan informal way,
# and generates a dictionary of SMLP command line option-value pairs; the letter is then converted
# into respective SMLP command and executed.
# Currently, the covertion of user description of an SMLP command into corresponding option-value 
# disctionary is performed by using a pre-trained model and the few-shot prompt method. The few-shot
# prompt has examples of user texts descriibing SLP command and the correponding option-vaue dictionary.
# User may use abbriviated or full names of SMLP CLI options, and abbriviated options are converted innt
# full option names before generating SMLP command. Sanity checks are also performed to make sure the
# inferred option-value dictionary has legitimate SMLP options and that the values are of the required type.
# Alternative options to genrate option-value dictionary from user provided informal description would be
# too train a dedicated model from scratch, or fine-tune an existing relevant LLM on SMLP related stuff
# (user manual, publications, dedicated manually or semi-manually created training dtaa with examples of 
# text to SMLP command conversion, etc.). Another option is to use RAG training based on the above SMLP
# related data. More detail on this approach can be found within comments at the bottom of this module.


'''
Your browser
   ↓
HTTP POST to /agent/text with {"query": "..."}
   ↓
FastAPI server (api_smlp_agent.py)
   ↓
agent.run_text_command()
   ↓
LLMInterpreter (returns SMLP args dict)
   ↓
SmlpExecutor runs run_smlp.py with parsed args
   ↓
Output sent back to you in browser

'''

'''
# TODO !!!!!! enable logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Generated plan: %s", plan)
'''

# --- Logging ---
def log_event(tag: str, payload: dict):
    with open("smlp_agent_log.jsonl", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.datetime.now().isoformat(),
            "tag": tag,
            "data": payload
        }) + "\n")

import requests

def check_ollama_alive(host="http://localhost:11434") -> bool:
    try:
        r = requests.get(host)
        return r.status_code in [200, 404]  # 404 is also OK for Ollama root endpoint
    except requests.ConnectionError:
        return False

# --- LLM Interpreter (Mock) ---
import os, json
import requests
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
#api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key = os.getenv("OPENAI_API_KEY")
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


'''
curl -X POST http://127.0.0.1:8000/agent/text \
  -H "Content-Type: application/json" \
  -d '{"query": "run rag on toy pdf and check if SMLP is published"}'

'''

class LLMInterpreter:
    def __init__(self, provider="ollama", model_name="mistral"):
        """
        provider: "ollama" or "openai"
        model_name: e.g. "mistral" for Ollama or "gpt-3.5-turbo" for OpenAI
        """
        self.provider = provider
        self.model_name = model_name
        self.few_shot_prompt = None
        
        print(f"""LLMInterpreter was initialized with provider {provider}, model {model_name}""", flush=True)
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "ollama":
            if not check_ollama_alive():
                raise RuntimeError("Ollama server is not running on port 11434. Run `ollama serve` in another terminal.")
            self.api_url = "http://localhost:11434/api/generate"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def load_prompt(self, prompt_str: str):
        print('LLMInterpreter load_prompt', prompt_str)
        self.few_shot_prompt = prompt_str
    
    
    def plan_from_text(self, query: str) -> dict:
        print('plan_from_text: self.few_shot_prompt', self.few_shot_prompt)
        if self.few_shot_prompt is not None:
            prompt = f"{self.few_shot_prompt.strip()}\n\nUser:\n\"{query.strip()}\""
        else:
            prompt = f"""You are an AI assistant for the SMLP system. Convert this user request into a task plan:
            ---
            {query}
            ---
            Output as a JSON dictionary of CLI-style options.
            """
        
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                content = response.choices[0].message.content
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    print("LLM returned invalid JSON", flush=True)
                    return {"error": "Invalid JSON from LLM"}
                return content
            except Exception as e:
                print("OpenAI error:", e, flush=True)
                return {"error": str(e)}
        elif self.provider == "ollama":
            '''
            # Now call your LLM here using the `prompt` string.
            # If you're using Ollama:
            import ollama
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

            # Then parse the JSON response from LLM output:
            return json.loads(response['message']['content'])
            '''
            try:
                res = requests.post(self.api_url, json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                })
                output = res.json().get("response", "")
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    print("LLM returned invalid JSON", flush=True)
                    return {"error": "Invalid JSON from LLM"}
                return output
            except Exception as e:
                print("Ollama error:", e, flush=True)
                return {"error": str(e)}
        else:
            return {"error": "Invalid provider"}

    def params_from_text_with_rag(self, query: str) -> dict:
        # Fallback or testing version
        rag_params_dict = {
            "analytics_mode": "rag",
            "rag_type": "lc",
            "rag_text": "../regr_smlp/texts/toy_smlp.pdf",
            "questions": "is there a published paper on SMLP?",
            "rag_train": True,
            "rag_eval": True,
            "rag_base_model_name": "deepseek-r1:1.5b",
            "index_backend": "cosine",
            "model_name": "dummy",
            "log_files_prefix": "try_fastapi"
        }
        rag_executer = SmlpExecutor()
        print('Running RAG to compute SMLP command line parameters', flush=True)
        result = rag_executer.run_spec(rag_params_dict)
        print('RAG completed with result', result, flush=True)
    

# --- Spec Generator ---
class SmlpSpecGenerator:
    def generate_from_plan(self, plan: dict, smlp_default_params_dict: dict) -> dict:
        print(">>> Inside generate_from_plan()", flush=True)
        print(">>> Received plan:", json.dumps(plan, indent=2), flush=True)

        # Start from default args
        full_spec = smlp_default_params_dict.copy()

        # Update with user plan (may override defaults)
        full_spec.update(plan)

        print(">>> Merged spec (defaults + plan):", json.dumps(full_spec, indent=2), flush=True)
        return full_spec

    def generate_from_dict(self, task_json: dict) -> dict:
        print(">>> Received structured task:", json.dumps(task_json, indent=2), flush=True)
        return task_json


# --- SMLP Executor ---
class SmlpExecutor:
    def run_spec(self, spec: dict) -> str:
        args = self._spec_to_args(spec)
        #print(">>> SmlpExecutor running spec with args:", args, flush=True)

        cmd = ["python", "-u", "./run_smlp.py"] + args  # -u = unbuffered output
        print("SmlpExecutor run_spec >>> Running command:", " ".join(cmd), flush=True)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("SmlpExecutor run_spec >>> Subprocess stdout:\n", result.stdout, flush=True)
            print("SmlpExecutor run_spec >>> Subprocess stderr:\n", result.stderr, flush=True)
        except subprocess.CalledProcessError as e:
            print("SmlpExecutor run_spec >>> SMLP failed:", e, flush=True)
            return {"SMLP failed": str(e)}
        
        return result.stdout or result.stderr
    
    def _spec_to_args(self, spec: dict) -> list:
        args = []
        for k, v in spec.items():
            args.append(f"--{k}")
            args.append(str(v))
        return args


# --- Main SMLP Agent ---
class SmlpAgent:
    def __init__(self, llm_interpreter=None, spec_generator=None, executor=None, doc_qa=None):
        self.llm = llm_interpreter or LLMInterpreter()
        self.spec_generator = spec_generator or SmlpSpecGenerator()
        self.executor = executor or SmlpExecutor()
        self.doc_qa = doc_qa
        
        # Load default few-shot prompt from file
        try: # before: "/prompts/smlp_fewshot.txt"
            with open("/home/zurabk/smlp/repo/smlp/regr_smlp/queries/few_shot_prompt.txt", "r") as f:
                prompt_str = f.read()
            self.llm.load_prompt(prompt_str)
        except Exception as e:
            print(f"Warning: could not load few-shot prompt: {e}")
        
        # We first to extract here the dictionary of default values of CLI parameters of SMLP.
        # These values will be used when running run_smlp(), for paramters whose values are not
        # specified by the user's query describing, in natural manguage, the SMLP task to perform.
        # need to pass "--model_name", "dummy", "--log_files_prefix", "dummy" because otherwise SmlpFlows.__init__() 
        # will fail due to missing values of arguments for which defauts are not defined.
        self.smlpInst = SmlpFlows(argv=["run_smlp.py", "--model_name", "dummy", "--log_files_prefix", "dummy"])
        #print('default args', self.smlpInst.args, flush=True)
        # self.smlp_default_params_dict represents deafaults, and values are always strings
        self.smlp_default_params_dict = self.smlpInst.args_dict.copy(); #print('params_dict', self.smlp_default_params_dict)
        # self.smlp_default_args_dict represents actual parameter values taken into account command line or config file values
        self.smlp_default_args_dict = vars(self.smlpInst.args).copy()
        
    
    def filter_valid_smlp_args(self, plan: dict, params_dict: dict):
        """
        Filters and validates an LLM-generated plan dictionary using SMLP parameter definitions.

        Args:
            plan (dict): Dictionary returned by LLM (param: value).
            params_dict (dict): Full SMLP CLI parameter schema (full_name: {'abbr': ..., 'type': ..., ...})

        Returns:
            Tuple[dict, list]: (filtered_plan, report)
              - filtered_plan: cleaned dictionary with valid parameter-value pairs (with full names).
              - report: list of tuples (key, value, reason) for each dropped key-value pair.
        """
        # Build full and abbrev lookup maps
        full_keys = set(params_dict.keys())
        abbrev_to_full = {v['abbr']: k for k, v in params_dict.items()}

        filtered_plan = {}
        report = []
            
        def check_type(val, expected_type):
            print(f"Checking type correctness for value {val} and expected_type {expected_type}")
            if isinstance(expected_type, type):
                is_valid = isinstance(val, expected_type)
                if not is_valid:
                    print('expected_type is a type but not the expected one')
                    
            elif callable(expected_type):  # e.g., str_to_str_list
                try:
                    expected_type(str(val))  # test if conversion is possible
                    is_valid = True
                except Exception:
                    is_valid = False
                    if not is_valid:
                        print(f'expected_type is a callable but failed on val')
            else:
                is_valid = False
            return is_valid
        
        def check_option_values(option, value):
            if option == "analytics_mode":
                assert value in self.smlpInst.smlp_modes, f"Invalid mode {option} in LLM inferred command" 
        
        # TODO bool type sanity check does not work correctly    
        for key, val in plan.items():
            if key in full_keys:
                check_option_values(key, val)
                expected_type = params_dict[key]['type']
                is_valid_type = check_type(val, expected_type)
                if is_valid_type:
                    filtered_plan[key] = val
                else:
                    print('expected_type', expected_type)
                    print(f"Pair {key}, {val} dropped due to invalid value type")
                    report.append((key, val, "invalid type"))
            elif key in abbrev_to_full:
                full_key = abbrev_to_full[key]
                check_option_values(full_key, val)
                expected_type = params_dict[full_key]['type']
                is_valid_type = check_type(val, expected_type)
                if is_valid_type:
                    filtered_plan[full_key] = val
                else:
                    print('expected_type', expected_type)
                    print(f"Pair {key}, {val} dropped due to invalid value type")
                    report.append((key, val, "invalid type"))
            else:
                report.append((key, val, "unknown parameter"))

        return filtered_plan, report

    
    def run_text_command(self, user_input: str):
        print("SmlpAgent: run_text_command >>> Received user input:", user_input, flush=True)
        
        rag_vs_few_shot = 'few_shot' # TODO !!!! option 'rag' is not properly supported yet
        if rag_vs_few_shot == 'rag':
            print('SmlpAgent: run_text_command >>> Executing RAG to compute SMLP command line', flush=True)
            llm_params_dict = self.llm.params_from_text_with_rag(user_input)
        elif rag_vs_few_shot == 'few_shot':
            llm_params_dict = self.llm.plan_from_text(user_input)        
        print("SmlpAgent: run_text_command >>> llm_params_dict:\n", json.dumps(llm_params_dict, indent=2), flush=True)

        # clean-up the llm_params_dict parameter:vallue dictionary for SMLP -- keys should be SMLP parameters
        # We therefore drop key-value pairs with incorrect keys 
        
        filtered_params_dict, dropped = self.filter_valid_smlp_args(llm_params_dict, self.smlp_default_params_dict)
        print("SmlpAgent: run_text_command >>> Filtered params dict:", filtered_params_dict)
        print("SmlpAgent: run_text_command >>> Dropped key-value pairs:")
        for key, val, reason in dropped:
            print(f" - {key}: {val}  [{reason}]")

        #spec = self.spec_generator.generate_from_plan(filtered_params_dict, self.smlp_default_args_dict)
        spec = filtered_params_dict
        print("SmlpAgent: run_text_command >>> Spec:\n", json.dumps(spec, indent=2), flush=True)

        log_event("text_command", {
            "input": user_input,
            "spec": spec
        })

        result = self.executor.run_spec(spec)
        print("SmlpAgent: run_text_command >>> Execution result:", result, flush=True)

        return result

    def run_task_command(self, task_json: dict):
        spec = self.spec_generator.generate_from_dict(task_json)
        log_event("structured_task", {"spec": spec})
        result = self.executor.run_spec(spec)
        return result

    def answer_doc_question(self, query: str):
        if self.doc_qa:
            return self.doc_qa.answer_question(query)
        return "Documentation QA not available."

    def load_prompt_from_text(self, prompt_str: str):
        print('load_prompt_from_text', prompt_str)
        if hasattr(self.llm, "load_prompt"):
            self.llm.load_prompt(prompt_str)
            print(">>> Few-shot prompt successfully updated.")
        else:
            raise Exception("LLMInterpreter does not support dynamic prompt loading.")

    def run_chat_command(self, user_input: str) -> dict:
        '''
        This method should:
        --Take user input (natural language)
        --Convert it to CLI dictionary using few-shot prompt
        --Optionally convert to command
        --Execute the command and return output
        '''
        return self.run_text_command(user_input)
    """
        try:
            # Step 1: Generate CLI dictionary using LLM interpreter
            interpreter = LLMInterpreter()
            cli_dict = interpreter.interpret(user_input)

            # Step 2: Convert dictionary to CLI command string
            from smlp_command import dict_to_command
            command = dict_to_command(cli_dict)

            # Step 3: Execute the command
            from smlp_executor import run_command
            stdout, stderr = run_command(command)

            return {
                "cli_dict": cli_dict,
                "command": command,
                "stdout": stdout,
                "stderr": stderr
            }

        except Exception as e:
            return {
                "error": str(e)
            }
    """
# convert SMLP command string into dictionary of parameter-value pairs.
import shlex

def parse_command_to_dict(command_str):
    tokens = shlex.split(command_str)
    parsed = {}
    i = 0
    while i < len(tokens):
        if tokens[i].startswith('-'):
            key = tokens[i].lstrip('-')
            if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                parsed[key] = tokens[i + 1]
                i += 1
            else:
                parsed[key] = True  # flag
        i += 1
    return parsed



'''
RAG training example:

Command:
run_smlp.py -mode rag -rag_type lc -rag_text mypdf.pdf -questions "What is SMLP?"

Descriptions:
"Use LangChain-based RAG to answer the question from the document mypdf.pdf."
"Run SMLP in RAG mode and extract answers to 'What is SMLP?' using local document."
"Perform retrieval-augmented generation with the provided PDF using LangChain indexing."


{
  "query": "Use LangChain-based RAG to answer the question from the document.",
  "smlp_args": {
    "mode": "rag",
    "rag_type": "lc",
    "rag_text": "mypdf.pdf",
    "questions": "What is SMLP?"
  }
}


{"input": "Run RAG to search if SMLP is published, using DeepSeek model and cosine similarity.", "label": {"mode": "rag", "rag_type": "lc", "rag_base_model_name": "deepseek-r1:1.5b", "rag_eval": True, "rag_train": False, "rag_text": "/project/smlp/texts/toy_smlp.pdf", "questions": "is there a published paper on SMLP?", "index_backend": "cosine", "log_files_prefix": "Test241"}}
{"input": "Perform text classification on a CSV input file. Save results in ./ directory.", "label": {"mode": "train", "data": "../data/smlp_toy_basic_text.csv", "out_dir": "./", "pref": "Test230"}}
{"input": "Use FastText embeddings and analyze spam classification task.", "label": {"mode": "train", "data": "../data/smlp_spam_dataset.csv", "text_embedding": "fasttext", "out_dir": "./", "pref": "FastTextSpam", "task": "text_classification"}}
{"input": "Classify Shakespeare sentences with bag-of-words model and store logs under bow_logs.", "label": {"mode": "train", "data": "../data/smlp_shakespeare.csv", "text_embedding": "bow", "pref": "bow_logs", "out_dir": "./"}}
{"input": "Do RAG over a PDF, asking 'What is the benefit of symbolic machine learning?'. Use LangChain and FAISS.", "label": {"mode": "rag", "rag_type": "lc", "rag_text": "/project/smlp/texts/smlp_intro.pdf", "questions": "What is the benefit of symbolic machine learning?", "index_backend": "faiss", "rag_base_model_name": "mistral", "rag_eval": True, "rag_train": False, "log_files_prefix": "Test_FAISS"}}
{"input": "Train LLM on labeled news data, use Mistral model and save it under ./outputs/mistral_model", "label": {"mode": "train", "data": "../data/ag_news.csv", "rag_base_model_name": "mistral", "out_dir": "./outputs", "pref": "mistral_model"}}
{"input": "Run sentiment analysis using word2vec embeddings on movie reviews.", "label": {"mode": "train", "data": "../data/imdb_reviews.csv", "text_embedding": "word2vec", "task": "sentiment_analysis", "out_dir": "./", "pref": "sentiment_w2v"}}
{"input": "Use saved model to answer: 'Who are the authors of SMLP?', using cosine similarity RAG.", "label": {"mode": "rag", "rag_type": "lc", "rag_trained_model_path": "/models/smlp_rag_model", "questions": "Who are the authors of SMLP?", "index_backend": "cosine", "rag_eval": True, "rag_train": False, "log_files_prefix": "authors_check"}}

User: Run RAG to search if SMLP is published, using DeepSeek model and cosine similarity.
SMLP Args: {"mode": "rag", "rag_type": "lc", "rag_base_model_name": "deepseek-r1:1.5b", "rag_eval": true, "rag_train": false, "rag_text": "/project/smlp/texts/toy_smlp.pdf", "questions": "is there a published paper on SMLP?", "index_backend": "cosine", "log_files_prefix": "Test241"}

User: Perform text classification on a CSV input file. Save results in ./ directory.
SMLP Args: {"mode": "train", "data": "../data/smlp_toy_basic_text.csv", "out_dir": "./", "pref": "Test230"}

User: Classify Shakespeare sentences with bag-of-words model and store logs under bow_logs.
SMLP Args: {"mode": "train", "data": "../data/smlp_shakespeare.csv", "text_embedding": "bow", "pref": "bow_logs", "out_dir": "./"}

User: Do RAG over a PDF, asking 'What is the benefit of symbolic machine learning?'. Use LangChain and FAISS.
SMLP Args: {"mode": "rag", "rag_type": "lc", "rag_text": "/project/smlp/texts/smlp_intro.pdf", "questions": "What is the benefit of symbolic machine learning?", "index_backend": "faiss", "rag_base_model_name": "mistral", "rag_eval": true, "rag_train": false, "log_files_prefix": "Test_FAISS"}

few-shot prompts:

User: Run RAG to search if SMLP is published, using DeepSeek model and cosine similarity.
SMLP Args: {"mode": "rag", "rag_type": "lc", "rag_base_model_name": "deepseek-r1:1.5b", "rag_eval": true, "rag_train": false, "rag_text": "/project/smlp/texts/toy_smlp.pdf", "questions": "is there a published paper on SMLP?", "index_backend": "cosine", "log_files_prefix": "Test241"}

User: Perform text classification on a CSV input file. Save results in ./ directory.
SMLP Args: {"mode": "train", "data": "../data/smlp_toy_basic_text.csv", "out_dir": "./", "pref": "Test230"}

User: Classify Shakespeare sentences with bag-of-words model and store logs under bow_logs.
SMLP Args: {"mode": "train", "data": "../data/smlp_shakespeare.csv", "text_embedding": "bow", "pref": "bow_logs", "out_dir": "./"}

User: Do RAG over a PDF, asking 'What is the benefit of symbolic machine learning?'. Use LangChain and FAISS.
SMLP Args: {"mode": "rag", "rag_type": "lc", "rag_text": "/project/smlp/texts/smlp_intro.pdf", "questions": "What is the benefit of symbolic machine learning?", "index_backend": "faiss", "rag_base_model_name": "mistral", "rag_eval": true, "rag_train": false, "log_files_prefix": "Test_FAISS"}

User: Use saved model to answer: 'Who are the authors of SMLP?', using cosine similarity RAG.
SMLP Args: {"mode": "rag", "rag_type": "lc", "rag_trained_model_path": "/models/smlp_rag_model", "questions": "Who are the authors of SMLP?", "index_backend": "cosine", "rag_eval": true, "rag_train": false, "log_files_prefix": "authors_check"}


'''

'''
-out_dir ./ -pref Test58 -mode optimize -pareto f -opt_strategy lazy -resp y1,y2 -feat x,p1,p2 -model dt_sklearn -dt_sklearn_max_depth 15 -tree_encoding nested -compress_rules f  -spec ../specs/smlp_toy_num_resp_mult.spec -objv_names objv_y1,objv_y2 -objv_exprs "y1;y2" -epsilon 0.01 -delta_rel 0.01 -data_scaler none -save_model_config f -mrmr_pred 0 -plots f -pred_plots f -resp_plots f -seed 10 -log_time f 

../../src/run_smlp.py -data "../data/smlp_toy_basic.csv" -out_dir ./ -pref Test114 -mode optimize -pareto t -opt_strategy lazy -model dt_sklearn -dt_sklearn_max_depth 15 -tree_encoding nested -compress_rules f -mrmr_pred 0 -epsilon 0.05 -delta_rel 0.01 -save_model f -plots f -pred_plots f -resp_plots f -seed 10 -log_time f -spec ../specs/smlp_toy_basic.spec  




# few-shot prompt
You are an assistant for SMLP. Convert the user’s description into a JSON of CLI-style options dictionary.

Examples:
#1
Input: "Train and pareto optimize with DT model on data file "../regr_smlp/data/smlp_toy_num_resp_mult.csv" , responses y1,y2, features x1,x2; save model as model1". Use spec file "../regr_smlp/specs/smlp_toy_num_resp_mult.spec', output directory "../regr_smlp/code", prefix "test1"
Output:
{"data":"../regr_smlp/data/smlp_toy_num_resp_mult.csv","out_dir":"../regr_smlp/code","pref":"test1","mode":"optimize","pareto":true,"resp":"y1,y2","feat":"x1,x2","model":"dt_sklearn","save_model":true,"model_name":"model1"}


#2
Input: "Train and pareto optimize with DT model on data file "../regr_smlp/data/smlp_toy_basic.csv",.use spec file "../regr_smlp/specs/smlp_toy_basic.spec', output directory "../regr_smlp/code", prefix "test2, use lazy optimization strategy, and nested tree encoding"
Output:
{"data":"../regr_smlp/data/smlp_toy_basic.csv","out_dir":"../regr_smlp/code","pref":"test2","mode":"optimize","pareto":true,"model":"dt_sklearn","save_model":false, "tree_encoding":"nested"}


User:
"Train and then pareto optimize with DT model on data file "../regr_smlp/data/smlp_toy_num_resp_mult.csv" , responses y1,y2, features x1,x2; save model as model_test". Use spec file "../regr_smlp/specs/smlp_toy_num_resp_mult.spec', output directory "../regr_smlp/code", prefix "model_test1"




{
  "query": "You are an assistant for SMLP. Convert the user’s description into a JSON of CLI-style options dictionary.\n\nExamples:\n#1\nInput: \"Train and pareto optimize with DT model on data file \\\"../regr_smlp/data/smlp_toy_num_resp_mult.csv\\\" , responses y1,y2, features x1,x2; save model as model1\". Use spec file \\\"../regr_smlp/specs/smlp_toy_num_resp_mult.spec', output directory \\\"../regr_smlp/code\\\", prefix \\\"test1\\\"\nOutput:\n{\"data\":\"../regr_smlp/data/smlp_toy_num_resp_mult.csv\",\"out_dir\":\"../regr_smlp/code\",\"pref\":\"test1\",\"mode\":\"optimize\",\"pareto\":true,\"resp\":\"y1,y2\",\"feat\":\"x1,x2\",\"model\":\"dt_sklearn\",\"save_model\":true,\"model_name\":\"model1\"}\n\n#2\nInput: \"Train and pareto optimize with DT model on data file \\\"../regr_smlp/data/smlp_toy_basic.csv\\\",.use spec file \\\"../regr_smlp/specs/smlp_toy_basic.spec', output directory \\\"../regr_smlp/code\\\", prefix \\\"test2, use lazy optimization strategy, and nested tree encoding\"\nOutput:\n{\"data\":\"../regr_smlp/data/smlp_toy_basic.csv\",\"out_dir\":\"../regr_smlp/code\",\"pref\":\"test2\",\"mode\":\"optimize\",\"pareto\":true,\"model\":\"dt_sklearn\",\"save_model\":false, \"tree_encoding\":\"nested\"}\n\nUser:\n\"Train and then pareto optimize with DT model on data file \\\"../regr_smlp/data/smlp_toy_num_resp_mult.csv\\\" , responses y1,y2, features x1,x2; save model as model_test\". Use spec file \\\"../regr_smlp/specs/smlp_toy_num_resp_mult.spec', output directory \\\"../regr_smlp/code\\\", prefix \\\"model_test1\\\""
}



'''
'''
Suggested training data format to genrate command dictionaries for RAG:

[
  {
    "type": "command_example",
    "description": "Train and pareto optimize with DT model on data file '../regr_smlp/data/smlp_toy_num_resp_mult.csv', responses y1,y2, features x1,x2; save model as model1. Use the eager optimization strategy and the flat tree encoding. Use spec file '../regr_smlp/specs/smlp_toy_num_resp_mult.spec', output directory '../regr_smlp/code', prefix 'test1'.",
    "dictionary": {
      "data": "../regr_smlp/data/smlp_toy_num_resp_mult.csv",
      "spec": "../regr_smlp/specs/smlp_toy_num_resp_mult.spec",
      "opt_strategy": "eager",
      "tree_encoding": "flat",
      "epsilon": 0.05,
      "delta_abs": 0.01,
      "out_dir": "../regr_smlp/code",
      "pref": "test1",
      "mode": "optimize",
      "pareto": true,
      "resp": "y1,y2",
      "feat": "x1,x2",
      "model": "dt_sklearn",
      "save_model": true,
      "model_name": "model1"
    }
  }
]

The above list has one example for now, many will be needed. 
Store these in your RAG store (e.g., FAISS + LangChain, or Elasticsearch, or ELSER).
When the RAG retriever returns a match, you pass the .dictionary field directly into the prompt context, like:
#------------ prompt start
User input: "Train a DT model with lazy optimization on x1,x2 predicting y1,y2"

Relevant example:
{
  "description": "...",
  "dictionary": { ... }
}

Generate a CLI options dictionary for the user prompt based on similar example above.
#------------ prompt end

Additional Tips
Embed both the description and dictionary in the retriever so both semantics and structure contribute to similarity.

If using LangChain RAG: define custom Document format with metadata tags (mode, model_type, etc.).

Use JSON-aware decoding in generation model (jsonformer, strictyaml, etc.) to ensure well-formed output.

'''


