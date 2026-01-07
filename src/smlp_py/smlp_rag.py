# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import os
import warnings
import json
import gc
import psutil
import os
        
    
    
import traceback
import random
from pathlib import Path

from dotenv import load_dotenv
#from docling.document_converter import DocumentConverter
import pymupdf4llm
from pathlib import Path
import faiss
from elasticsearch import Elasticsearch

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from smlp_py.smlp_utils import str_to_bool, str_to_str_list


# HF RAG
from typing import Callable, Any, Union, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from transformers.models.rag.retrieval_rag import RagRetriever, CustomHFIndex
from transformers import (
    Seq2SeqTrainingArguments,
    RagTokenizer,
    #RagRetriever, not required due to from transformers.models.rag.retrieval_rag import RagRetriever, CustomHFIndex
    RagTokenForGeneration,
    RagSequenceForGeneration,
    RagModel,
    Seq2SeqTrainer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    RagConfig, 
    BartConfig, 
    DPRConfig,
    GenerationConfig
)

from smlp_py.smlp_judge import SmlpRagJudge

# determinizm / reproducibility
import torch
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# TODO !!!!!!!!!! issues:
# PDF parsing might need improvement -- say currently several paragraps in toy_smlp.pdf are collapsed as one passage


'''
Module Purpose

This script builds and runs a Retrieval-Augmented Generation (RAG) pipeline that:
-- Ingests and parses a PDF
-- Splits its content into structured chunks
-- Embeds those chunks and indexes them using FAISS
-- Retrieves the most relevant chunks given a question
-- Passes them into a local LLM (Ollama) to generate bullet-point answers grounded in the source.

It is a local, self-hosted RAG system using:
-- pymupdf4llm for PDF → Markdown conversion
-- LangChain for document handling, prompt chaining, and LLM calls
-- Ollama for local embeddings and language model inference
-- FAISS for efficient vector similarity search.

Technologies Used
| Component       | Technology                                 |
| --------------- | ------------------------------------------ |
| PDF Parsing     | `pymupdf4llm`                              |
| Text Splitting  | `LangChain.MarkdownHeaderTextSplitter`     |
| Embedding       | `OllamaEmbeddings` (`nomic-embed-text`)    |
| Vector Search   | `FAISS` (L2 index)                         |
| LLM             | `ChatOllama` (`deepseek-r1:1.5b`)          |
| Prompt Chaining | LangChain Runnables & `ChatPromptTemplate` |

Key Features
-- Fully local RAG system with no external API calls
-- Structured document parsing and intelligent chunking
-- Support for streaming responses
-- Modular design that can be reused or extended easily
'''

'''
In LangChain RAG (rag_type = 'lc'):

The prompt is relevant at generation / inference time.

There is no training of the underlying LLM or prompt — LangChain RAG is typically zero-shot, where the prompt guides the behavior during generation.

You’re not fine-tuning or training the Ollama model itself; you're feeding it context + a prompt at runtime.

In HuggingFace RAG (rag_type = 'hf'):

The model learns during training (SFT) how to combine retrieved docs + question → so prompt template is typically baked into the model architecture / input preprocessing.

But you can still apply generation-time templates for formatting or prepending inputs.

So:
-- In LangChain RAG, the prompt is mainly for generation/inference.
-- In HF RAG, it can matter during both training and generation (especially if prompt tokens are part of input encoding).
'''

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

import re
from typing import List, Literal
import numpy as np
#from datasets import Dataset, DatasetDict

from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset

from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence

import pymupdf4llm

from typing import List, Union
import pandas as pd
import json
from datasets import DatasetDict, Dataset
from langchain.schema import Document
import pymupdf4llm
import re

class BaseRagPreprocessor:
    def __init__(self, rag_pdf_mode="langchain"):
        self.rag_pdf_mode = rag_pdf_mode

    def load_texts_from_csv(self, csv_path: str, text_column: str = "text") -> List[str]:
        df = pd.read_csv(csv_path)
        return df[text_column].dropna().tolist()

    def load_texts_from_json(self, json_path: str, text_key: str = "text") -> List[str]:
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item[text_key] for item in data if text_key in item]
        else:
            raise ValueError("Expected JSON file to contain a list of dicts")

    def load_pdf_markdown(self, pdf_path: str) -> str:
        return pymupdf4llm.to_markdown(pdf_path)

    def split_markdown_langchain(self, markdown: str) -> List[str]:
        splitter = MarkdownHeaderTextSplitter(
            [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
            strip_headers=False
        )
        docs = splitter.split_text(markdown)
        return [doc.page_content for doc in docs]

    def split_markdown_headers(self, markdown: str) -> List[str]:
        header_levels = ["#", "##", "###"]
        header_pattern = "|".join(re.escape(level) for level in header_levels)
        pattern = rf"(?=^({header_pattern}) )"
        sections = re.split(pattern, markdown, flags=re.MULTILINE)
        chunks = []
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            body = sections[i + 1].strip() if i + 1 < len(sections) else ""
            chunks.append(f"{header}\n{body}")
        return chunks

    def split_markdown_paragraphs(self, markdown: str) -> List[str]:
        return [p.strip() for p in markdown.split("\n\n") if len(p.strip()) > 0]

    def load_texts_from_pdf(self, pdf_path: str) -> List[str]:
        markdown = self.load_pdf_markdown(pdf_path)
        if self.rag_pdf_mode == "langchain":
            return self.split_markdown_langchain(markdown)
        elif self.rag_pdf_mode == "headers":
            return self.split_markdown_headers(markdown)
        elif self.rag_pdf_mode == "paragraphs":
            return self.split_markdown_paragraphs(markdown)
        else:
            raise ValueError(f"Unsupported rag_pdf_mode: {self.rag_pdf_mode}")

    def get_passages(self, path: str) -> List[str]:
        if path.endswith(".csv"):
            return self.load_texts_from_csv(path)
        elif path.endswith(".json"):
            return self.load_texts_from_json(path)
        elif path.endswith(".pdf"):
            return self.load_texts_from_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def get_hf_dataset(self, passages: List[str], test_split=0.25, seed=42) -> DatasetDict:
        hf_dataset = Dataset.from_dict({"text": passages})
        return hf_dataset.train_test_split(test_size=test_split, seed=seed)

    def get_lc_documents(self, passages: List[str]) -> List[Document]:
        return [Document(page_content=p) for p in passages]
    

class BaseRAG:
    def __init__(self, overrides: dict=None):
        
        # Default values (unified)
        self._DEF_RAG_BASE_MODEL_NAME = None  # "facebook/rag-token-base" 
        self._DEF_RAG_TRAINED_MODEL_PATH = None #
        self._DEF_RAG_MODEL = None
        self._DEF_RAG_QUESTIONS = None
        self._DEF_RAG_TOP_K_PASSAGES = 3
        self._DEF_RAG_TEXT = None
        self._DEF_RAG_INDEX_BACKEND = "faiss"
        self._DEF_RAG_MAX_NEW_TOKENS = 64
        self._DEF_RAG_SAMPLE = False
        self._DEF_RAG_TRAIN = True
        self._DEF_RAG_EVAL = True
        self._DEF_RAG_PDF_MODE = 'langchain'
        
        # Shared argument schema (unified)
        self._base_rag_params = {
            'rag_base_model_name': {
                'abbr': 'rag_base_model_name', 'default': self._DEF_RAG_BASE_MODEL_NAME, 'type': str,
                'help': 'Path or name of the pretrained RAG model to load. When rag_type is "hg", '
                        'the model Must be compatible with HuggingFace RAG models (e.g., "facebook/rag-token-base"), '
                        'while when rag_type is "lc", the model must be compatible with local Ollama RAG models '
                        '(e.g., "deepseek-r1:1.5b", "llama2:7b", "mistral:7b", "phi3:3b").'
            },
            'rag_trained_model_path': {
                'abbr': 'rag_trained_model_path', 'default': self._DEF_RAG_TRAINED_MODEL_PATH, 'type': str,
                'help': '\nPath or name of the RAG-based trained model in local work area '
                        '(not HuggingFace or other RAG pre-trained base model.'
            },
            'rag_questions': {
                'abbr': 'rag_questions', 'default': self._DEF_RAG_QUESTIONS, 'type': str_to_str_list,
                'help': 'List of questions to be answered by the RAG system.'
            },
            'rag_top_k_passages': {
                'abbr': 'rag_top_k', 'default': self._DEF_RAG_TOP_K_PASSAGES, 'type': int,
                'help': 'Number of top relevant passages to retrieve.'
            },
            'rag_index_backend': {
                'abbr': 'rag_index_backend', 'default': self._DEF_RAG_INDEX_BACKEND, 'type': str,
                'help': (
                    'Retrieval index backend to use. '
                    'Options:\n'
                    '  - "faiss": Fast approximate nearest neighbor search (default and currently only supported).\n'
                    '  - "cosine": Cosine similarity, supported for HuggingFace based RAG (option "--rag_type hf".\n'
                    '  - "elastic": Planned support for Elasticsearch-based retrieval (not yet implemented).\n'
                )
            },
            'rag_max_new_tokens': {
                'abbr': 'rag_max_new_tokens', 'default': self._DEF_RAG_MAX_NEW_TOKENS, 'type': int,
                'help': '\nMaximum number of tokens to generate in the answer.'
            },
            'rag_sample': {
                'abbr': 'rag_sample', 'default': self._DEF_RAG_SAMPLE, 'type': str_to_bool,
                'help': '\nWhether to use sampling during generation (top-k/top-p).'
            },
            'rag_train': {
                'abbr': 'rag_train', 'default': self._DEF_RAG_TRAIN, 'type': str_to_bool,
                'help': '\nWhether to run RAG training. Set to False to skip RAG training.'
            },
            'rag_eval': {
                'abbr': 'rag_eval', 'default': self._DEF_RAG_EVAL, 'type': str_to_bool,
                'help': '\nWhether to run RAG evaluation on the validation set.'
            },
            'rag_pdf_mode': {
                'abbr': 'rag_pdf_mode', 'default': self._DEF_RAG_PDF_MODE, 'type': str,
                'help': (
                    'PDF processing mode for converting PDF documents into passages for retrieval.\n'
                    'Options:\n'
                    '  - "langchain": Uses LangChain PDF loaders (e.g., PyPDFLoader, PDFMinerLoader, UnstructuredPDFLoader).\n'
                    '    Suitable for PDFs with selectable text and simple to moderately complex layouts. Default option.\n'
                    '  - "huggingface": Assumes text was pre-extracted and is processed as HuggingFace datasets or token chunks.\n'
                    '    Suitable for token-level chunking and fine control over dataset preparation.\n'
                    '  - "unstructured": Uses Unstructured.io parser to extract semantically meaningful sections from complex layouts.\n'
                    '    Suitable for PDFs with headers, tables, lists, and multi-column structures.\n'
                    '  - "ocr": Uses optical character recognition (OCR) to extract text from scanned image PDFs with no text layer.\n'
                    '    Suitable for scanned documents or image-based PDFs.\n'
                    'Note: The choice of "rag_pdf_mode" impacts how passages are generated, which affects retrieval accuracy '
                    'and downstream model performance.'
                )
            }
        }

        # Final merged config
        self.cfg = {key: entry["default"] for key, entry in self._base_rag_params.items()}
        
        if overrides:
            self.cfg.update(overrides)
        
        # Set up preprocessor with the chosen rag_pdf_mode
        self.preprocessor = BaseRagPreprocessor(
            rag_pdf_mode=self._base_rag_params["rag_pdf_mode"]
        )

'''
LangChain RAG is flexible:
You can plug any LLM LangChain supports as the generator (model). For example:
-- Ollama (local endpoint LLMs) → e.g., ChatOllama, like you do now.
-- OpenAI API → ChatOpenAI
-- HuggingFace Hub models → HuggingFaceHub
-- Locally downloaded HF models (via Transformers) → HuggingFacePipeline wrapped around a local Transformers pipeline.
-- Llama.cpp / llama-cpp-python / GPT4All backends
-- Mistral, Cohere, Anthropic, Groq, etc.

Instead of using Ollama at runtime (via its server), you can:
Download a local Transformers model.
Wrap it with HuggingFacePipeline and integrate with LangChain.
Example:
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
lc_model = HuggingFacePipeline(pipeline=pipe)
End of example

'''
class LangChainRag(BaseRAG):
    def __init__(self, overrides: dict=None):
        # Initialize BaseRAG shared args
        super().__init__(overrides)
        self.preprocessor = BaseRagPreprocessor(rag_pdf_mode="langchain")
        
        # LangChain-specific defaults
        self._DEF_RAG_MODEL_BASE_URL = "http://localhost:11434"
        self._DEF_RAG_EMBEDDING_MODEL = "nomic-embed-text"
        self._DEF_RAG_HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        self._DEF_RAG_PROMPT_TYPE = "document_focused"
        
        # LangChain-specific prompt templates
        prompt_chain_of_thought = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Answer in bullet points. Make sure your answer is relevant to the 
            question and it is answered from the context only.
            
            Question: {question}
            Context: {context}
            Answer:"""
        
        prompt_document_focused = """You are an assistant for question-answering tasks.
            Use ONLY the retrieved context below to answer the question.
            DO NOT think aloud or explain your reasoning.
            DO NOT provide chain-of-thought or assumptions.
            If you don't know the answer based on the context, say "I don't know."
            Provide your answer concisely in bullet points. 
            Do not add any extra commentary or filler text.

            Question: {question}
            Context: {context}
            Answer:"""
        
        prompt_extractive = """Use ONLY the context below to extract an answer. 
            If not found, say 'Answer not found'.
            
            Question: {question}
            Context: {context}
            Answer:"""
        
        prompt_strict_qa = """Answer briefly and factually based only on context.
            
            Question: {question}
            Context: {context}
            Answer:"""
        
        prompt_summarizer = """Summarize the relevant content for the following question.
            
            Question: {question}
            Context: {context}
            Summary:"""
        
        self.prompt_templates = {
            "document_focused": prompt_document_focused,
            "chain_of_thought": prompt_chain_of_thought,
            "extractive": prompt_extractive,
            "strict_qa": prompt_strict_qa,
            "summarizer": prompt_summarizer
        }
        
        # LangChain-only parameters
        self._lc_rag_params = {
            'rag_base_url': {
                'abbr': 'rag_base_url', 'default': self._DEF_RAG_MODEL_BASE_URL, 'type': str,
                'help': 'Base URL for Ollama server.'
            },
            'rag_embedding_model': {
                'abbr': 'rag_embedding_model', 'default': self._DEF_RAG_EMBEDDING_MODEL, 'type': str,
                'help': '\nEmbedding model to use via Ollama.'
            },
            'rag_prompt_type': {
                'abbr': 'rag_prompt_type', 'default': self._DEF_RAG_PROMPT_TYPE, 'type': str,
                'help': (
                    'Prompt style to guide the model’s response formatting:\n'
                    '  - "chain_of_thought": Encourage step-by-step reasoning and explanation.\n'
                    '  - "document_focused": Focus strictly on the context, concise and precise (default).\n'
                    '  - "extractive": Extract answer directly or say "Answer not found".\n'
                    '  - "strict_qa": Brief, direct answers without elaboration.\n'
                    '  - "summarizer": Summarize key points related to the question.\n'
                )
            },
            'rag_temperature': {
                'abbr': 'rag_temp', 'default': 0.0, 'type': float,
                'help': 'Temperature for generation. 0.0 for deterministic output.'
            },
            'rag_top_p': {
                'abbr': 'rag_top_p', 'default': 1.0, 'type': float,
                'help': 'Top-p for nucleus sampling. Use 1.0 for deterministic.'
            },
            'rag_seed': {
                'abbr': 'rag_seed', 'default': 42, 'type': int,
                'help': 'Random seed for reproducible generation.'
            },
        }
        
        self._raglc_logger = None
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._raglc_logger = logger 
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
    
    def split_markdown(self, markdown: str):
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self._DEF_RAG_HEADERS_TO_SPLIT_ON, strip_headers=False)
        return splitter.split_text(markdown)

    def setup_vector_store(self, docs, rag_embedding_model, rag_base_url):
        embeddings = OllamaEmbeddings(model=rag_embedding_model, base_url=rag_base_url)
        dim = len(embeddings.embed_query("test"))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=faiss.IndexFlatL2(dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.add_documents(docs)
        return vector_store

    def create_rag_chain(self, retriever, rag_base_model_name, rag_base_url, rag_prompt_type, 
            temperature=0.0, top_p=1.0, seed=42):
        """Create RAG chain with deterministic generation settings."""
        self._raglc_logger.info(f"Using rag_base_model_name {rag_base_model_name} at {rag_base_url}")

        model = ChatOllama(
            model=rag_base_model_name,
            base_url=rag_base_url,
            temperature=temperature,  # 0.0 for deterministic
            top_p=top_p,             # 1.0 or very low for deterministic
            seed=seed,               # Fixed seed for reproducibility
            num_predict=256,         # Control max output length
        )

        prompt_template = ChatPromptTemplate.from_template(
            self.prompt_templates[rag_prompt_type]
        )

        return (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | model
            | StrOutputParser()
        )
    
    def format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])

    
    def save_lc_rag_artifacts(self, rag_base_model_name:str, rag_trained_model_path:str, rag_embedding_model:str, 
            vector_store, rag_prompt_type:str, rag_top_k_passages:int, rag_base_url:str):
        '''
        Saved/loaded mmmodel config parameters that directly affect the vector store and retrieval logic:
        * rag_embedding_model → You must use the same embedding model (e.g., nomic-embed-text with Ollama or 
          another embedding) that was used to generate the FAISS vectors: If you change this, the embeddings 
          won’t align, therefore retrieval will break.
        * rag_top_k_passages → Not required to match, but should be compatible. You can change it at generation time 
          (e.g., 3 vs. 5 passages), but it shouldn't exceed what FAISS can handle (no issue unless your index is tiny).
        * faiss_index structure → Not a param in config, but the index must match the saved embeddings → can't 
          switch to a totally different index without rebuilding.

        Saved/loaded mmmodel config parameters that are used at generation time and don’t affect the pre-built index:
        * rag_base_model_name (generator LLM) → You can change the generator (e.g., from deepseek-r1:1.5b to another 
          Ollama-supported model).
        * rag_base_url → You can point to a different Ollama server as long as it's serving a compatible model.
        * rag_prompt_type → You can use a different prompt template (e.g., switch from “chain-of-thought” to “document-focused”).
        '''
        # Save FAISS index
        faiss_index_path = os.path.join(rag_trained_model_path, "faiss_index")
        os.makedirs(faiss_index_path, exist_ok=True)
        self._raglc_logger.info(f"Saving FAISS index into file {faiss_index_path}")
        vector_store.save_local(faiss_index_path)

        config = {
            "rag_prompt_type": rag_prompt_type,
            "rag_retriever_top_k": rag_top_k_passages,
            "rag_embedding_model": rag_embedding_model,
            "rag_base_url": rag_base_url,
            "rag_base_model_name": rag_base_model_name,
        }
        
        config_path = os.path.join(rag_trained_model_path, "rag_config.json")
        self._raglc_logger.info(f"Saving model artifacts config into file {config_path}")
        self._raglc_logger.info(config)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        self._raglc_logger.info(f"Saved RAG artifacts to: {rag_trained_model_path}")

    def load_lc_rag_artifacts(self, rag_embedding_model, rag_trained_model_path, rag_base_url):
        '''
        Vector store (e.g., FAISS index) is already built -- saves time.
        Retriever doesn’t need to recompute embeddings or build index.
        Ollama model loads as normal → no change in inference speed.
        '''
        # Load FAISS index
        self._raglc_logger.info(f"Loading embeddings {rag_embedding_model} from {rag_base_url}")
        embeddings = OllamaEmbeddings(model=rag_embedding_model, base_url=rag_base_url)
        faiss_index_path = os.path.join(rag_trained_model_path, "faiss_index")
        #print('load_lc_rag_artifacts: faiss_index_path', faiss_index_path, 'embeddings=embedding_model', rag_embedding_model)
        #print('rag_base_url', rag_base_url)
        self._raglc_logger.info(f"Loading FAISS index from {faiss_index_path}")
        vector_store = FAISS.load_local(faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        #print('embeddings', embeddings); print('vector_store', vector_store)
        config_path = os.path.join(rag_trained_model_path, "rag_config.json")
        with open(config_path) as f:
            config = json.load(f)
        self._raglc_logger.info(f"Loaded model artifacts config {config}")
        return vector_store, config

    def validate_lc_rag_config(self, saved_config, runtime_config):
        """
        Validate critical LangChain RAG parameters between saved config and current runtime config.
        Raises error if critical mismatch is found.
        """
        #print('saved_config', saved_config); print('runtime_config', runtime_config)
        critical_keys = ["rag_embedding_model"]  # Can expand if needed

        for key in critical_keys:
            saved_val = saved_config.get(key)
            runtime_val = runtime_config.get(key)
            #print('saved_val', saved_val, 'runtime_val', runtime_val)
            if saved_val != runtime_val:
                raise ValueError(
                    f"[RAG CONFIG MISMATCH] Critical parameter '{key}' mismatch:\n"
                    f"  Saved: {saved_val}\n"
                    f"  Runtime: {runtime_val}\n"
                    f"Your loaded FAISS index is incompatible with this embedding model.\n"
                    f"Either rebuild the index or use the same embedding model."
                )

        # Non-critical: just warn
        non_critical_keys = ["rag_retriever_top_k", "rag_prompt_type", "rag_base_model_name", "rag_base_url"]
        for key in non_critical_keys:
            saved_val = saved_config.get(key)
            runtime_val = runtime_config.get(key)
            #print('saved_val', saved_val, 'runtime_val', runtime_val)
            if saved_val != runtime_val:
                self._raglc_logger.info(
                    f"[RAG CONFIG WARNING] Non-critical parameter '{key}' differs:\n"
                    f"  Saved: {saved_val}\n"
                    f"  Runtime: {runtime_val}\n"
                    f"This is generally safe, but review if behavior seems unexpected."
                )

    def run(self, rag_questions, rag_text=None, rag_base_model_name:str=None, rag_trained_model_path:str=None, 
            rag_top_k_passages=None, rag_base_url:str=None, rag_embedding_model=None, rag_prompt_type=None, 
            rag_train=True, rag_eval=True):
        if rag_train:
            self._raglc_logger.info("Starting RAG training")
            #print(f"Converting: {rag_text}")
            
            # Get passages from any supported file type using BaseRagPreprocessor
            self._raglc_logger.info('Computing passages from input text')
            passages = self.preprocessor.get_passages(rag_text)
            self._raglc_logger.info(f"Number of passages loaded: {len(passages)}")
            # Convert passages to LangChain Document objects
            docs = self.preprocessor.get_lc_documents(passages)
            #print('docs', len(docs), docs)
            # Fresh run: build index + save artifacts
            self._raglc_logger.info("Computing embedding and indexing")
            vector_store = self.setup_vector_store(docs, rag_embedding_model, rag_base_url)
            self._raglc_logger.info("Creating RAG chain for training")
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": rag_top_k_passages})
            rag_chain = self.create_rag_chain(retriever, rag_base_model_name, rag_base_url, rag_prompt_type)

            if rag_trained_model_path:
                self._raglc_logger.info("Saving RAG artifacts in directory: " + str(rag_trained_model_path))
                self.save_lc_rag_artifacts(rag_base_model_name, rag_trained_model_path, rag_embedding_model, 
                    vector_store, rag_prompt_type, rag_top_k_passages, rag_base_url)
        else:
            # Load pre-built index + config
            if not rag_trained_model_path:
                raise ValueError("rag_trained_model_path must be specified if rag_train is False")

            self._raglc_logger.info(f"Loading RAG artifacts from: {rag_trained_model_path}")
            vector_store, saved_config = self.load_lc_rag_artifacts(rag_embedding_model, rag_trained_model_path, rag_base_url)

            # Validate critical params
            runtime_config = {
                "rag_embedding_model": rag_embedding_model,
                "rag_retriever_top_k": rag_top_k_passages,
                "rag_prompt_type": rag_prompt_type,
                "rag_base_model_name": rag_base_model_name,
                "rag_base_url": rag_base_url
            }
            self.validate_lc_rag_config(saved_config, runtime_config)
            self._raglc_logger.info('Loaded configuration ' + str(runtime_config))
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": rag_top_k_passages})
            rag_chain = self.create_rag_chain(retriever, rag_base_model_name, rag_base_url, rag_prompt_type)

        if rag_eval:
            self._raglc_logger.info("Creating RAG chain for answering question(s)")
            rag_chain = self.create_rag_chain(retriever, rag_base_model_name, rag_base_url, rag_prompt_type)
            
            self._raglc_logger.info("Answering questions...\n")
            retrieved = []
            for q in rag_questions:
                self._raglc_logger.info(f"Q: {q}")
                top_retrieved = retriever.get_relevant_documents(q)
                q_retrieved = ""
                #print('top_retrieved type and length', type(top_retrieved), len(top_retrieved), 'top_k_passages', rag_top_k_passages)
                self._raglc_logger.info(f"Top {rag_top_k_passages} retrieved passages for this question")
                for i, doc in enumerate(top_retrieved, 1):
                    #print('doc.page_content', doc.page_content)
                    q_retrieved = f"[{i}] {doc.page_content}"
                    self._raglc_logger.info(q_retrieved)  # Show first 200 chars for brevity
                    retrieved.append(q_retrieved)
  
                answer = ""
                for chunk in rag_chain.stream(q):
                    #print(chunk, end="", flush=True)
                    answer += chunk

                self._raglc_logger.info(f"Answer: {answer}\n")

                return answer, retrieved
            
    def cleanup_memory(self):
        """
        Free up memory after LangChain RAG operations.
        """
        self._raglc_logger.info("Cleaning up LangChain RAG memory...")

        # LangChain models are typically lighter, but still clean up
        if hasattr(self, 'rag_chain'):
            del self.rag_chain
            self.rag_chain = None

        # Force garbage collection
        gc.collect()

        self._raglc_logger.info("Memory cleanup complete")

# Seq2SeqTrainer is designed for encoder-decoder models (which RAG uses under the hood: BERT + BART)
# Note: HuggingFace RagModel is calling CosineRetriever.__call__() method with a prefix keyword argument.
# There CosineRetriever needs to accept **kwargs as a parameter (e.g. prefix param is extracted from kwargs)
'''
--- RagSequenceForGeneration (rag_token=False)
“Sequence-level fusion” = Encode each retrieved passage separately, then let the model choose one during generation.
Only one document at a time is attended to during decoding.
The model runs multiple forward passes and fuses results at the output (via marginalization over documents).
Useful if retrieved passages are redundant or independently useful.
--- RagTokenForGeneration (rag_token=True)
“Token-level fusion” = All retrieved passages are encoded together in a long tensor.
The decoder attends to all retrieved documents at once (cross-attention).
Each passage is fused into a single “flattened” sequence of contexts.
Decoder has richer context but more complexity.
Example:
P1: SMLP is a tool.
P2: SMLP was developed by Franz, Zurab and konstantin.
RagSequenceForGeneration:
Runs generation per passage.
Chooses most likely answer based on one passage at a time.
RagTokenForGeneration:
Flattens [P1; P2] into a long sequence.
Lets the decoder use both passages together when generating.
'''
class CustomRAGTrainer(Seq2SeqTrainer):
    def __init__(self, *args, rag_top_k_passages=None, rag_batch_size=None, raghf_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag_top_k_passages = rag_top_k_passages
        self.rag_batch_size = rag_batch_size
        self._crt_logger = raghf_logger
        self.extra_kwargs = kwargs  # Save any additional kwargs if needed
        assert rag_top_k_passages is not None, "CustomRAGTrainer requires rag_top_k_passages to be psitive integer"
        assert rag_top_k_passages > 0, "CustomRAGTrainer requires rag_top_k_passages to be psitive integer"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        self._crt_logger.info('Computing loss: start')
                
        #print("[CHECK] input_ids shape:", inputs["input_ids"].shape, type(inputs["input_ids"]))
        #print("[CHECK] attention_mask shape:", inputs["attention_mask"].shape, type(inputs["attention_mask"]))
        #print("[CHECK] attention_mask dtype:", getattr(inputs["attention_mask"], "dtype", "N/A"))

        # Check for corruption
        if not isinstance(inputs["attention_mask"], torch.Tensor):
            raise TypeError("attention_mask was corrupted before retriever call!")

        if isinstance(inputs["attention_mask"], torch.Tensor):
            if inputs["attention_mask"].max() > 1 or inputs["attention_mask"].min() < 0:
                raise ValueError("attention_mask has invalid values BEFORE retriever call")

        # Merge all kwargs
        ### disabling dynamic override of rag_top_k_passages using kwargs: 
        #rag_top_k_passages = kwargs.get("rag_top_k_passages", self.rag_top_k_passages)
        batch_size = kwargs.get("batch_size", self.rag_batch_size)
        assert batch_size == self.rag_batch_size
        #print(f"compute_loss received kwargs: {kwargs}")
        #self._crt_logger.info(f"Using rag_top_k_passages = {self.rag_top_k_passages}, batch_size = {batch_size}")
        #num_items = int(kwargs.get("num_items_in_batch", 0))
        if "num_items_in_batch" in kwargs:
            # remove num_items_in_batch safely (we do not want it when calling model()
            self._crt_logger.warning('Dropping num_items_in_batch from kwargs')
            num_items_in_batch = kwargs.pop("num_items_in_batch", None) 
        #print('num_items_in_batch', kwargs.get("num_items_in_batch", 0))
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        
        #print("Before retriever call: input_ids.shape", inputs["input_ids"].shape)
        #print("Before retriever call: attention_mask.shape", inputs["attention_mask"].shape)

        # Debug assertions
        assert input_ids.size(0) == labels.size(0), "input_ids and labels batch size mismatch"
        #if batch_size is not None:
        #    assert batch_size == input_ids.size(0), f"Expected batch_size {batch_size}, got {input_ids.size(0)}"
        if batch_size is not None and input_ids.size(0) != batch_size:
            self._crt_logger.warning(f"Batch size mismatch: expected {batch_size}, got {input_ids.size(0)}")

        #print("Inside compute_loss, attention_mask dtype:", inputs["attention_mask"].dtype)
        #print("Inside compute_loss, attention_mask max:", inputs["attention_mask"].max())

        for bad_key in ("attention_mask", "input_ids"):
            if bad_key in kwargs:
                self._crt_logger.warning(f"[WARNING] Deleting leaked `{bad_key}` from kwargs")
                del kwargs[bad_key]

        #print("attention_mask dtype (compute_loss):", inputs["attention_mask"].dtype)
        #print("attention_mask unique (compute_loss):", torch.unique(inputs["attention_mask"]))
        #print("inputs.get(attention_mask) unique passed to model() call (compute_loss):\n", torch.unique(inputs.get("attention_mask")))
        
        if "attention_mask" in inputs and not torch.all((inputs["attention_mask"] == 0) | (inputs["attention_mask"] == 1)):
            raise ValueError("Corrupted attention_mask in compute_loss")

        # Forward pass
        # CustomRAGTrainer now passes a safe, cloned attention_mask to model(...)
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].clone().detach(),  # protect original
            #attention_mask=inputs.get("attention_mask"),
            decoder_input_ids=inputs.get("decoder_input_ids"),
            decoder_attention_mask=inputs.get("decoder_attention_mask"),
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )

        logits = outputs.logits  # (batch_size * rag_top_k_passages, seq_len, vocab_size)

        # Process logits if rag_top_k_passages > 1
        if self.rag_top_k_passages > 1:
            bsz, seq_len, vocab_size = logits.size()
            assert bsz % self.rag_top_k_passages == 0, f"logits batch size {bsz} not divisible by rag_top_k_passages={self.rag_top_k_passages}"
            new_bsz = bsz // self.rag_top_k_passages
            logits = logits.view(new_bsz, self.rag_top_k_passages, seq_len, vocab_size)
            logits = logits.mean(dim=1)

        # Shift for teacher forcing
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Truncate for safety
        seq_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :seq_len, :]
        shift_labels = shift_labels[:, :seq_len]

        # Compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        self._crt_logger.info('Computing loss: end')
        return (loss, outputs) if return_outputs else loss

    
'''
-- Subclassing from RagRetriever (best practice, avoids all patching).
-- Passing config, tokenizer, and index correctly to super().__init__().
-- Avoiding fallback to FAISS or wiki_dpr by setting index_name="custom" and passing a dummy CustomHFIndex.
-- Storing and loading passage embeddings and metadata manually.
-- Custom __call__() method that handles similarity and tokenization correctly.
-- Dummy dataset and index handling
-- .to_dict() is necessary because RagConfig expects sub-configs as dictionaries, not config objects.
'''
class CosineRetriever(RagRetriever):
    def __init__(self, passage_embeddings, question_encoder, tokenizer, passages, top_k_passages, max_input_length, raghf_logger):
        #from transformers import RagConfig, DPRConfig, BartConfig

        # .to_dict() is required because RagConfig expects sub-configs as dictionaries, not config objects.
        dummy_config = RagConfig(
            question_encoder=DPRConfig().to_dict(),
            generator=BartConfig().to_dict(),
            index_name="custom",
            passages_path="unused",
        )

        # Build dummy dataset to avoid fallback index construction
        dummy_dataset = HFDataset.from_dict({
            "title": [""] * len(passages),
            "text": passages,
            "embeddings": [np.zeros(passage_embeddings.shape[1], dtype=np.float32) for _ in passages],  # Required
        })
        
        # Add dummy FAISS index
        dummy_dataset.add_faiss_index(column="embeddings")
        
        # Create dummy dataset and dummy index
        dummy_index = CustomHFIndex(vector_size=passage_embeddings.shape[1], dataset=dummy_dataset)

        # Super init
        super().__init__(
            config=dummy_config,
            question_encoder_tokenizer=tokenizer,
            generator_tokenizer=tokenizer,
            index=dummy_index
        )

        # Store real data
        self.passage_embeddings = passage_embeddings
        self.question_encoder = question_encoder # TODOD !!!! is it used?
        self.tokenizer = tokenizer
        self.passages = passages
        self.n_docs = min(top_k_passages, len(passages))
        self.max_input_length = max_input_length
        self._cr_logger = raghf_logger
        #print(self.n_docs, len(passages), passages)
        #print("Default max_length for question_encoder tokenizer:", tokenizer.model_max_length)

    def __call__(self, input_ids, question_hidden_states, **kwargs):
        #print("[CosineRetriever] input_ids shape:", input_ids.shape if input_ids is not None else 'none')
        #print("[CosineRetriever] question_hidden_states shape:", 
        #      question_hidden_states.shape if question_hidden_states is not None else 'none')
        #print("[CosineRetriever] kwargs:", kwargs)

        if isinstance(question_hidden_states, np.ndarray):
            question_hidden_states = torch.tensor(question_hidden_states, dtype=torch.float32)

        norm_q = question_hidden_states / question_hidden_states.norm(dim=1, keepdim=True)
        norm_p = self.passage_embeddings / self.passage_embeddings.norm(dim=1, keepdim=True)
        sims = torch.matmul(norm_q, norm_p.T)
        top_docs = sims.topk(k=self.n_docs, dim=1)

        '''
        print("[DEBUG] Top passage indices:", top_docs.indices)
        for row_idx, row in enumerate(top_docs.indices):
            print(f"[DEBUG] Query {row_idx}:")
            for rank, passage_idx in enumerate(row.tolist()):
                print(f"  Rank {rank+1}: passage[{passage_idx}] = {self.passages[passage_idx]}")
        '''
        retrieved_passages = [
            [self.passages[idx] for idx in row.tolist()]
            for row in top_docs.indices
        ]
        flat_passages = [p for doc_list in retrieved_passages for p in doc_list]

        tokenized = self.tokenizer(
            flat_passages,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )

        batch_size, n_docs = top_docs.indices.shape
        retrieved_doc_embeds = self.passage_embeddings[top_docs.indices]

        return {
            "doc_scores": top_docs.values,
            "doc_ids": top_docs.indices,
            "retrieved_passages": retrieved_passages,
            "context_input_ids": tokenized["input_ids"],
            "context_attention_mask": tokenized["attention_mask"],
            "retrieved_doc_embeds": retrieved_doc_embeds,
        }

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        torch.save(self.passage_embeddings, os.path.join(save_directory, "passage_embeddings.pt"))
        with open(os.path.join(save_directory, "passages.txt"), "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(p.strip() + "\n")
        with open(os.path.join(save_directory, "cosine_retriever_config.json"), "w") as f:
            json.dump({"top_k_passages": self.n_docs}, f)

        self._cr_logger.info(f"CosineRetriever saved to directory {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory, question_encoder, tokenizer):
        passage_embeddings = torch.load(os.path.join(load_directory, "passage_embeddings.pt"))
        with open(os.path.join(load_directory, "passages.txt"), "r", encoding="utf-8") as f:
            passages = [line.strip() for line in f]
        with open(os.path.join(load_directory, "cosine_retriever_config.json"), "r") as f:
            config = json.load(f)
            
        self._cr_logger.info(f"CosineRetriever loaded from directory {save_directory}")
        return cls(
            passage_embeddings=passage_embeddings,
            question_encoder=question_encoder,
            tokenizer=tokenizer,
            passages=passages,
            top_k_passages=config["top_k_passages"],
        )

class HuggingFaceRag(BaseRAG):
    def __init__(self, overrides: dict = None):
        super().__init__(overrides)
        self.preprocessor = BaseRagPreprocessor(rag_pdf_mode="langchain")
        
        self._DEF_QUESTION_COLUMN = "title"
        self._DEF_CONTEXT_COLUMN = "text"
        self._DEF_INDEX_NAME = "custom"
        self._DEF_BATCH_SIZE = 2
        self._DEF_EPOCHS = 1
        self._DEF_MAX_INPUT_LENGTH = 256
        self._DEF_MAX_TARGET_LENGTH = 64
        self._DEF_MAX_NEW_TOKENS = 64
        self._DEF_RAG_TOKEN = True
        self._DEF_EVAL_STRATEGY = "epoch"
        self._DEF_SAVE_STEPS = 100
        self._DEF_LOGGING_STEPS = 10
        self._DEF_REPORT_TO = "none"
        self._DEF_RAG_TRUST_REMOTE_CODE = True # TODO !!!!
        self._DEF_LR = 2e-5
        self._DEF_WEIGHT_DECAY = 0.01
        self._DEF_SAVE_TOTAL_LIMIT = 2
        self._DEF_DEVICE = 'cpu'
        self._DEF_RAG_SEED = 42
        
        self._hf_rag_params = {
            'rag_trust_remote_code': {
                'abbr': 'rag_trust_remote_code', 'default': self._DEF_RAG_TRUST_REMOTE_CODE, 'type': str_to_bool,
                'help': 'Whether to allow loading and executing custom code from model remote repositories. '
                        'When set to False, only load models using predefined architectures in the '
                        'Transformers library (e.g., BertModel, RagTokenForGeneration); a custom RAG '
                        'model from an experimental repo (e.g., "username/my-custom-rag-model") would fail.'
            },
            'rag_question_column': {
                'abbr': 'rag_question_column', 'default': self._DEF_QUESTION_COLUMN, 'type': str,
                'help': '\nName of the dataset column that contains the question or query text.'
            },
            'rag_context_column': {
                'abbr': 'rag_context_column', 'default': self._DEF_CONTEXT_COLUMN, 'type': str,
                'help': '\nName of the dataset column that contains the answer or document context.'
            },
            #'index_name': { 'abbr': 'index_name', 'default': self._DEF_INDEX_NAME, 'type': str,
            #    'help': (
            #        'Name used to save/load the FAISS index and retriever files.\n'
            #        'Options:\n'
            #        '  - "custom": Recommended for all current use cases. Loads/saves the index from ./custom_index/.\n'
            #        '  - "exact": Reserved for future support of in-memory exact retrieval.\n'
            #        'Note: Using "exact" currently raises an error, as this mode is not implemented.'
            #    )
            #},
            'rag_batch_size': {
                'abbr': 'rag_batch', 'default': self._DEF_BATCH_SIZE, 'type': int,
                'help': '\nBatch size per device for training and evaluation.'
            },
            'rag_epochs': {
                'abbr': 'rag_epochs', 'default': self._DEF_EPOCHS, 'type': int,
                'help': '\nTotal number of training epochs to run.'
            },
            'rag_max_input_length': {
                'abbr': 'rag_max_input_length', 'default': self._DEF_MAX_INPUT_LENGTH, 'type': int,
                'help': (
                    'Controls how long input queries to the RAG retriever can be.'
                    'Maximum token length of the input sequence (i.e., the question for the encoder, '
                    'and optionally context during training); '
                    'relevant for both training and generation (default {}).'
                ).format(self._DEF_MAX_INPUT_LENGTH)
            },
            'rag_max_target_length': {
                'abbr': 'tag_max_target_length', 'default': self._DEF_MAX_TARGET_LENGTH, 'type': int,
                'help': (
                    'Maximum length of target labels (answers) during training for teacher-forcing; '
                    'relevant for training only, not generation (default {}).'
                ).format(self._DEF_MAX_TARGET_LENGTH)
            },
            'rag_max_new_tokens': {
                'abbr': 'max_new_tokens', 'default': self._DEF_MAX_NEW_TOKENS, 'type': int,
                'help': (
                    'Maximum number of tokens to generate during inference; '
                    'controls how many new tokens the model is allowed to output at generation time; '
                    'relevant for generation only, not training (default {}).'
                ).format(self._DEF_MAX_NEW_TOKENS)
            },
            'rag_token': {
                'abbr': 'rag_token', 'default': self._DEF_RAG_TOKEN, 'type': str_to_bool,
                'help': 'Whether to tokenize input samples during preprocessing.'
            },
            'rag_eval_strategy': {
                'abbr': 'rag_eval_strategy', 'default': self._DEF_EVAL_STRATEGY, 'type': str,
                'help': 'Evaluation frequency: "epoch" (every epoch), "steps" (every N steps), '
                        'or "no" (never evaluate).'
            },
            'rag_save_steps': {
                'abbr': 'rag_save_steps', 'default': self._DEF_SAVE_STEPS, 'type': int,
                'help': 'How often to save model checkpoints during training (in steps).'
            },
            'rag_logging_steps': {
                'abbr': 'rag_logging_steps', 'default': self._DEF_LOGGING_STEPS, 'type': int,
                'help': 'How often to log training metrics (in steps).'
            },
            'rag_report_to': {
                'abbr': 'rag_report_to', 'default': self._DEF_REPORT_TO, 'type': str,
                'help': 'Logging/reporting backend: "none", "tensorboard", or "wandb".'
            },
            'rag_lr': {
                'abbr': 'rag_lr', 'default': self._DEF_LR, 'type': float,
                'help': 'Learning rate to use for the optimizer during training.'
            },
            'rag_weight_decay': {
                'abbr': 'rag_weight_decay', 'default': self._DEF_WEIGHT_DECAY, 'type': float,
                'help': 'Weight decay coefficient for L2 regularization.'
            },
            'rag_save_total_limit': {
                'abbr': 'rag_save_total_limit', 'default': self._DEF_SAVE_TOTAL_LIMIT, 'type': int,
                'help': 'Maximum number of checkpoints to keep. Older ones will be deleted.'
            },
            'rag_compute_device': {
                'abbr': 'rag_device', 'default': self._DEF_DEVICE, 'type': str,
                'help': 'Device to use: cpu or gpu (default {}).'.format(self._DEF_DEVICE)
            },
            'rag_seed': {
                'abbr': 'rag_seed', 'default': self._DEF_RAG_SEED, 'type': int,
                'help': 'Random seed for reproducible generation across runs.'
            },
        }

    
    # set logger from a caller script
    def set_logger(self, logger):
        self._raghf_logger = logger 
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
            
    def build_passage_dataset(self, dataset, rag_context_column):
        passages = list(set([x.strip().replace("\n", " ") for x in dataset[rag_context_column]]))
        return HFDataset.from_dict({"text": passages})

    def make_rag_collate_fn(self, tokenizer, padding_value, rag_top_k_passages, rag_compute_device):
        #print('make_rag_collate_fn: tokenizer', tokenizer, 'padding_value', padding_value)

        def ensure_tensor(t, device):
            if isinstance(t, np.ndarray):
                t = torch.tensor(t)
            return t.to(device)

        def rag_collate_fn(batch):
            #print(f"\n[DEBUG] Collating batch of size {len(batch)}")
            #print("[DEBUG] Raw input_ids lengths:", [len(ex["input_ids"]) for ex in batch])
            #print("[DEBUG] Raw labels lengths:", [len(ex["labels"]) for ex in batch])

            #rag_top_k_passages = rag_top_k_passages
            assert rag_top_k_passages is not None, \
                "Positive integer values for rag_top_k_passages should be passed to RAG collate function"

            pad_token_id = tokenizer.question_encoder.pad_token_id

            input_ids = [ensure_tensor(ex["input_ids"], rag_compute_device) for ex in batch]
            labels = [ensure_tensor(ex["labels"], rag_compute_device) for ex in batch]

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

            attention_mask = (input_ids != pad_token_id).long()
            
            #print("[DEBUG] input_ids shape after padding:", input_ids.shape)
            #print("[DEBUG] attention_mask shape:", attention_mask.shape)
            
            #print("attention_mask dtype (collate, check 1):", attention_mask.dtype)
            #print("attention_mask unique (collate, check 1):", torch.unique(attention_mask))
            assert torch.all((attention_mask == 0) | (attention_mask == 1)), "Invalid values in attention_mask"

            input_ids = input_ids.unsqueeze(1).expand(-1, rag_top_k_passages, -1).reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.unsqueeze(1).expand(-1, rag_top_k_passages, -1).reshape(-1, attention_mask.size(-1))

            # Repeat labels for each passage (same as input_ids)
            labels = labels.repeat_interleave(rag_top_k_passages, dim=0)

            #print("attention_mask dtype (collate, check 2):", attention_mask.dtype)
            #print("attention_mask unique (collate, check 2):", torch.unique(attention_mask))
            assert torch.all((attention_mask == 0) | (attention_mask == 1)), "Invalid values in attention_mask"

            #print("[DEBUG] input_ids shape after expansion:", input_ids.shape)
            #print("[DEBUG] attention_mask shape after expansion:", attention_mask.shape)
            
            if (attention_mask.max() > 1 or attention_mask.min() < 0):
                self._raghf_logger.error("[FATAL] Detected invalid attention_mask at collate time")
                self._raghf_logger.info(attention_mask)
                raise ValueError("Invalid attention_mask")
            
            assert input_ids.size(0) == labels.size(0), f"Mismatch: input_ids {input_ids.size(0)}, labels {labels.size(0)}"

            return {
                "input_ids": input_ids.clone().detach(),
                "attention_mask": attention_mask.clone().detach(),
                "labels": labels,
            }

        return rag_collate_fn


    def prepare_and_index_passages_faiss(self, train_dataset: HFDataset, question_encoder: PreTrainedModel,
            question_encoder_tokenizer: PreTrainedTokenizerBase, rag_trained_model_path:str, 
            rag_context_column:str) -> Tuple[HFDataset, str, str]:
        self._raghf_logger.info('Prepare and index passages for FAISS')
        #print("prepare_and_index_passages_faiss...")

        # Step 1: Build passage dataset
        passage_dataset = self.build_passage_dataset(train_dataset, rag_context_column)
        os.makedirs(rag_trained_model_path, exist_ok=True)
        
        # Step 2: Embedding function
        def embed_passages_with_rag_encoder(batch):
            inputs = question_encoder_tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = question_encoder(**inputs, return_dict=True)
                if outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :]
            return {
                "title": ["some-title"] * len(batch["text"]),
                "embeddings": embeddings.cpu().numpy()
            }

        # Step 3: Embed and index
        passage_dataset = passage_dataset.map(embed_passages_with_rag_encoder, batched=True, batch_size=32)

        # Step 4: Save FAISS index
        index_dir = os.path.join(rag_trained_model_path, "faiss_index")
        os.makedirs(index_dir, exist_ok=True)
        index_file = os.path.join(index_dir, "faiss.index")

        passage_dataset.add_faiss_index("embeddings")
        passage_dataset.get_index("embeddings").save(index_file)
        passage_dataset.drop_index("embeddings")

        # Step 5: Save dataset to disk
        passages_path = os.path.join(rag_trained_model_path, "passages_dataset")
        passage_dataset.save_to_disk(passages_path)

        self._raghf_logger.info("Passages prepared and saved.")
        return passage_dataset, passages_path, index_file
    
    
    def save_rag_model(self, model, tokenizer, retriever, model_dir, rag_index_backend="faiss"):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        self._raghf_logger.info(f"[{rag_index_backend.upper()}] Saving model, tokenizer, and retriever")
        # use_safetensors=True saves RAG model weights and tensors in a more memory efficient format,
        # and loading is faster. When use_safetensors=True, the tensors are saved as model.safetensors,
        # and otherwise as pytorch_model.bin -- in the model_dir.
        model.save_pretrained(model_dir, use_safetensors=True)
        tokenizer.save_pretrained(model_dir)
        retriever.save_pretrained(model_dir)

        self._raghf_logger.info(f"[{rag_index_backend.upper()}] Model, tokenizer, and retriever saved to: {model_dir}")

    def prepare_and_index_passages_elser(self, train_dataset, rag_context_column):
        self._raghf_logger.info("Preparing passages for ELSER")
        
        passages = train_dataset[rag_context_column]
        # optionally push passages to ELSER here using es_client.bulk or ingest pipeline
        
        return passages

    def load_pdf_as_hf_dataset(pdf_path: str, test_split=0.25, seed=42):
        self._raghf_logger.info(f"Loading PDF from {pdf_path}")
        
        markdown = pymupdf4llm.to_markdown(pdf_path)
        splitter = MarkdownHeaderTextSplitter([("#", "H1"), ("##", "H2"), ("###", "H3")], strip_headers=False)
        documents = splitter.split_text(markdown)
        dataset = [doc.page_content for doc in documents]
        hf_dataset = Dataset.from_dict({"text": dataset})
        
        return hf_dataset.train_test_split(test_size=test_split, seed=seed)

    # RagModel wraps The generator model (e.g., BART, T5) and The question encoder (e.g., DPRQuestionEncoder).
    def train(self, rag_text=None, rag_base_model_name=None, rag_trained_model_path=None, rag_index_backend=None, 
            rag_top_k_passages=None, rag_max_input_length=None, rag_max_target_length=None, rag_token=None, 
            rag_trust_remote_code=None, rag_batch_size=None, rag_epochs=None, rag_question_column=None, 
            rag_context_column=None, rag_eval_strategy=None, rag_save_steps=None,  rag_logging_steps=None, 
            rag_report_to=None, rag_lr=None, rag_weight_decay=None, rag_save_total_limit=None, rag_compute_device=None):
        '''
        The flow during training looks like:
        Trainer calls: trainer.train()
        Trainer calls: model(input_ids, attention_mask, labels, ...)
        RAG model calls: retriever(input_ids=input_ids, attention_mask=attention_mask)
        Your CosineRetriever.__call__() is triggered
        Inside it, you again call the question encoder tokenizer on input_ids
        '''
        self._raghf_logger.info(f"Loading dataset (text) {rag_text}")

        # Get passages from any supported file type
        passages = self.preprocessor.get_passages(rag_text)

        # Convert to HuggingFace Dataset
        split_dataset = self.preprocessor.get_hf_dataset(passages, test_split=0.25, seed=42)

        # Now proceed with HF RAG Trainer setup using train_dataset and eval_dataset
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        #print("1 Train columns:", train_dataset.column_names)
        #print("1 eval columns:", eval_dataset.column_names)
        
        self._raghf_logger.info(f"Loading tokenizer and model {rag_base_model_name}")
        tokenizer = RagTokenizer.from_pretrained(rag_base_model_name, trust_remote_code=rag_trust_remote_code)
        # === Apply tokenizer overrides here ===
        tokenizer.question_encoder.model_max_length = rag_max_input_length
        tokenizer.question_encoder.init_kwargs['model_max_length'] = rag_max_input_length

        # Debug checks
        #print("[APPLIED] tokenizer.question_encoder.model_max_length:", tokenizer.question_encoder.model_max_length)
        #print("[APPLIED] tokenizer.question_encoder.init_kwargs['model_max_length']:", tokenizer.question_encoder.init_kwargs.get('model_max_length'))
        
        if tokenizer.question_encoder.pad_token is None:
            tokenizer.question_encoder.pad_token = tokenizer.question_encoder.eos_token
        if tokenizer.generator.pad_token is None:
            tokenizer.generator.pad_token = tokenizer.generator.eos_token
        tokenizer.question_encoder.padding_side = "right"
        tokenizer.generator.padding_side = "right"

        model_class = RagTokenForGeneration if rag_token else RagSequenceForGeneration
        # expected printout: Training with: <class 'transformers.models.rag.modeling_rag.RagTokenForGeneration'>
        self._raghf_logger.info(f"Training with: {model_class}") 
        self.rag_trained_model = model_class.from_pretrained(rag_base_model_name, trust_remote_code=rag_trust_remote_code)

        # question_encoder is the actual PyTorch model used to embed questions/passages. 
        # In the case of RagTokenForGeneration or RagSequenceForGeneration, this is a 
        # DPRQuestionEncoder, which wraps a BERT-style encoder.
        question_encoder = self.rag_trained_model.rag.question_encoder; 
        #print('question_encoder from rag_trained_model', question_encoder)
        
        # question_encoder_tokenizer is the tokenizer specific to the question_encoder  
        # (typically a DPRQuestionEncoderTokenizer or a BERT tokenizer). It must be used to 
        # ensure that tokenization is compatible with the encoder's input requirements.
        # That is, question_encoder and question_encoder_tokenizer work together, as a pair.
        question_encoder_tokenizer = tokenizer.question_encoder

        self._raghf_logger.info("Preparing passages and retriever for FAISS")
        if rag_index_backend == "faiss":
            passage_dataset, passages_path, index_path = self.prepare_and_index_passages_faiss(
                train_dataset, question_encoder, question_encoder_tokenizer, rag_trained_model_path, rag_context_column)
            passages_save_path = os.path.join(rag_trained_model_path, "passages_dataset")
            os.makedirs(passages_save_path, exist_ok=True)
            passage_dataset.save_to_disk(passages_save_path)
            #print("passages_path:", passages_path) # passages_path = "./rag_smlp_toy_faiss/passages_dataset"
            #print("index_path:", index_path) # index_path = "./rag_smlp_toy_faiss/faiss_index/faiss.index"
            assert passages_path is not None, "passages_path is None"
            assert index_path is not None, "index_path is None"
            assert os.path.exists(index_path), f"Index file not found: {index_path}"
            assert os.path.exists(passages_path), f"Passages path not found: {passages_path}"

            retriever = RagRetriever.from_pretrained(
                retriever_name_or_path=rag_base_model_name,   # e.g., "facebook/rag-token-base"
                index_name="custom", #"exact",                     # "exact" means use FAISS, "custom" is the opposite (???)
                passages_path=passages_path,                       # e.g., ./rag_smlp_toy_faiss/passages_dataset
                index_path=index_path,                             # e.g., ./rag_smlp_toy_faiss/faiss_index/faiss.index
                n_docs=rag_top_k_passages,
                use_dummy_dataset=False                            # must be False for real passage set
            )
            self._raghf_logger.info('FAISS passages and retriever created')
        elif rag_index_backend == "elastic":
            self._raghf_logger.info("Preparing passages and retriever for ELSER")
            
            '''
            -- Your Elasticsearch server is version 8.x (from Docker).
            -- But your Python client (elasticsearch-py) is sending headers indicating v9 compatibility, 
                which Elasticsearch 8.x does not support.
            -- Fix that worked: pip install elasticsearch==8.13.0; this version auto-detects and matches 
                server version better. Alternative was to use the new Elasticsearch class from elastic-transport 
                with version compatibility explicitly declared (less common in simpler use cases).
            '''
            
            # 1. Get the passages and index name
            passages = self.prepare_and_index_passages_elser(train_dataset, rag_context_column)

            # 2. Encode passages (optional, for logging/debug)
            self._raghf_logger.info("Encoding passages with tokenizer (for dimension check)")
            inputs = tokenizer.question_encoder(
                passages,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=rag_max_input_length
            ).to(rag_compute_device)

            with torch.no_grad():
                outputs = question_encoder(**inputs)
                passage_embeddings = outputs[0]  # (N, H) or (N, T, H)

                if passage_embeddings.ndim == 3:
                    # fallback: mean pooling
                    passage_embeddings = passage_embeddings.mean(dim=1)

                self._raghf_logger.info(f"Passage embeddings' shape: {passage_embeddings.shape}")

            # 3. Create ElasticRetriever
            es_client = Elasticsearch("http://localhost:9200")
            retriever = ElasticRetriever(
                es_client=es_client,  # You should have initialized this earlier
                index_name="rag_smlp_elser",
                tokenizer=tokenizer,  # must be RagTokenizer
                passages=passages,
                passage_embeddings=passage_embeddings,
                top_k_passages=rag_top_k_passages,
                max_input_length=rag_max_input_length,
            )
            self._raghf_logger.info('ELSER retriever created')
        elif rag_index_backend == "cosine":
            self._raghf_logger.info("Encoding passages for cosine similarity")
            passages = train_dataset[rag_context_column]; #print('passages', passages)
            # Here use tokenizer.question_encoder(...) to tokenize passages (producing input_ids, attention_mask, etc.).
            def safe_tokenize(tokenizer, texts, max_length, device):
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                return {
                    "input_ids": encoded["input_ids"].to(device),
                    "attention_mask": encoded["attention_mask"].to(device),
                }

            inputs = safe_tokenize(tokenizer.question_encoder, passages, rag_max_input_length, rag_compute_device)
            #print('tokenizer.question_encoder.model_max_length =', tokenizer.question_encoder.model_max_length)
            #print("input_ids shape:", inputs["input_ids"].shape)
            #print("attention_mask shape:", inputs["attention_mask"].shape)

            assert inputs["input_ids"].shape == inputs["attention_mask"].shape, \
                f"Shape mismatch: {inputs['input_ids'].shape} vs {inputs['attention_mask'].shape}"

            #print('inputs', inputs)
            assert inputs["input_ids"].shape == inputs["attention_mask"].shape, \
                "input_ids and attention_mask length mismatch!"

            # Use question_encoder(...) to generate the embeddings from tokenized input.
            with torch.no_grad():
                outputs = question_encoder(**inputs); #print('outputs', outputs)
                passage_embeddings = outputs[0]  # shape: (batch_size, hidden_size)
                #print("passage_embeddings:", passage_embeddings.shape)
            
            #print("Max length (model_max_length):", tokenizer.question_encoder.model_max_length)
            #print("Max input length (from config):", tokenizer.question_encoder.init_kwargs.get("max_length", "Not set"))
            #print("Truncation side:", tokenizer.question_encoder.truncation_side)
            #print("Padding side:", tokenizer.question_encoder.padding_side)

            retriever = CosineRetriever(
                passage_embeddings=passage_embeddings, # shape [n_passages, hidden_dim]
                question_encoder=question_encoder,     # DPRQuestionEncoder (BERT-based)
                tokenizer=question_encoder_tokenizer,  # Tokenizer for question encoding
                passages=passages,                     # List of original passage strings
                top_k_passages=rag_top_k_passages,
                max_input_length=rag_max_input_length,
                raghf_logger=self._raghf_logger
            )
            self._raghf_logger.info('Cosine similarity passage embbeddings and retriever created')
        else:
            raise ValueError(f"Unsupported rag_index_backend: {rag_index_backend}")

        self._raghf_logger.info("Loading retriever")
        self.rag_trained_model.set_retriever(retriever)

        if hasattr(self.rag_trained_model, "config"):
            self.rag_trained_model.config.num_beams = 1
            self.rag_trained_model.config.num_return_sequences = 1
            self.rag_trained_model.config.n_docs = rag_top_k_passages
        if hasattr(self.rag_trained_model, "generation_config"):
            if self.rag_trained_model.generation_config is None:
                self.rag_trained_model.generation_config = GenerationConfig.from_model_config(
                    self.rag_trained_model.config
                )
            self.rag_trained_model.generation_config.num_beams = 1
            self.rag_trained_model.generation_config.num_return_sequences = 1
            self.rag_trained_model.generation_config.n_docs = rag_top_k_passages

        def get_padding_value(model, tokenizer):
            is_rag = isinstance(getattr(model, "base_model", model), RagModel)
            pad_token_id = tokenizer.generator.pad_token_id if is_rag else tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("Tokenizer has no pad_token_id set.")
            return pad_token_id if is_rag else -100

        padding_value = get_padding_value(self.rag_trained_model, tokenizer); #print('padding_value', padding_value)

        def preprocess(example):
            question = example.get(rag_question_column, "")
            context = example.get(rag_context_column, "")

            question_encoding = tokenizer.question_encoder(
                question,
                truncation=True,
                padding="max_length",
                max_length=rag_max_input_length,
                return_tensors="pt"
            )

            context_encoding = tokenizer.generator(
                context,
                truncation=True,
                padding="max_length",
                max_length=rag_max_target_length,
                return_tensors="pt"
            )

            decoder_input_ids = context_encoding["input_ids"].squeeze(0)
            labels = decoder_input_ids.clone()
            labels[labels == tokenizer.generator.pad_token_id] = padding_value

            return {
                "input_ids": question_encoding["input_ids"].squeeze(0),
                "attention_mask": question_encoding["attention_mask"].squeeze(0),
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": context_encoding["attention_mask"].squeeze(0),
                "labels": labels,
            }

        self._raghf_logger.info("Tokenizing datasets")
        train_dataset = train_dataset.map(lambda x: preprocess(x), remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(lambda x: preprocess(x), remove_columns=eval_dataset.column_names)
        train_dataset.set_format(type="torch")
        eval_dataset.set_format(type="torch")

        self._raghf_logger.info("Creating training_args")
        training_args = Seq2SeqTrainingArguments(
            output_dir=rag_trained_model_path,
            per_device_train_batch_size=rag_batch_size,
            per_device_eval_batch_size=rag_batch_size,
            do_train=True,
            do_eval=True,
            num_train_epochs=rag_epochs,
            logging_dir=os.path.join(rag_trained_model_path, "logs"),
            eval_strategy="epoch",
            save_steps=100,
            logging_steps=10,
            report_to="none",
        )
        
        self._raghf_logger.info("Creating trainer")      
        assert retriever.n_docs == rag_top_k_passages
        trainer = CustomRAGTrainer(
            model=self.rag_trained_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=self.make_rag_collate_fn(tokenizer, padding_value, rag_top_k_passages, rag_compute_device),
            rag_batch_size=rag_batch_size,
            rag_top_k_passages=retriever.n_docs,
            raghf_logger=self._raghf_logger
        )
        
        tokenizer.question_encoder.model_max_length = rag_max_input_length
        tokenizer.generator.model_max_length = rag_max_input_length
        tokenizer.question_encoder.init_kwargs['max_length'] = rag_max_input_length
        tokenizer.generator.init_kwargs['max_length'] = rag_max_input_length
        
        #print("[FINAL] tokenizer.generator.model_max_length:", tokenizer.generator.model_max_length)
        #print("[FINAL] tokenizer.question_encoder.model_max_length =", tokenizer.question_encoder.model_max_length)
        #print("[FINAL] tokenizer.question_encoder.padding_side =", tokenizer.question_encoder.padding_side)
        #print("[FINAL] tokenizer.question_encoder.truncation_side =", tokenizer.question_encoder.truncation_side)
        #print("[FINAL] max_input_length =", rag_max_input_length)

        self._raghf_logger.info("Training starts")
        trainer.train()
        self._raghf_logger.info("Training completed")

        # Save everything
        self._raghf_logger.info(f"Saving model, tokenizer, and retriever to {rag_trained_model_path}")
        self.save_rag_model(trainer.model, tokenizer, retriever, rag_trained_model_path, rag_index_backend)

        # Store for in-memory use during generate()
        self.tokenizer = tokenizer
        self.retriever = retriever

        with open(os.path.join(rag_trained_model_path, "rag_smlp_config.json"), "w") as f:
            json.dump({"rag_token": rag_token}, f)

        self._raghf_logger.info(f"Model, tokenizer, and retriever saved to: {rag_trained_model_path}")
        
        
    '''
    To rerun a trained RAG model from disk (e.g., RagSequenceForGeneration.from_pretrained()), you need:
    --config.json: Model architecture and settings
    --model.safetensors or pytorch_model.bin: Actual model weights
    --tokenizer_config.json + tokenizer.json: Top-level tokenizer wrapper config (RagTokenizer)
    --question_encoder_tokenizer/: Subdir with config.json and vocab.txt or equivalent (DPR tokenizer)
    --generator_tokenizer/: Subdir with config.json and tokenizer files (Bart tokenizer)
    --rag_config.json (sometimes separate): Configuration for the retriever/tokenizer pairing
    --faiss_index.faiss (for FAISS retriever): Optional — used if you're using FAISS retriever backend
    '''
    def load_model_faiss(self, model_dir, rag_top_k_passages, rag_compute_device, use_dummy_dataset=True):
        """
        Load a previously saved RAG model and tokenizer from disk.

        Args:
            model_dir (str): Directory where the model and tokenizer were saved.
            use_dummy_dataset (bool): If True, loads a dummy dataset to satisfy dataset dependency.
        """
        self._raghf_logger.info(f"Loading model and tokenizer from {model_dir}") # expected: LegacyFaissIndex

        self.tokenizer = RagTokenizer.from_pretrained(model_dir)

        self.retriever = RagRetriever.from_pretrained(
            retriever_name_or_path=model_dir, 
            index_name="custom", #"exact",
            passages_path=os.path.join(model_dir, "passages_dataset"),
            index_path=os.path.join(model_dir, "faiss_index", "faiss.index"),
            _docs=rag_top_k_passages,              # or model.config.n_docs
            use_dummy_dataset=False
        )

        model = RagTokenForGeneration.from_pretrained(
            model_dir,
            retriever=self.retriever
        )

        self._raghf_logger.info(f"Retriever index type: {type(self.retriever.index).__name__}")

        self.rag_trained_model = RagSequenceForGeneration.from_pretrained(model_dir, retriever=self.retriever)

        # Ensure model is on the right device
        self.rag_trained_model.to(rag_compute_device)

        self._raghf_logger.info("Model, tokenizer, and retriever loaded successfully")
        
    def load_model_cosine(self, model_dir, rag_token, rag_top_k_passages, rag_compute_device, rag_max_input_length):
        """
        Load a RAG model trained with cosine similarity-based retriever.
        """
        self._raghf_logger.info(f"Loading model and retriever for cosine similarity from: {model_dir}")

        # Load tokenizers directly
        self.tokenizer = RagTokenizer.from_pretrained(model_dir)
        
        # Load passage embeddings and text
        passage_embeddings = torch.load(os.path.join(model_dir, "passage_embeddings.pt"))
        with open(os.path.join(model_dir, "passages.txt")) as f:
            passages = [line.strip() for line in f]

        #print(len(passages), rag_top_k_passages, passages)

        retriever = CosineRetriever(
            passage_embeddings=passage_embeddings,
            question_encoder=None,  # optional
            tokenizer=self.tokenizer.generator,
            passages=passages,
            top_k_passages=rag_top_k_passages,
            max_input_length=rag_max_input_length,
            raghf_logger=self._raghf_logger
        )
        #print("[DEBUG] Retriever created:", type(retriever))

        # Use the context manager ONLY here
        model_class = RagTokenForGeneration if rag_token else RagSequenceForGeneration
        self.rag_trained_model = model_class.from_pretrained(model_dir, retriever=retriever)
        assert self.rag_trained_model is not None, "Model load failed"
        #print("[DEBUG] Model loaded:", type(self.rag_trained_model))
        self.rag_trained_model.to(rag_compute_device)
        self._raghf_logger.info("Model, retriever, and tokenizer loaded successfully (cosine)")

    def retrieve_only(self, rag_questions, rag_trained_model_path, rag_index_backend, 
                  rag_top_k_passages, rag_max_input_length, rag_token, rag_compute_device):
        """Retrieve passages without generating answers (for debugging)."""

        if isinstance(rag_questions, str):
            rag_questions = [rag_questions]

        # Load model
        if rag_index_backend == 'faiss':
            self.load_model_faiss(
                model_dir=rag_trained_model_path, 
                rag_top_k_passages=rag_top_k_passages, 
                rag_compute_device=rag_compute_device
            )
        elif rag_index_backend == 'cosine':
            self.load_model_cosine(
                model_dir=rag_trained_model_path, 
                rag_token=rag_token, 
                rag_top_k_passages=rag_top_k_passages, 
                rag_compute_device=rag_compute_device, 
                rag_max_input_length=rag_max_input_length
            )
        else:
            raise ValueError(f"Unsupported rag_index_backend: {rag_index_backend}")

        tokenizer = self.tokenizer
        model = self.rag_trained_model
        device = model.device

        results = []
        for question in rag_questions:
            self._raghf_logger.info(f"Retrieving for question: {question}")

            # Tokenize and encode question
            inputs = tokenizer.question_encoder(
                [question],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=rag_max_input_length
            ).to(device)

            # Encode question
            with torch.no_grad():
                outputs = model.question_encoder(**inputs)
                hidden_states = outputs[0]
                attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
                masked = hidden_states * attention_mask
                summed = masked.sum(dim=1)
                count = attention_mask.sum(dim=1)
                question_hidden_states = summed / count
                question_hidden_states = question_hidden_states.cpu().numpy()

            # Retrieve documents
            if rag_index_backend == "cosine":
                retrieved_docs = model.retriever(
                    input_ids=None,
                    attention_mask=None,
                    question_hidden_states=question_hidden_states,
                    prefix=model.generator.config.prefix,
                    n_docs=model.config.n_docs,
                    return_tensors="pt"
                )
                passages = retrieved_docs["retrieved_passages"][0]
            else:  # faiss or elastic
                retrieved_doc_embeds, doc_ids, doc_dicts = model.retriever.retrieve(
                    question_hidden_states=question_hidden_states,
                    n_docs=model.config.n_docs
                )

                if isinstance(doc_dicts, list):
                    if isinstance(doc_dicts[0], str):
                        passages = doc_dicts
                    elif isinstance(doc_dicts[0], dict) and "text" in doc_dicts[0]:
                        passages = [doc["text"] for doc in doc_dicts]
                    elif isinstance(doc_dicts[0], list):
                        passages = [doc["text"] for sublist in doc_dicts for doc in sublist]
                    else:
                        passages = [str(d) for d in doc_dicts]
                else:
                    passages = [str(doc_dicts)]

            results.append({
                "question": question,
                "retrieved_passages": passages[:rag_top_k_passages]
            })

            self._raghf_logger.info(f"Retrieved {len(passages[:rag_top_k_passages])} passages")

        return results
        
    def generate_batched(self, rag_questions, batch_size=4, **kwargs):
        """Process multiple questions in batches for efficiency."""
        all_answers = []
        all_retrieved = []

        for i in range(0, len(rag_questions), batch_size):
            batch = rag_questions[i:i+batch_size]
            answers, retrieved = self.generate(rag_questions=batch, **kwargs)
            all_answers.extend(answers)
            all_retrieved.extend(retrieved)

        return all_answers, all_retrieved

    # TODO !!!! define command line options for num_return_sequences=1, num_beams=1, max_length=64 ???    
    def generate(self, rag_questions=None, rag_trained_model_path=None, rag_index_backend=None, 
            rag_top_k_passages=None, rag_max_input_length=None, rag_max_new_tokens=None, 
            rag_token=None, rag_sample=None, num_return_sequences=1, num_beams=1, 
            rag_compute_device=None, seed=42):
        """
        Generate answers using cosine similarity retriever and RAG model.
        Now supports multiple questions with deterministic generation.

        Args:
            rag_questions: Single question (str) or list of questions
            seed: Random seed for reproducible generation (default: 42)
            ... (other args as before)

        Returns:
            Tuple of (answers, retrieved_passages) where each is a list
        """
        # Set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            self._raghf_logger.info(f"Set random seed to {seed} for deterministic generation")

        # TODO !!!! although loading saved model is OK also with in-memory flow
        # (meaning, generation in the same session when training was performed), it is
        # best to investigate why pure in-memory generation (when model is not loaded from disk)
        # is not working correctly -- maybe some of the parameters in saved model are not
        # reflected properly in self object, causing in-memory and from-disk generation to diverge.
        self._raghf_logger.info(f"Loading RAG trained model from: {rag_trained_model_path}")
        config_path = os.path.join(rag_trained_model_path, "rag_smlp_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            rag_token_from_config = config.get("rag_token", False)  # Default to False if not found
            self._raghf_logger.info(f"rag_token loaded from config: {rag_token_from_config}")
        else:
            #print("[WARNING] Config file not found. Using default rag_token:", rag_token)
            raise Exception('Config file not found. Cannot read value of rag_token')

        assert rag_token == rag_token_from_config, f"Generation was launched with wrong rag_token value {rag_token}"

        if rag_index_backend == 'faiss':
            self.load_model_faiss(model_dir=rag_trained_model_path, rag_top_k_passages=rag_top_k_passages, 
                rag_compute_device=rag_compute_device)
        elif rag_index_backend == 'cosine':
            self.load_model_cosine(model_dir=rag_trained_model_path, rag_token=rag_token, rag_top_k_passages=rag_top_k_passages, 
                rag_compute_device=rag_compute_device, rag_max_input_length=rag_max_input_length)
        elif rag_index_backend == 'elastic':
            assert False, "elastic based RAG is not yet supported"
        else:
            assert False, ("Unexpected index backend " + str(rag_index_backend))

        # Ensure questions is a list
        if isinstance(rag_questions, str):
            rag_questions = [rag_questions]
            single_question = True
        else:
            single_question = False

        tokenizer = self.tokenizer
        model = self.rag_trained_model
        device = model.device

        self._raghf_logger.info(f"Generating answers for {len(rag_questions)} question(s)")

        # Storage for all results
        all_answers = []
        all_retrieved_passages = []

        # Process each question
        for question_idx, question in enumerate(rag_questions):
            try:
                self._raghf_logger.info(f"{'='*60}")
                self._raghf_logger.info(f"Processing question {question_idx+1}/{len(rag_questions)}: {question}")
                self._raghf_logger.info(f"{'='*60}")

                #print('rag_max_input_length used in inputs = tokenizer.question_encoder(...)', rag_max_input_length)
                # Tokenize for question encoder
                inputs = tokenizer.question_encoder(
                    [question],  # Single question as list
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=rag_max_input_length
                ).to(device)

                # Encode question
                with torch.no_grad():
                    outputs = model.question_encoder(**inputs)
                    hidden_states = outputs[0]  # (B, T, H)
                    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
                    masked = hidden_states * attention_mask
                    summed = masked.sum(dim=1)
                    count = attention_mask.sum(dim=1)
                    question_hidden_states = summed / count  # (B, H)
                    question_hidden_states = question_hidden_states.cpu().numpy()

                if rag_index_backend in ["faiss", "elastic"]:
                    # FAISS or other HuggingFace retrievers
                    retrieved_doc_embeds, doc_ids, doc_dicts = model.retriever.retrieve(
                        question_hidden_states=question_hidden_states,
                        n_docs=model.config.n_docs
                    )

                    # Flatten out the retrieved texts
                    #flat_passages = [doc["text"] for doc_list in doc_dicts for doc in doc_list]
                    # Handles both flat list of strings or nested list of dicts
                    # Sanitize flat_passages
                    if isinstance(doc_dicts, list):
                        if isinstance(doc_dicts[0], str):
                            flat_passages = doc_dicts
                        elif isinstance(doc_dicts[0], dict) and "text" in doc_dicts[0]:
                            flat_passages = [doc["text"] for doc in doc_dicts]
                        elif isinstance(doc_dicts[0], list):  # Nested
                            flat_passages = [doc["text"] for sublist in doc_dicts for doc in sublist]
                        else:
                            raise ValueError(f"Unexpected doc_dicts[0] structure: {type(doc_dicts[0])}")
                    elif isinstance(doc_dicts, str):
                        flat_passages = [doc_dicts]
                    else:
                        raise ValueError(f"Unexpected doc_dicts type: {type(doc_dicts)}")

                    # FIX HERE — ensure tokenizer input is flat list of strings
                    if isinstance(flat_passages[0], list):
                        flat_passages = [p for sublist in flat_passages for p in sublist]

                    #print("flat_passages:", flat_passages)
                    #print("flat_passages[0]:", flat_passages[0])
                    #print("len(flat_passages):", len(flat_passages))

                    # Tokenize retrieved docs
                    tokenized = tokenizer.generator(
                        flat_passages,
                        padding="max_length",
                        truncation=True,
                        max_length=rag_max_input_length or model.config.max_length,
                        return_tensors="pt"
                    )

                    context_input_ids = tokenized["input_ids"]
                    context_attention_mask = tokenized["attention_mask"]

                    # Wrap it into a dictionary so rest of the code can proceed
                    retrieved_docs = {
                        "context_input_ids": context_input_ids,
                        "context_attention_mask": context_attention_mask,
                        "retrieved_doc_embeds": torch.tensor(retrieved_doc_embeds)
                    }

                    #self._raghf_logger.info("Top Retrieved Documents:")
                    #for i, doc_text in enumerate(flat_passages[:model.config.n_docs]):
                    #    self._raghf_logger.info(f"  {i+1}. {doc_text.strip()[:200]}...")

                elif rag_index_backend == "cosine":
                    # CosineRetriever: use question_hidden_states directly
                    #question_hidden_states = question_hidden_states.cpu().numpy()
                    retrieved_docs = model.retriever(
                        input_ids=None,
                        attention_mask=None,
                        question_hidden_states=question_hidden_states,
                        prefix=model.generator.config.prefix,
                        n_docs=model.config.n_docs,
                        return_tensors="pt"
                    )
                else:
                    raise ValueError("Unexpected index_backend " + str(rag_index_backend)) 

                retrieved_passages = None
                if rag_index_backend in ["faiss", "elastic"]:  # FAISS flow
                    # Do not expect 'retrieved_passages' key in FAISS
                    retrieved_passages = flat_passages[:model.config.n_docs]
                elif rag_index_backend == "cosine":  # CosineRetriever or custom
                    retrieved_passages = retrieved_docs.get("retrieved_passages", [])
                else:
                    raise ValueError("Unexpected index_backend " + str(rag_index_backend)) 

                # Print top passages (robust for both retrievers)
                #print("Top Retrieved Documents -- original text:")
                self._raghf_logger.info("Top Retrieved Passages:")
                q_retrieved = ""
                if isinstance(retrieved_passages[0], list):  # Cosine may return nested
                    for i, passage in enumerate(retrieved_passages[0]):
                        q_retrieved = q_retrieved + '  ' + passage
                        self._raghf_logger.info(f"  {i+1}. {passage[:200]}...")
                else:  # FAISS
                    for i, passage in enumerate(retrieved_passages):
                        q_retrieved = q_retrieved + '  ' + passage
                        self._raghf_logger.info(f"  {i+1}. {passage[:200]}...")

                if model.config.decoder_start_token_id is None:
                    model.config.decoder_start_token_id = tokenizer.generator.bos_token_id

                # Generate final answers
                generate_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "context_input_ids": retrieved_docs["context_input_ids"].to(device),
                    "context_attention_mask": retrieved_docs["context_attention_mask"].to(device),
                    "max_new_tokens": rag_max_new_tokens,
                    "num_beams": num_beams,
                    "num_return_sequences": num_return_sequences,
                    "do_sample": rag_sample,
                    "forced_bos_token_id": model.config.decoder_start_token_id,
                    "decoder_start_token_id": model.config.decoder_start_token_id,
                }

                # Only include doc_scores if available -- that is, for cosine similarity index backend
                doc_scores = retrieved_docs.get("doc_scores")
                if rag_index_backend == 'cosine':
                    assert doc_scores is not None, "doc_scores are required for cosine similarity based RAG flow"
                    generate_kwargs["doc_scores"] = doc_scores.to(model.device)
                elif rag_index_backend in ['faiss', 'elastic']:
                    assert doc_scores is None, "doc_scores are irrelevant for FAISS/elastic based RAG flow"

                if rag_sample:
                    generate_kwargs.update({
                        "temperature": 0.1,  # Low temperature for more determinism
                        "top_p": 0.9,
                        "top_k": 50,
                        "no_repeat_ngram_size": 3,
                    })
                    # Add generator for reproducible sampling
                    if seed is not None:
                        generator = torch.Generator(device=device)
                        generator.manual_seed(seed + question_idx)  # Different seed per question
                        generate_kwargs["generator"] = generator
                else:
                    generate_kwargs["early_stopping"] = True

                # Final call
                with torch.no_grad():
                    outputs = model.generate(**generate_kwargs)

                decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                self._raghf_logger.info(f"Generated Answer: {decoded_output}")

                all_answers.append(decoded_output)
                all_retrieved_passages.append(q_retrieved)

            except Exception as e:
                self._raghf_logger.error(f"Failed to process question {question_idx+1}: {question}")
                self._raghf_logger.error(f"Error: {str(e)}")
                self._raghf_logger.error(traceback.format_exc())

                # Append error placeholders so we don't lose track of questions
                all_answers.append(f"[ERROR: {str(e)}]")
                all_retrieved_passages.append("")

        # Summary log
        self._raghf_logger.info(f"{'='*60}")
        self._raghf_logger.info(f"Generation complete. Processed {len(rag_questions)} questions.")
        self._raghf_logger.info(f"Successful: {sum(1 for a in all_answers if not a.startswith('[ERROR'))}")
        self._raghf_logger.info(f"Failed: {sum(1 for a in all_answers if a.startswith('[ERROR'))}")
        self._raghf_logger.info(f"{'='*60}\n")

        # Return in same format as before
        return all_answers, all_retrieved_passages

    def run(self, rag_questions=None, rag_text=None, rag_base_model_name=None, rag_trained_model_path=None, 
            rag_top_k_passages=None, rag_index_backend=None, rag_max_input_length=None, 
            rag_max_target_length=None, rag_max_new_tokens=None, rag_sample=None, rag_token=None, 
            rag_trust_remote_code=None, rag_train=None, rag_eval=None, rag_batch_size=None, rag_epochs=None,
            rag_question_column=None, rag_context_column=None, 
            rag_eval_strategy=None, rag_save_steps=None,  rag_logging_steps=None, rag_report_to=None, rag_lr=None,
            rag_weight_decay=None, rag_save_total_limit=None, rag_compute_device=None, rag_seed=42):   
        
        if rag_train:
            self._raghf_logger.info("Starting RAG training")
            self.train(rag_text=rag_text, rag_base_model_name=rag_base_model_name, 
                rag_trained_model_path=rag_trained_model_path, 
                rag_top_k_passages=rag_top_k_passages, rag_index_backend=rag_index_backend, 
                rag_max_input_length=rag_max_input_length, rag_max_target_length=rag_max_target_length, 
                rag_token=rag_token, rag_trust_remote_code=rag_trust_remote_code, 
                rag_batch_size=rag_batch_size, rag_epochs=rag_epochs, 
                rag_question_column=rag_question_column, rag_context_column=rag_context_column, 
                rag_eval_strategy=rag_eval_strategy, rag_save_steps=rag_save_steps, 
                rag_logging_steps=rag_logging_steps, rag_report_to=rag_report_to, rag_lr=rag_lr,
                rag_weight_decay=rag_weight_decay, rag_save_total_limit=rag_save_total_limit,
                rag_compute_device=rag_compute_device)

        if rag_eval:
            self._raghf_logger.info("Starting RAG generation")
            return self.generate(rag_questions=rag_questions, rag_trained_model_path=rag_trained_model_path, 
                rag_top_k_passages=rag_top_k_passages, rag_index_backend=rag_index_backend, 
                rag_max_new_tokens=rag_max_new_tokens, rag_token=rag_token, rag_sample=rag_sample, 
                rag_compute_device=rag_compute_device, seed=rag_seed)
            
    def cleanup_memory(self):
        """
        Free up GPU/CPU memory after RAG operations.
        Call this after generation is complete and before loading judge model.
        """
        self._raghf_logger.info("Cleaning up RAG model memory...")

        # Delete model and retriever
        if hasattr(self, 'rag_trained_model') and self.rag_trained_model is not None:
            del self.rag_trained_model
            self.rag_trained_model = None

        if hasattr(self, 'retriever') and self.retriever is not None:
            del self.retriever
            self.retriever = None

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self._raghf_logger.info(f"GPU memory freed. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)} bytes")

        # Force garbage collection
        gc.collect()

        self._raghf_logger.info("Memory cleanup complete")
        

class SmlpRag:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        
        # Import or instantiate the other classes
        # Union of all parameters into one unified dictionary
        self.hfragInst = HuggingFaceRag(overrides=self.overrides)
        self.lcragInst = LangChainRag(overrides=self.overrides)
        
        base_params = BaseRAG()._base_rag_params
        hf_params = self.hfragInst._hf_rag_params
        lc_params = self.lcragInst._lc_rag_params
        rag_type_params = {
            'rag_type': {
                'abbr': 'rag_type', 'default': 'hf', 'type': str,
                'help': 'whether to use HuggingFace based RAG (value hf) or LangChain based one (value lc). '
                    'A RAG flow built using LangChain’s orchestration — chaining retriever, prompt, model, parser '
                    'The retrieval part (FAISS, etc.) and chain building is handled by LangChain. '
                    'Ollama provides the LLM endpoint — it’s the model LangChain calls during generation. '
            },
        }
        
        self.judge = SmlpRagJudge()
        
        self.rag_params_dict = rag_type_params | base_params | hf_params | lc_params
    
    def _normalize_rag_outputs(self, output_text, retrieved_contexts, rag_questions):
        """
        Convert backend-specific outputs to a common format.

        Args:
            output_text: Generated answers
            retrieved_contexts: List of retrieved passage strings (one per question)
            rag_questions: List of questions
        """
        normalized = []

        if isinstance(output_text, list):
            for q, ans, ctx in zip(rag_questions, output_text, retrieved_contexts):
                normalized.append({
                    "question": q,
                    "answer": ans,
                    "context": ctx  # Retrieved passages for THIS question
                })
        else:
            # Single question case
            normalized.append({
                "question": rag_questions[0],
                "answer": output_text,
                "context": retrieved_contexts[0]
            })

        return normalized
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._rag_logger = logger 
        self.hfragInst.set_logger(logger)
        self.lcragInst.set_logger(logger)
        self.judge.set_logger(logger)
        
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        self.hfragInst.set_report_file_prefix(report_file_prefix)
        self.lcragInst.set_report_file_prefix(report_file_prefix)
        self.judge.set_report_file_prefix(report_file_prefix)
    
    @property
    def generated_text_file_name(self):
        return self.report_file_prefix + '_rag_generated.txt'
    
    def retrieve_only(self, rag_questions, rag_type, rag_trained_model_path, 
                      rag_index_backend, rag_top_k_passages, rag_max_input_length, 
                      rag_token, rag_compute_device):
        """
        Retrieve passages without generating answers (debugging/analysis).
        Only works with HF RAG.
        """
        if rag_type != "hf":
            raise ValueError("retrieve_only is only supported for HuggingFace RAG (rag_type='hf')")

        return self.hfragInst.retrieve_only(
            rag_questions=rag_questions,
            rag_trained_model_path=rag_trained_model_path,
            rag_index_backend=rag_index_backend,
            rag_top_k_passages=rag_top_k_passages,
            rag_max_input_length=rag_max_input_length,
            rag_token=rag_token,
            rag_compute_device=rag_compute_device
        )
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage for debugging."""

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        self._rag_logger.info(f"[{stage}] Memory usage:")
        self._rag_logger.info(f"  - RAM: {mem_info.rss / 1024**2:.2f} MB")
        
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            self._rag_logger.info(f"  - GPU allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            self._rag_logger.info(f"  - GPU reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            self._rag_logger.info(f"  - GPU free: {free_memory / 1024**2:.2f} MB")

    def smlp_rag(self, rag_questions:list[str]=None, rag_text:str=None, rag_type:str=None, rag_base_model_name:str=None, 
            rag_trained_model_path:str=None, rag_index_backend=None, rag_top_k_passages=None, rag_base_url:str=None, 
            rag_embedding_model=None,  rag_prompt_type:str=None, rag_max_input_length=None, rag_max_target_length=None, 
            rag_max_new_tokens=None, rag_trust_remote_code=None, rag_sample=None, rag_token=None, rag_train=None,
            rag_eval=None, rag_batch_size=None, rag_epochs=None, rag_question_column=None, rag_context_column=None, 
            rag_eval_strategy=None, rag_save_steps=None,  rag_logging_steps=None, rag_report_to=None, rag_lr=None,
            rag_weight_decay=None, rag_save_total_limit=None, rag_compute_device=None,
            llm_quality_method=None, llm_judge_model=None, llm_judge_max_examples=None, 
            llm_judge_prompt=None, llm_judge_do_sample=None, llm_judge_temperature=None,
            llm_judge_top_p=None, llm_judge_repetition_penalty=None, llm_judge_max_new_tokens=None,
            llm_judge_max_input_length=None, llm_judge_retry_attempts=None, llm_judge_validate_consistency=None,
            llm_judge_strip_cot=None, llm_judge_debug_logging=None, llm_judge_load_in_8bit=None,
            llm_judge_load_in_4bit=None):
        self._rag_logger.info('Running RAG with base model ' + str(rag_base_model_name) + ', using ' + \
            ('LangChain' if rag_type == 'lc' else 'HuggingFace') + ' libs.')
        
        if rag_type == "hf":
            self.rag_runner = self.hfragInst #HuggingFaceRag(overrides=self.overrides)
        elif rag_type == "lc":
            #self.lcragInst.prompt = self.lcragInst.prompt_templates.get(rag_prompt_type)
            #print('prompt', self.lcragInst.prompt)
            self.rag_runner = self.lcragInst #LangChainRag(overrides=self.overrides)
        else:
            raise ValueError(f"Unsupported rag_type: {rag_type}")
        
        if rag_base_model_name is None:
            raise ValueError('RAG base model must be specified using option --rag_base_model_name')
        if rag_text is None:
            raise ValueError('RAG training data must be specified using option --rag_text')
        if rag_questions is None:
            raise ValueError('RAG questions must be specified using option --rag_questions')
                    
        if rag_type == "hf":
            output_text, rag_retrieved = self.rag_runner.run(rag_questions=rag_questions, rag_text=rag_text, rag_base_model_name=rag_base_model_name,
                rag_trained_model_path=rag_trained_model_path, rag_top_k_passages=rag_top_k_passages, rag_index_backend=rag_index_backend, 
                rag_max_input_length=rag_max_input_length, rag_max_target_length=rag_max_target_length, 
                rag_max_new_tokens=rag_max_new_tokens, rag_sample=rag_sample, rag_token=rag_token,
                rag_trust_remote_code=rag_trust_remote_code, rag_train=rag_train, rag_eval=rag_eval, 
                rag_batch_size=rag_batch_size, rag_epochs=rag_epochs, 
                rag_question_column=rag_question_column, rag_context_column=rag_context_column, 
                rag_eval_strategy=rag_eval_strategy, rag_save_steps=rag_save_steps, 
                rag_logging_steps=rag_logging_steps, rag_report_to=rag_report_to, rag_lr=rag_lr,
                rag_weight_decay=rag_weight_decay, rag_save_total_limit=rag_save_total_limit,
                rag_compute_device=rag_compute_device)
        elif rag_type == "lc":
            output_text, rag_retrieved = self.rag_runner.run(rag_questions=rag_questions, rag_text=rag_text, rag_base_model_name=rag_base_model_name,
                rag_trained_model_path=rag_trained_model_path, rag_top_k_passages=rag_top_k_passages, rag_base_url=rag_base_url, 
                rag_embedding_model=rag_embedding_model, rag_prompt_type=rag_prompt_type, rag_train=rag_train, rag_eval=rag_eval)
        
        self._rag_logger.info(f"Saving generated text into {self.generated_text_file_name}")
        with open(self.generated_text_file_name, "w", encoding="utf-8") as file:
            if isinstance(output_text, str):
                file.write(output_text + "\n")
            elif isinstance(output_text, list):
                for text in output_text:
                    file.write(str(text) + "\n")
            else:
                raise Exception(f"Unexpected generated text type: {type(output_text)} - {output_text}")

        # Normalize outputs
        rag_outputs = self._normalize_rag_outputs(
            output_text=output_text,
            retrieved_contexts=rag_retrieved,
            rag_questions=rag_questions
        )

        # Save retrieved contexts for analysis
        retrieved_contexts_file = self.report_file_prefix + '_rag_retrieved_contexts.json'
        with open(retrieved_contexts_file, 'w') as f:
            json.dump({
                "questions": rag_questions,
                "retrieved_contexts": rag_retrieved,
                "answers": output_text
            }, f, indent=2)

        self._rag_logger.info(f"Saved retrieved contexts to {retrieved_contexts_file}")
        
        # Free up memory before loading judge model
        self._rag_logger.info("=" * 70)
        self._rag_logger.info("MEMORY CLEANUP: Freeing RAG model memory before loading judge")
        self._rag_logger.info("=" * 70)

        # Call cleanup on the RAG runner
        if hasattr(self.rag_runner, 'cleanup_memory'):
            self.rag_runner.cleanup_memory()

        # Additional cleanup at this level
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._rag_logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB used")

        self._rag_logger.info("=" * 70)

        
        # evaluate RAG quality
        # Optional judge
        if llm_quality_method == "judge":
            #print('rag_outputs:', rag_outputs)    
            #print('rag_retrieved:', rag_retrieved)    
            self._rag_logger.info("Running RAG quality evaluation with LLM-as-a-Judge")
            self.judge.run(
                rag_outputs=rag_outputs,
                rag_retrieved=rag_retrieved,
                llm_quality_method=llm_quality_method,
                llm_judge_model=llm_judge_model,
                llm_judge_max_examples=llm_judge_max_examples,
                llm_judge_prompt=llm_judge_prompt,
                llm_judge_do_sample=llm_judge_do_sample,
                llm_judge_temperature=llm_judge_temperature,
                llm_judge_top_p=llm_judge_top_p,
                llm_judge_repetition_penalty=llm_judge_repetition_penalty,
                llm_judge_max_new_tokens=llm_judge_max_new_tokens,
                llm_judge_max_input_length=llm_judge_max_input_length,
                llm_judge_retry_attempts=llm_judge_retry_attempts,
                llm_judge_validate_consistency=llm_judge_validate_consistency,
                llm_judge_strip_cot=llm_judge_strip_cot,
                llm_judge_debug_logging=llm_judge_debug_logging,
                llm_judge_load_in_8bit=llm_judge_load_in_8bit,
                llm_judge_load_in_4bit=llm_judge_load_in_4bit
            )
        
        return output_text

'''
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS index with cosine similarity
vectorstore = FAISS.from_documents(docs, embeddings, normalize_L2=True)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

'''