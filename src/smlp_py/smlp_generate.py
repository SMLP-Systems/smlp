# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.
 
from smlp_py.transformer import Transformer, LanguageModel
from smlp_py.smlp_text import SmlpText


'''
Module Purpose:

The SmlpGenerate module is a component of the SMLP system for generating text using a Transformer-based model 
or a bigram language model. It integrates:
-- A custom Transformer class (seq-to-seq model),
-- A character-level LanguageModel (bigram or Transformer-based),
-- NLP preprocessing using a provided nlp_inst.
It supports reading raw text, preprocessing it into tokens, training a model on those tokens, and generating 
new text based on learned patterns.

Main Roles:

-- Coordinates text preprocessing, vocabulary setup, and model training.
-- Provides a method smlp_generate() that:
   -- Reads and preprocesses input text,
   -- Creates vocabulary tokens,
   -- Trains a LanguageModel,
   -- Generates and saves text.
   
Workflow Summary

-- Text Preprocessing:
Load raw .txt file (e.g., Shakespeare).
Clean and tokenize it using Spacy-based NLP instance.
Build vocabulary and map tokens to integers.

-- Model Training:
Train either a Transformer or Bigram model on encoded text using teacher forcing.
Use cross-entropy loss to optimize prediction of next character given context.

-- Text Generation:
Begin from an empty context.
Generate new tokens sequentially (up to 500 tokens).
Decode tokens into characters and save as output.

-- Output Files
Generated text is written to:
{report_file_prefix}_generated_text.txt

-- Additional Notes
Uses character-level modeling, not word-level.
Supports switching between custom Transformer and Bigram model.
Training duration is short (default 30 iterations), but designed to be easily extended.
NLP pipeline must be initialized and passed before calling smlp_generate().
'''

class SmlpGenerate:
    def __init__(self):
        '''
        Instantiates a Transformer and LanguageModel with predefined hyperparameters
        '''
        self._gen_logger = None
        self._transformer = Transformer(src_vocab_size = 50, tgt_vocab_size = 50, d_model = 32, num_heads = 8, num_layers = 6,
            d_ff = 256, max_seq_length = 10, dropout = 0.1)
        self._langmodel = LanguageModel()

    # set logger from a caller script
    def set_logger(self, logger):
        self._gen_logger = logger 
        self._langmodel.set_logger(logger)
        self._transformer.set_logger(logger)
        
    # set NLP instance for text pre-processing (used for tokenizing and cleaning input).
    def set_nlp_inst(self, nlp_inst):
        self._nlpInst = nlp_inst
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        self._langmodel.set_report_file_prefix(report_file_prefix)
        #self._transformer.set_report_file_prefix(report_file_prefix)
    
    
    # Main method to: (1) load and preprocess text, (2) build vocab, 
    #   (3) train and run the language model for text generation.
    # generate text after training a transformer model
    def smlp_generate(self, data_fname):
        nlp = self._nlpInst.create_nlp()
        path_to_file = data_fname #'./tiny_shekespeare.txt'
        with open(path_to_file, 'r', encoding='utf-8') as f:
            text = f.read()
        #text = text[0:1000]
        print('text', text[:100])
        vocab = self._nlpInst.nlp_preprocess(text, nlp); print('vocab', type(vocab)); print(vocab); 
        vocab = vocab.split(' ')
                
        self._langmodel.set_vocab_tokens(vocab)
        self._langmodel.flow(data_fname) #'./tiny_shekespeare.txt'
    