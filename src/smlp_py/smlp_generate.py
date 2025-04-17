# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.
 
from smlp_py.transformer import Transformer, LanguageModel
from smlp_py.smlp_text import SmlpText

class SmlpGenerate:
    def __init__(self):
        self._gen_logger = None
        self._transformer = Transformer(src_vocab_size = 50, tgt_vocab_size = 50, d_model = 32, num_heads = 8, num_layers = 6,
            d_ff = 256, max_seq_length = 10, dropout = 0.1)
        self._langmodel = LanguageModel()

    # set logger from a caller script
    def set_logger(self, logger):
        self._gen_logger = logger 
        self._langmodel.set_logger(logger)
        self._transformer.set_logger(logger)
        
    # set NLP instance for text pre-processing
    def set_nlp_inst(self, nlp_inst):
        self._nlpInst = nlp_inst
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        self._langmodel.set_report_file_prefix(report_file_prefix)
        #self._transformer.set_report_file_prefix(report_file_prefix)
    
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
        #if False:
        #    transformer = Transformer(src_vocab_size = 500, tgt_vocab_size = 500, d_model = 512, num_heads = 8, num_layers = 6,
        #        d_ff = 2048, max_seq_length = 100, dropout = 0.1)
        #else:
        #    transformer = Transformer(src_vocab_size = 50, tgt_vocab_size = 50, d_model = 32, num_heads = 8, num_layers = 6,
        #        d_ff = 256, max_seq_length = 10, dropout = 0.1)
        #self._transformer.set_train_test_data()
        #self._transformer.train_transformer_model()
                
        self._langmodel.set_vocab_tokens(vocab)
        self._langmodel.flow(data_fname) #'./tiny_shekespeare.txt'
        
        
        '''
        self._text_logger.info('Processing test data: start')
        
        if self.analytics_task == 'generate':
            nlp = self._nlpInst.create_nlp()
            path_to_file = './tiny_shekespeare.txt'
            with open(path_to_file, 'r', encoding='utf-8') as f:
                text = f.read()
            text = text[0:1000]
            print('text', text[:100])
            vocab = self.preprocess(text, nlp); print('vocab', type(vocab)); print(vocab); 
            vocab = vocab.split(' ')
            if False:
                transformer = Transformer(src_vocab_size = 500, tgt_vocab_size = 500, d_model = 512, num_heads = 8, num_layers = 6,
                    d_ff = 2048, max_seq_length = 100, dropout = 0.1)
            else:
                transformer = Transformer(src_vocab_size = 50, tgt_vocab_size = 50, d_model = 32, num_heads = 8, num_layers = 6,
                    d_ff = 256, max_seq_length = 10, dropout = 0.1)
            transformer.set_train_test_data()
            transformer.train_transformer_model()
            
        lang_model = LanguageModel()
        lang_model.set_vocab_tokens(vocab)
        lang_model.flow('./tiny_shekespeare.txt')
        '''