# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import os

# fastText https://fasttext.cc/docs/en/unsupervised-tutorial.html 
import fasttext

# gensim
import gensim.downloader as gs
#rom gensim.models import FastText

# ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# SMLP
from smlp_py.smlp_utils import str_to_bool, str_to_int_tuple, lists_union_order_preserving_without_duplicates
from smlp_py.smlp_mrmr import SmlpMrmr
from smlp_py.transformer import Transformer, LanguageModel
from smlp_py.smlp_nlp import SmlpNlp

# TODO add ELMo
class SmlpText:
    def __init__(self):
        self._text_logger = None
        self._nlpInst = None
        
        self._DEF_TEXT_EMBEDDING = 'bow'
        self._DEF_TEXT_COLNAME = None
        self._DEF_NGRAM_RANGE = (1, 1)
        self._DEF_USE_WORDVEC = False
        self._DEF_SAVE_WORDVEC = False
        self._DEF_WORDVEC_MODEL = None
        self._DEF_WORDVEC_DIMENSION = 150
        self._DEF_FASTTEXT_MINN = 3
        self._DEF_FASTTEXT_MAXN = 6
        
        self.text_params_dict = {
            'text_embedding': {'abbr':'text_embedding', 'default':self._DEF_TEXT_EMBEDDING, 'type':str, 
                'help':'Word embedding method (into numeric vectors) to be used for text ' + 
                    '[default {}]'.format(str(self._DEF_TEXT_EMBEDDING))},
            'text_colname': {'abbr':'text_colname', 'default':self._DEF_TEXT_COLNAME, 'type':str, 
                'help':'Name of the text column in text data ' + 
                    '[default {}]'.format(str(self._DEF_TEXT_COLNAME))},
            'ngram_range': {'abbr':'ngram_range', 'default':self._DEF_NGRAM_RANGE, 'type':str_to_int_tuple, 
                'help':'Comma separated list of two integers, to represent ngram-range tuple, to be used by vectorizer. ' + 
                    'Say value 1,3 represents tuple (1,3), and instructs vectorizer to generate text embeddings ' + 
                    'for subsequences of lengths 1 to 3 of text in each row of input text data ' + 
                    '[default {}]'.format(str(self._DEF_NGRAM_RANGE))},
            'wordvec_model': {'abbr':'wordvec', 'default':self._DEF_WORDVEC_MODEL, 'type':str, 
                'help':'Word embedding model name. Used for saving user-trained model if option save_wordvec is on,  ' + 
                    'and used for loading pre-trained model when option use_wordvec is on. In the latter case, wordvec ' +
                    'model name should contain also the path to the pre-trained vector model, while in the former case  ' +
                    'the saved model will be written in the output directory [default {}]'.format(str(self._DEF_WORDVEC_MODEL))},
            'use_wordvec': {'abbr':'use_wordvec', 'default':self._DEF_USE_WORDVEC, 'type':str_to_bool, 
                'help':'Should pre-trained word to vector embedding be used, or a new word embedding model should be trained? ' + 
                    'Relevant for word embeddings cwbo and skipgram [default {}]'.format(str(self._DEF_USE_WORDVEC))},
            'save_wordvec': {'abbr':'save_wordvec', 'default':self._DEF_SAVE_WORDVEC, 'type':str_to_bool, 
                'help':'Should user-trained word embedding model be saved for future re-use? Relevant for word embeddings ' + 
                    'cwbo and skipgram [default {}]'.format(str(self._DEF_SAVE_WORDVEC))},
            'wordvec_dimension': {'abbr':'wordvec_dim', 'default':self._DEF_WORDVEC_DIMENSION, 'type':int, 
                'help':'The dimension of word embedding vectors that are generated using word embedding methods ' + 
                    'specified using option "text_embedding" [default {}]'.format(self._DEF_WORDVEC_DIMENSION)},
            'fasttext_minn': {'abbr':'fasttext_minn', 'default':self._DEF_FASTTEXT_MINN, 'type':int, 
                'help':'The minn parameter to be passed to unsupervised fasttext training for generating "cbow" and "skipgram" ' +
                    'word embeddings. This parameter determines the minimal length of char ngrams that are considered in training ' + 
                    'fasttext word vector embeddings (representations) [default {}]'.format(self._DEF_FASTTEXT_MINN)},
            'fasttext_maxn': {'abbr':'fasttext_maxn', 'default':self._DEF_FASTTEXT_MAXN, 'type':int, 
                'help':'The maxn parameter to be passed to unsupervised fasttext training for generating "cbow" and "skipgram" ' +
                    'word embeddings. This parameter determines the maximal length of char ngrams that are considered in training ' + 
                    'fasttext word vector embeddings (representations) [default {}]'.format(self._DEF_FASTTEXT_MAXN)}
        }
        
        self.wordvec_file_prefix = None
       
    # set logger from a caller script
    def set_logger(self, logger):
        self._text_logger = logger 
        
    def set_nlp_inst(self, nlp_inst):
        self._nlpInst = nlp_inst
    
    # pre-trained or user-trained vector embegging model file name prefix
    def set_wordvec_file_prefix(self, wordvec_file_prefix):
        self.wordvec_file_prefix = wordvec_file_prefix
    
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        
    def set_text_params(self, text_colname:str, text_embedding:str, ngram_range:tuple, wordvec_model:str,
            wordvec_dimension:int, wordvec_file_prefix:str, use_wordvec:bool, save_wordvec:bool, 
            fasttext_minn:int, fasttext_maxn:int, analytics_task:str=None):
        self.text_embedding = text_embedding
        self.text_colname = text_colname
        self.ngram_range = ngram_range
        self.wordvec_model = wordvec_model
        self.wordvec_dimension = wordvec_dimension
        self.wordvec_file_prefix = wordvec_file_prefix
        self.use_wordvec = use_wordvec
        self.save_wordvec = save_wordvec
        self.analytics_task = analytics_task
        self.fasttext_minn = fasttext_minn
        self.fasttext_maxn = fasttext_maxn
        
        if self.text_embedding == 'word2vec' and self._nlpInst.nlp_spacy_core not in ['en_core_web_md', 'en_core_web_lg']:
            raise Exception('When word embedding word2vec is used, spacy_core must be set to en_core_web_md or en_core_web_lg')
        if self.wordvec_model is None and self.use_wordvec:
            raise Exception('When option self.use_wordvec is set to True, option wordvec_model must be specified')
        if self.use_wordvec and self.text_embedding not in ['cbow', 'skipgram']:
            raise Exception('Option use_wordvec is supported only for word embeddings "cbow" and "skipgram" (from fasttext package)')
    
    # gnerate file name into which to store saved word emebediings cbow and skipgram generated with fasttext
    def wordvec_filename(self, wordvec_algo, response=None):
        assert self.wordvec_file_prefix is not None
        if response is None:
            return self.wordvec_file_prefix + '_' + wordvec_algo + '.bin'
        else:
            return self.wordvec_file_prefix + '_' + str(response) + '_' + wordvec_algo + '.bin'
    
    # The Bag Of Words (BOW) and TfIdf vectorization methods using vectorizers CountVectorizer and 
    # TfidfVectorizer from sklearn package. These vectorizers compute word embeddings by checking
    # word occurrence counts and (in case of TfIDF) apply some transformations to compute numeric
    # vector representations of words and n-grams. no "vector model training" is performed, and 
    # vector embedding do not have properties that similar words are associated with similar numeric 
    # vectors. These encodings are useful say for log/monitoring data analysis where event occurrences
    # are analysed and there is no meaning for similarity of event names or for their natural language 
    # (say English) meaning.
    # TODO: add one-hot encoding, can be useful for trace log analysis -- support encoding of a subset of vocabulary
    def text_vectorize_sklearn(self, X_train:pd.DataFrame, X_test:pd.DataFrame, vectorizer:str):
        self._text_logger.info('Creating word embedding using method ' + str(self.text_embedding))
        if self.text_embedding == 'bow':
            v = CountVectorizer(ngram_range=self.ngram_range)
        elif self.text_embedding == 'tfidf':
            v = TfidfVectorizer(ngram_range=self.ngram_range)
            '''
            v.fit_transform(X_train) #X_train.values
            print(v.vocabulary_); print(v.get_feature_names_out())
            for word in v.get_feature_names_out():
                index = v.vocabulary_.get(word)
                print(f"{word} {v.idf_[index]}")
            '''
        else:
            raise Exception('Unsupported vectorixer ' + str(self.text_embedding) + ' in function text_embed')

        X_train_cv = v.fit_transform(X_train.values)
        X_test_cv = v.transform(X_test.values); #print(v.vocabulary_)
        #print('X_train_cv (1)', type(X_train_cv), '\n', pd.DataFrame(X_train_cv.toarray()).head())
        
        text_colnames = list(v.get_feature_names_out())
        #print('train and test data shapes', X_train_cv.shape, X_test_cv.shape)
        assert len(text_colnames) == X_train_cv.shape[1]
        assert len(text_colnames) == X_test_cv.shape[1]

        X_train_cv = pd.DataFrame(X_train_cv.toarray())
        X_test_cv = pd.DataFrame(X_test_cv.toarray())
        feat_names_map = dict(zip(range(len(text_colnames)), text_colnames))
        X_train_cv.rename(columns=feat_names_map, inplace=True);
        X_test_cv.rename(columns=feat_names_map, inplace=True);
        X_train_cv.set_index(X_train.index, inplace=True)
        X_test_cv.set_index(X_test.index, inplace=True)

        #print('X_train and X_test after converting to DF')
        #print(X_train.index); print(X_test.index);
        #print(X_train_cv); print(X_test_cv)

        return X_train_cv, X_test_cv, text_colnames
    
    
    # text embedding (vectorization) using Spacy's pre-trained core language models (word2vec embedding)
    # and with gensim's pre-trained language models. These packages have API functions to calculate
    # a numeric vector for a text fragment (in our case, one row in the text data column).
    # These packages both support two versions of word2vec: CNOW (Continuous BOW) and Skip Gram.
    #
    # fatsText:
    # Use maxn=0 to avaoid usage of subword information for generating word embedding, including for 
    # unseen words in vocabolary. fastText can build word2vec word embeddings CBOW and Skip Gram. 
    # These embeddings require unsuperwised training of a vector model from text data.
    # fastText also has pre-trained vector models: fasttext.load_model('path/cc.en.300.bin')
    # Most useful API functions: 
    # vect.get_word_vector("the"), 
    # vect.get_analogies('Luki', 'pizza', 'Tror')
    # vect2.get_nearest_neighbors('Luki')
    #
    # TODO: distinguish between CBOW and Skip-gram options within word2vec
    # TODO: currently each row (text fragment) is encoded as one row. Instead, assuming there is
    # an API function to extract numeric word for each word or ngram, we could generate multi-column,
    # per word/per ngram encoding in the style used in BOW and TkIdf encodings using sklearn vectorizers.
    # This will enable to analyse impact of each ngram on final model or final analysis like root-causing.
    # TODO: enabling saving and re-loading user trainted model with fasttext and other (?)
    def text_embed(self, X_train:pd.DataFrame, X_test:pd.DataFrame, nlp, vectorizer:str, dim, minn, maxn):
        self._text_logger.info('Creating word embedding using method ' + str(self.text_embedding))
        
        print('X_train before vectors\n', X_train)
        if self.text_embedding == 'word2vec':
            X_train_cv = X_train.apply(lambda x: nlp(x).vector)
            X_test_cv = X_test.apply(lambda x: nlp(x).vector)
        elif self.text_embedding == 'glove':
            # https://github.com/piskvorky/gensim-data -- to download word embedings
            #wv = gs.load("word2vec-google-news-300")
            wv = gs.load(self.wordvec_model) #"glove-wiki-gigaword-50"
            X_train_cv = X_train.apply(lambda x: wv.get_mean_vector(x, pre_normalize=False))
            X_test_cv = X_test.apply(lambda x: wv.get_mean_vector(x, pre_normalize=False))
        elif self.text_embedding in ['cbow', 'skipgram']:
            if self.use_wordvec:
                vect = fasttext.load_model(self.wordvec_model)
            else:
                X = pd.concat([X_train, X_test], axis=0); print(X_train, X_test, X)
                tmp_fasttext_data_file = self.report_file_prefix + '_wordvec_tmp.csv'
                print('tmp_fasttext_data_file', tmp_fasttext_data_file)
                X.to_csv(tmp_fasttext_data_file, columns=[X.name], header=None, index=False)
                vect = fasttext.train_unsupervised(tmp_fasttext_data_file, self.text_embedding, minn=minn, maxn=maxn, dim=dim)
                os.remove(tmp_fasttext_data_file)
                if self.save_wordvec:
                    wordvec_file = self.wordvec_filename(self.text_embedding, response=None)
                    self._text_logger.info('Seving word embedding to numeric vectors in file ' + str(wordvec_file))
                    vect.save_model(wordvec_file)
            
            def get_mean_vector(text_row):
                return np.mean([vect.get_word_vector(str(e)) for e in nlp(text_row)], axis=0)
            
            X_train_cv = X_train.apply(lambda x: get_mean_vector(x))
            X_test_cv = X_test.apply(lambda x: get_mean_vector(x))
        else:
            raise Exception('Unsupported vectorizer ' + str(self.text_embedding) + ' in function text_vectorize')
        
        print('after embedding\nX_train_cv\n', X_train_cv, '\nX_test_cv\n', X_test_cv) 
        print('X_train_cv', X_train_cv.shape, type(X_train_cv)); print(X_train_cv)
        X_train_cv = pd.DataFrame(np.stack(X_train_cv, axis=0))
        X_test_cv = pd.DataFrame(np.stack(X_test_cv, axis=0))
        print('X_train_cv', X_train_cv.shape, type(X_train_cv)); print(X_train_cv)
        text_colnames = ['_'.join([self.text_embedding, str(i)]) for i in range(0, X_train_cv.shape[1])]
        
        feat_names_map = dict(zip(range(len(text_colnames)), text_colnames))
        X_train_cv.rename(columns=feat_names_map, inplace=True);
        X_test_cv.rename(columns=feat_names_map, inplace=True);
        X_train_cv.set_index(X_train.index, inplace=True)
        X_test_cv.set_index(X_test.index, inplace=True)
        return X_train_cv, X_test_cv, text_colnames

    
        
    def process_text(self, df, feat_names:list[str], resp_names:list[str]):
        self._text_logger.info('Processing test data: start')
        
        nlp = self._nlpInst.create_nlp()
        
        print('df in process_text (1) \n', df); print(type(df)); print(feat_names, resp_names)
        df[self.text_colname] = df[self.text_colname].apply(lambda t: self._nlpInst.nlp_preprocess(t, nlp))
        print('df in process_text (2) \n', df); print(type(df)); print(df.columns.tolist())
        print(df.head())
        test_size = 0.1
        if (test_size > 0 and test_size < 1) or test_size > 1:
            # split can be performed
            X_train, X_test, y_train, y_test = train_test_split(
                df[self.text_colname],
                df[[ft for ft in feat_names if ft !=self.text_colname]+resp_names],
                #stratify=df[resp_name],
                random_state=10,
                test_size=0.2)
        else:
            X_train = df[self.text_colname]
            y_train = df[[ft for ft in feat_names if ft !=self.text_colname]+resp_names]
            X_test = pd.DataFrame(columns=[X_train.name])
            y_test = pd.DataFrame(columns=y_train.columns.tolist())
        
        print('X train and test before vectorization -- shapes:', X_train.shape, X_test.shape); 
        print('X train and test before vectorization -- types:', type(X_train), type(X_test)); 
        print(X_train); 
        
        print('y train and test before vectorization -- shapes', y_train.shape, y_test.shape)
        print('y train and test before vectorization -- types', type(y_train), type(y_test))
        print('y_train df\n', y_train); 
        
        if self.text_embedding in ['bow', 'tfidf']:
            X_train, X_test, new_feat_names = self.text_vectorize_sklearn(X_train, X_test, self.text_embedding)
        elif self.text_embedding in ['word2vec', 'glove', 'cbow', 'skipgram']:
            X_train, X_test, new_feat_names = self.text_embed(X_train, X_test, nlp, self.text_embedding, 
                 self.wordvec_dimension, self.fasttext_minn, self.fasttext_maxn)
        else:
            raise Exception('Unsupported vectorizer ' + str(self.text_embedding) +
                            ' in function test_classify')
        
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        print('after vectorization -- shapes:', X_train.shape, X_test.shape); 
        print('after vectorization -- types:', type(X_train), type(X_test)); 
        print(X_train)
        print('new_feature names', new_feat_names)
        
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        print('y vectors after converting to df -- shapes', y_train.shape, y_test.shape)
        print('y vectors after converting to df -- types', type(y_train), type(y_test))
        print('y_train df\n', y_train); print('y_test df\n', y_test); 
        
        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)
        #T = pd.concat([T_train, T_test], axis=0)
        print('full X and y -- shapes:', X.shape, y.shape)
        print('full X and y -- types:', type(X), type(y))
        print('X\n', X); print('y\n', y)
        print('concat\n', pd.concat([X, y], axis=1))
        
        self._text_logger.info('Processing test data: end')
        return pd.concat([X, y], axis=1), new_feat_names
    
           