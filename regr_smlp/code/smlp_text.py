# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import os

# Spacy
import spacy
from spacy.symbols import ORTH
from spacy.tokens import Span # a fragment of a text
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS

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
from smlp_py.transformer import Transformer, LanguageModel #MultiHeadAttention

# TODO add ELMo
class SmlpText:
    def __init__(self):
        self._doepy_logger = None
        self._mrmrInst = SmlpMrmr() 
        
        self._DEF_NLP_SPACY_BLANK = False
        self._DEF_NLP_SPACY_LEMMATIZER = True
        self._DEF_NLP_SPACY_TAGGER = True
        self._DEF_NLP_SPACY_RULER = True
        self._DEF_NLP_SPACY_SENTER = True
        self._DEF_NLP_SPACY_PARSER = False
        self._DEF_NLP_SPACY_TOK2VEC = False
        self._DEF_NLP_SPACY_NER = False
        self._DEF_NLP_SPACY_MORPHOLOGIZER = False
        self._DEF_TEXT_EMBEDDING = 'bow'
        self._DEF_TEXT_COLNAME = None
        self._DEF_NGRAM_RANGE = (1, 1)
        self._DEF_SPACY_CORE = 'en_core_web_sm'
        self._DEF_NLP_USE_WORDVEC = False
        self._DEF_NLP_SAVE_WORDVEC = False
        
        self.nlp_params_dict = {
            'nlp_blank': {'abbr':'nlp_blank', 'default':self._DEF_NLP_SPACY_BLANK, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should a blank instance be ' + 
                    'created first and required pipe stage added on demand? [default {}]'.format(str(self._DEF_NLP_SPACY_BLANK))},
            'nlp_lemmatizer': {'abbr':'nlp_lemmatizer', 'default':self._DEF_NLP_SPACY_LEMMATIZER, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should the lemmatizer be included ' + 
                    'as part of text preprocessing pipeline? [default {}]'.format(str(self._DEF_NLP_SPACY_LEMMATIZER))},
            'nlp_tagger': {'abbr':'nlp_tagger', 'default':self._DEF_NLP_SPACY_TAGGER, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should the tagger be included ' + 
                    'as part of text preprocessing pipeline? [default {}]'.format(str(self._DEF_NLP_SPACY_TAGGER))},
            'nlp_ruler': {'abbr':'nlp_ruler', 'default':self._DEF_NLP_SPACY_RULER, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should the attribute_ruler be included ' + 
                    'as part of text preprocessing pipeline? [default {}]'.format(str(self._DEF_NLP_SPACY_RULER))},
            'nlp_senter': {'abbr':'nlp_senter', 'default':self._DEF_NLP_SPACY_SENTER, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should the sentencizer be included ' + 
                    'as part of text preprocessing pipeline? [default {}]'.format(str(self._DEF_NLP_SPACY_SENTER))},
            'nlp_parser': {'abbr':'nlp_parser', 'default':self._DEF_NLP_SPACY_PARSER, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should the centencizer be included ' + 
                    'as part of text preprocessing pipeline? [default {}]'.format(str(self._DEF_NLP_SPACY_PARSER))},
            'nlp_tok2vec': {'abbr':'nlp_tok2vec', 'default':self._DEF_NLP_SPACY_TOK2VEC, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should the tok2vec be included ' + 
                    'as part of text preprocessing pipeline? [default {}]'.format(str(self._DEF_NLP_SPACY_TOK2VEC))},
            'nlp_ner': {'abbr':'nlp_ner', 'default':self._DEF_NLP_SPACY_NER, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should the named entity recognition be included ' + 
                    'as part of text preprocessing pipeline? [default {}]'.format(str(self._DEF_NLP_SPACY_NER))},
            'nlp_morphologizer': {'abbr':'nlp_morphologizer', 'default':self._DEF_NLP_SPACY_MORPHOLOGIZER, 'type':str_to_bool, 
                'help':'When creating NLP instance with Spacy package, should the morphologizer be included ' + 
                    'as part of text preprocessing pipeline? [default {}]'.format(str(self._DEF_NLP_SPACY_MORPHOLOGIZER))},
            
        }
        
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
            'spacy_core': {'abbr':'spacy_core', 'default':self._DEF_SPACY_CORE, 'type':str, 
                'help':'Spacy core pre-trained language model; it should be downloaded using a command like:  ' + 
                    '"python -m spacy download en_core_web_sm", where "en_core_web_sm" is Spacy small English ' +
                    'language model. Two more English language models with medium and large sizes are respectively ' +
                     '"en_core_web_md" and "en_core_web_lg", and these two also contain pre-trained word embeddings ' +
                    '[default {}]'.format(str(self._DEF_SPACY_CORE))},
            'wordvec_model': {'abbr':'wordvec', 'default':None, 'type':str, 
                'help':'Word embedding model name. Used for saving user-trained model if option save_wordvec is on,  ' + 
                    'and used for loading pre-trained model when option use_wordvec is on, In the latter case, wordvec ' +
                    'model name should contain also the path to the pre-trained vector model, while in the former case  ' +
                    'the saved model will be written in the output directory [default {}]'.format(str(self._DEF_SPACY_CORE))},
            'use_wordvec': {'abbr':'use_wordvec', 'default':self._DEF_NLP_USE_WORDVEC, 'type':str_to_bool, 
                'help':'Should pre-trained word to vector embedding be used, or a new word embedding model should be trained? ' + 
                    'Relevant for word embeddings cwbo and skipgram [default {}]'.format(str(self._DEF_NLP_USE_WORDVEC))},
            'save_wordvec': {'abbr':'save_wordvec', 'default':self._DEF_NLP_SAVE_WORDVEC, 'type':str_to_bool, 
                'help':'Should user-trained word embedding model be saved for future re-use? Relevant for word embeddings ' + 
                    'cwbo and skipgram [default {}]'.format(str(self._DEF_NLP_SAVE_WORDVEC))},
            
        } | self.nlp_params_dict
        
        self.wordvec_file_prefix = None
       
    # set logger from a caller script
    def set_logger(self, logger):
        self._text_logger = logger 
        self._mrmrInst.set_logger(logger)
    
    # pre-trained or user-trained vector embegging model file name prefix
    def set_wordvec_file_prefix(self, wordvec_file_prefix):
        self.wordvec_file_prefix = wordvec_file_prefix
    
    def set_report_file_prefix(self, wordvec_file_prefix):
        self.report_file_prefix = wordvec_file_prefix
       
    def set_text_params(self, nlp_blank:bool, nlp_lemmatizer:bool, nlp_tagger:bool, nlp_ruler:bool, 
            nlp_senter:bool, nlp_parser:bool, nlp_tok2vec:bool, nlp_ner:bool, nlp_morphologizer:bool,
            text_colname:str, text_embedding:str, ngram_range:tuple, spacy_core:str, wordvec_model:str,
            wordvec_file_prefix:str, use_wordvec:bool, save_wordvec:bool):
        self.nlp_blank = nlp_blank
        self.nlp_lemmatizer = nlp_lemmatizer
        self.nlp_tagger = nlp_tagger
        self.nlp_ruler = nlp_ruler
        self.nlp_senter = nlp_senter
        self.nlp_parser = nlp_parser
        self.nlp_tok2vec = nlp_tok2vec
        self.nlp_ner = nlp_ner
        self.nlp_morphologizer = nlp_morphologizer
        self.text_embedding = text_embedding
        self.text_colname = text_colname
        self.ngram_range = ngram_range
        self.spacy_core = spacy_core
        self.wordvec_model = wordvec_model
        self.wordvec_file_prefix = wordvec_file_prefix
        self.use_wordvec = use_wordvec
        self.save_wordvec = save_wordvec
        print('self.wordvec_model', self.wordvec_model); print('self.wordvec_file_prefix', self.wordvec_file_prefix)
        #assert False
        
        if self.text_embedding == 'word2vec' and self.spacy_core not in ['en_core_web_md', 'en_core_web_lg']:
            raise Exception('When word embedding word2vec is used, spacy_core must be set to en_core_web_md or en_core_web_lg')
            
    def wordvec_filename(self, wordvec_algo, response=None):
        assert self.wordvec_file_prefix is not None
        if response is None:
            return self.wordvec_file_prefix + '_' + wordvec_algo
        else:
            return self.wordvec_file_prefix + '_' + str(response) + '_' + wordvec_algo
        
    # blank=True and opt_load=True version does not work, due to an issue in spacy.
    # Suggested fix like installing spacy-lookups-data does not work either:
    # https://github.com/explosion/spaCy/discussions/9512
    # A workarroud is to use blank=True and opt_load=True, but then there are
    # issues with senter and morpholizer.
    # Currently one should use blank=False, and then disable stages that are not
    # used: an observation suggests that for a better performance one should not
    # include in nlp the pipe stages (nlp_pipe) that are not used -- see:
    # https://github.com/explosion/spaCy/discussions/8402
    def create_nlp(self, blank=False, spacy_core_web_name:str='en_core_web_sm', lemmatizer=True, tagger=True, 
            ruler=True, senter=True, parser=False, tok2vec=False, ner=False, morphologizer=False):
        assert spacy_core_web_name in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

        #print("Create document from text")
        opt1 = False # does not work due to internal issue in spacy mentioned above
        if blank:
            if opt1:
                nlp = spacy.blank('en')
                nlp.add_pipe('senter')
                nlp.add_pipe('lemmatizer')
                nlp.add_pipe('tagger')
                nlp.add_pipe('attribute_ruler', config={'validate':True})
            else:
                source_nlp = spacy.load(spacy_core_web_name) #"en_core_web_sm"
                # source_nlp's nlp_pipe is ['tok2vec', 'tagger', 'parser',
                # 'attribute_ruler', 'lemmatizer', 'ner']; thus it does not
                # include 'senter' and "morphologizer" from the list below.
                print('nlp_pipe', source_nlp.pipe_names)
                nlp = spacy.blank('en')
                if ner:
                    nlp.add_pipe('ner', source=source_nlp)
                if senter:
                    nlp.add_pipe('senter')
                    #nlp.add_pipe('sentencizer')
                if lemmatizer:
                    print('adding lemmatizer')
                    nlp.add_pipe('lemmatizer', source=source_nlp)
                if tagger:
                    print('adding tagger')
                    nlp.add_pipe('tagger', source=source_nlp)
                if ruler:
                    print('adding ruler')
                    nlp.add_pipe('attribute_ruler', config={'validate':True})
                if parser:
                    nlp.add_pipe('parser', source=source_nlp)
                if tok2vec:
                    nlp.add_pipe('tok2vec', source=source_nlp)
                if morphologizer:
                    nlp.add_pipe("morphologizer")
        else:
            nlp = spacy.load(spacy_core_web_name) #'en_core_web_sm'
            # nlp's nlp_pipe is ['tok2vec', 'tagger', 'parser',
            # 'attribute_ruler', 'lemmatizer', 'ner']; thus it does not
            # include 'senter' and "morphologizer" from the list below.
            print(nlp.pipe_names); print('senter' in nlp.pipe_names)
            if not ner and "ner" in nlp.pipe_names:
                nlp.disable_pipe("ner")
            print(nlp.pipe_names); print('senter' in nlp.pipe_names)
            if not senter and "senter" in nlp.pipe_names:
                nlp.disable_pipe("senter")
            elif senter and not "senter" in nlp.pipe_names:
                try:
                    nlp.add_pipe("senter")
                except:
                    print('senter already exists in the pipeline')
                    try:
                        nlp.add_pipe("sentencizer")
                    except:
                        print('sentencizer already exists in the pipeline')
            if not lemmatizer and "lemmatizer" in nlp.pipe_names:
                nlp.disable_pipe("lemmatizer")
            if not tagger and "tagger" in nlp.pipe_names:
                nlp.disable_pipe("tagger")
            if not ruler and "attribute_ruler" in nlp.pipe_names:
                nlp.disable_pipe("attribute_ruler")
            if not parser and "parser" in nlp.pipe_names:
                nlp.disable_pipe("parser")
            if not tok2vec and "tok2vec" in nlp.pipe_names:
                nlp.disable_pipe("tok2vec")
            if not morphologizer and "morphologizer" in nlp.pipe_names:
                nlp.disable_pipe("morphologizer")
            elif morphologizer and not "morphologizer" in nlp.pipe_names:
                #morph = nlp.create_pipe("morphologizer")
                #morph.add_label('dummy')
                #print('here!!!!!!!!!!!!')
                nlp.add_pipe("morphologizer") #, name='dummy'
        #nlp.initialize()
        print('nlp_pipe', nlp.pipe_names)
        return nlp


    # TODO extend this function to support more text processing heuristics
    # Currently we just apply lemmatization and drop the stop words and
    # punctuation marks as well as tokens of categories ['SPACE', 'X', 'PUNCT'].
    # Examples of stop words: to, for, over, a, from, had, not.
    # STOP_WORDS from spacy.lang.en.stop_words contains all English stop words.
    # For chat-bot/Q&R system, sentiment classification, and language
    # translation tasks, removing stop words could be a bad idea,
    # e.g. dropping stop word "not" in sentiment classification task.
    def preprocess(self, text, nlp):
        doc = nlp(text)

        '''
        # debug code: display which tokens will be dropped
        print("++++++++++++++++ Drop irrelevant tokens")
        for token in doc:
            print(token, 'pos_', token.pos_, 'is_stop', token.is_stop, 'is_punc', token.is_punct)
        '''
        '''
        # examples of usefull text preprocessing steps using regular expressions re. See regex101.com site
        text = re.sub(r'^[\w\s\']', ' ', text) -- replace punctuations, newlines with space
        text = re.sub(r'[ \n]+', ' ', text) -- replace repeated spaces with space
        text = text.strip().lower() -- drop leading and traiiling (?) spaces and make into lower case
        '''
        # TODO: are conditions token.is_punct and token.pos_ == 'PUNCT' equivalent?
        # "etc" is of category 'X', thus not covered by is_punct/is_stop criteria
        # "  " is of category 'SPACE" and not covered by is_punct/is_stop criteria
        def filter_irrelevant(token):
            return token.is_stop or \
                token.is_punct or \
                token.pos_ in ['SPACE', 'X', 'PUNCT'] 

        no_stop_words = [token.lemma_ for token in doc if not filter_irrelevant(token)]
        #print(no_stop_words)

        '''
        counts = doc.count_by(spacy.attrs.POS)
        print('pos statistics:')
        for k, v in counts.items():
            print(doc.vocab[k].text, " | ", v)
        '''
        return ' '.join(no_stop_words)

    # The Bag Of Words (BOW) and TfIdf vectorization methods using vectorizers CountVectorizer and 
    # TfidfVectorizer from sklearn package. These vectorizers compute word embeddings by checking
    # word occurrence counts and (in case of TfIDF) apply some transformations to compute numeric
    # vector representations of words and n-grams. no "vector model training" is performed, and 
    # vector embedding do not have properties that similar words are associated with similar numeric 
    # vectors. These encodings are useful say for log/monitoring data analysis where event occurrences
    # are analysed and there is no meaning for similarity of event names or for their natural language 
    # (say English) meaning.
    # TODO: add one-hot encoding, can be useful for trace log analysis -- support encoding of a subset of vocabulary
    def text_vectorize(self, X_train:pd.DataFrame, X_test:pd.DataFrame, vectorizer:str, ngram_range=(1,2)):
        print('Embedding text data into numeric vectors')
        #print('X_train\n', X_train)
        if vectorizer == 'bow':
            v = CountVectorizer(ngram_range=ngram_range)
        elif vectorizer == 'tfidf':
            v = TfidfVectorizer(ngram_range=ngram_range)
            '''
            v.fit_transform(X_train) #X_train.values
            print(v.vocabulary_); print(v.get_feature_names_out())
            for word in v.get_feature_names_out():
                index = v.vocabulary_.get(word)
                print(f"{word} {v.idf_[index]}")
            '''
        else:
            raise Exception('Unsupported vectorixer ' + str(vectorizer) + ' in function text_embed')

        X_train_cv = v.fit_transform(X_train.values)
        X_test_cv = v.transform(X_test.values); #print(v.vocabulary_); exit()
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
    # unseen words in vocabolary
    # fastText can build word2vec word embeddings CBOW and Skip Gram. These embeddings require 
    # unsuperwised training of a vector model from text data.
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
    # TODO: enabling saving and re-loading user trainted model with fattext and other (?)
    def text_embed(self, X_train:pd.DataFrame, X_test:pd.DataFrame, nlp, vectorizer:str, minn=3, maxn=6, dim=300):
        if self.save_wordvec or self.use_wordvec:
            wordvec_file = self.wordvec_filename(vectorizer, response=None)
        if vectorizer == 'word2vec':
            X_train_cv = X_train.apply(lambda x: nlp(x).vector)
            X_test_cv = X_test.apply(lambda x: nlp(x).vector)
        elif vectorizer == 'glove':
            # https://github.com/piskvorky/gensim-data -- to download word embedings
            #wv = gs.load("word2vec-google-news-300")
            wv = gs.load(self.wordvec_model) #"glove-wiki-gigaword-50"
            X_train_cv = X_train.apply(lambda x: wv.get_mean_vector(x, pre_normalize=False))
            X_test_cv = X_test.apply(lambda x: wv.get_mean_vector(x, pre_normalize=False))
        elif vectorizer in ['cbow', 'skipgram']:
            if self.use_wordvec:
                vect = fasttext.load_model(self.wordvec_model)
            else:
                X = pd.concat([X_train, X_test], axis=0); print(X_train, X_test, X)
                tmp_fasttext_data_file = self.report_file_prefix + '_wordvec_tmp.csv'
                print('tmp_fasttext_data_file', tmp_fasttext_data_file)
                X.to_csv(tmp_fasttext_data_file, columns=[X.name], header=None, index=False) #header=None, index=False)
                vect = fasttext.train_unsupervised(tmp_fasttext_data_file, vectorizer, minn=minn, maxn=maxn, dim=dim)
                os.remove(tmp_fasttext_data_file)
                if self.save_wordvec:
                    wordvec_model_file = self.report_file_prefix + '_' + self.wordvec_model
                    print('Saving trained wordvec model to file ' + self.wordvec_file_prefix)
                    vect.save_model(self.wordvec_file_prefix)
            def get_mean_vector(text_row):
                #[print(token, vect.get_word_vector(str(token))) for token in nlp(text_row)]
                return np.mean([vect.get_word_vector(str(e)) for e in nlp(text_row)], axis=0)
            
            X_train_cv = X_train.apply(lambda x: get_mean_vector(x))
            X_test_cv = X_test.apply(lambda x: get_mean_vector(x))
        else:
            raise Exception('Unsupported vectorizer ' + str(vectorizer) + ' in function text_vectorize')
        
        print('after embedding\nX_train_cv\n', X_train_cv, '\nX_test_cv\n', X_test_cv) 
        print('X_train_cv', X_train_cv.shape, type(X_train_cv)); print(X_train_cv)
        X_train_cv = pd.DataFrame(np.stack(X_train_cv, axis=0))
        X_test_cv = pd.DataFrame(np.stack(X_test_cv, axis=0))
        print('X_train_cv', X_train_cv.shape, type(X_train_cv)); print(X_train_cv)
        text_colnames = ['_'.join([vectorizer, str(i)]) for i in range(0, X_train_cv.shape[1])]
        
        feat_names_map = dict(zip(range(len(text_colnames)), text_colnames))
        X_train_cv.rename(columns=feat_names_map, inplace=True);
        X_test_cv.rename(columns=feat_names_map, inplace=True);
        X_train_cv.set_index(X_train.index, inplace=True)
        X_test_cv.set_index(X_test.index, inplace=True)
        return X_train_cv, X_test_cv, text_colnames

    # Use maxn=0 to avaoid usage of subword information for generating word embedding, including for 
    # unseen words in vocabolary
    # fastText can build word2vec word embeddings CBOW and Skip Gram. These embeddings require 
    # unsuperwised training of a vector model from text data.
    # fastText also has pre-trained vector models: fasttext.load_model('path/cc.en.300.bin')
    # Most useful API functions: 
    # vect.get_word_vector("the"), 
    # vect.get_analogies('Luki', 'pizza', 'Tror')
    # vect2.get_nearest_neighbors('Luki')
    def text_vectorize_fasttext(self, X_train:pd.DataFrame, X_test:pd.DataFrame, nlp,
           algo='skipgram', minn=3, maxn=6, dim=300, epoch=5, lr=0.05, thread=12):
        assert algo in ['cbow', 'skipgram']
        # TODO: enable model saving and loading pretrained user model
        save_model = True
        load_model = False
        if load_model:
            vect = fasttext.load_model('nlp_fasttext3.bin')
        else:
            X = pd.concat([X_train, X_test], axis=0); print(X_train, X_test, X)
            X.to_csv('/home/zurabk/smlp/repo/smlp/regr_smlp/code/nlp_fasttext3.txt', #header=None, index=False)
                      columns=[X.name], header=None, index=False)
            #df.to_csv('/mnt/c/Users/khasi/Downloads/Cleaned_Indian_Food_Dataset.csv', columns=['TranslatedInstructions'], header=None, index=False)
            vect = fasttext.train_unsupervised('/home/zurabk/smlp/repo/smlp/regr_smlp/code/nlp_fasttext3.txt', 
                algo, minn=minn, maxn=maxn, dim=3) # TODO !!!!!!!!!!!!!! pass dim
            #vect = fastText.train_unsupervised('/home/zurabk/smlp/repo/smlp/regr_smlp/code/nlp_fasttext3.txt', 
            #    algo, minn=minn, maxn=maxn, dim=dim)
            if save_model:
                vect.save_model('nlp_fasttext3.bin')
        
        print('all words', vect.words)
        #print(vect.get_word_vector("the"))

        #vect2 = FastText(vector_size=300, window=5, min_count=1, sentences='/home/zurabk/smlp/repo/smlp/regr_smlp/code/nlp_fasttext3.txt', epochs=10)
        
        '''
        print('a', vect.get_nearest_neighbors('Luki'))
        #print('b', vect.get_word_vector('Luki'))
        print('c', vect.get_analogies('Luki', 'pizza', 'Tror')) # to produce Delhi
        print('a2', vect2.get_nearest_neighbors('Luki'))
        #print('b2', vect2.get_word_vector('Luki'))
        print('c2', vect2.get_analogies('Luki', 'pizza', 'Tror')) # to produce Delhi
        '''
        #vectorizer = fasttext.load_model('/mnt/c/Users/khasi/Downloads/cc.en.300.bin')
        #print(vectorizer.get_nearest_neighbors('Luki') == vect2.get_nearest_neighbors('Luki'))
        #print(vectorizer.get_word_vector('Luki') == vect2.get_word_vector('Luki'))
        #print(vectorizer.get_analogies('Luki', 'pizza', 'Tror') == vect2.get_analogies('Luki', 'pizza', 'Tror'))
        #X_train.apply(lambda x: print(x))
        def get_mean_vector(text_row):
            #[print(token, vect.get_word_vector(str(token))) for token in nlp(text_row)]
            return np.mean([vect.get_word_vector(str(e)) for e in nlp(text_row)], axis=0)
            
        print('X_train before stack \n', X_train, '\nX_test\n', X_test)
        '''
        print('X_train[0]\n', X_train[0])
        first = []
        for i in X_train.index.tolist():
            print(i)
            for x in nlp(X_train[i]):
                print(x, type(x)); print(vect.get_word_vector(str(x)))
                if i == 0:
                    first.append(vect.get_word_vector(str(x))[1])
        #print([vect.get_word_vector(str(e)) for e in nlp(X_train[0])])
        print('first', first, np.mean(first))
        '''
        
        X_train_cv = X_train.apply(lambda x: get_mean_vector(x))
        X_test_cv = X_test.apply(lambda x: get_mean_vector(x))
        print('after embedding\nX_train_cv\n', X_train_cv, '\nX_test_cv\n', X_test_cv)   
        
        X_train_cv = pd.DataFrame(np.stack(X_train_cv, axis=0))
        X_test_cv = pd.DataFrame(np.stack(X_test_cv, axis=0))
        print('after stack\nX_train_cv\n', X_train_cv, '\nX_test_cv\n', X_test_cv) 
        print('X_train_cv', X_train_cv.shape, type(X_train_cv)); print(X_train_cv); exit()
        text_colnames = ['_'.join([algo, str(i)]) for i in range(0, X_train_cv.shape[1])]
        
        feat_names_map = dict(zip(range(len(text_colnames)), text_colnames))
        X_train_cv.rename(columns=feat_names_map, inplace=True);
        X_test_cv.rename(columns=feat_names_map, inplace=True);
        X_train_cv.set_index(X_train.index, inplace=True)
        X_test_cv.set_index(X_test.index, inplace=True)
        return X_train_cv, X_test_cv, text_colnames
        return X_train_cv, X_test_cv, text_colnames
    
        
    def process_text(self, df, feat_names:list[str], resp_names:list[str], text_colname:str, vectorizer:str, 
            ngram_range=(1,1), blank=False, spacy_core_web_name:str='en_core_web_sm', lemmatizer=True, 
            tagger=True, ruler=True, senter=False, parser=False, tok2vec=False, ner=False, morphologizer=False):
        
        if True:
            '''
            if False:
                transformer = Transformer(src_vocab_size = 500, tgt_vocab_size = 500, d_model = 512, num_heads = 8, num_layers = 6,
                    d_ff = 2048, max_seq_length = 100, dropout = 0.1)
            else:
                transformer = Transformer(src_vocab_size = 50, tgt_vocab_size = 50, d_model = 32, num_heads = 8, num_layers = 6,
                    d_ff = 256, max_seq_length = 10, dropout = 0.1)
            transformer.set_train_test_data()
            transformer.train_transformer_model()
            '''
            lang_model = LanguageModel()
            lang_model.flow('./tiny_shekespeare.txt')
            #'''
            exit()
        
        print('Pre-processing text data'); print(text_colname, vectorizer)
        nlp = self.create_nlp(blank, spacy_core_web_name, lemmatizer, tagger, ruler, senter, parser, tok2vec, ner, morphologizer)
        #df = X_train #[1:200, ]
        
        print('df in process_text (1) \n', df); print(type(df)); print(feat_names, resp_names)
        df[text_colname] = df[text_colname].apply(lambda t: self.preprocess(t, nlp))
        #df[resp_name] = df[resp_name].map({pos_val:1, neg_val:0})
        #df[feat_name] = [self.preprocess(t, nlp) for t in texts]
        print('df in process_text (2) \n', df); print(type(df)); print(df.columns.tolist())
        print(df.head())
        test_size = 0.1
        if (test_size > 0 and test_size < 1) or test_size > 1:
            # split can be performed
            X_train, X_test, y_train, y_test = train_test_split(
                df[text_colname],
                df[[ft for ft in feat_names if ft !=text_colname]+resp_names],
                #stratify=df[resp_name],
                random_state=10,
                test_size=0.2)
        else:
            X_train = df[text_colname]
            y_train = df[[ft for ft in feat_names if ft !=text_colname]+resp_names]
            X_test = pd.DataFrame(columns=[X_train.name])
            y_test = pd.DataFrame(columns=y_train.columns.tolist())
        
        print('X train and test before vectorization -- shapes:', X_train.shape, X_test.shape); 
        print('X train and test before vectorization -- types:', type(X_train), type(X_test)); 
        print(X_train); 
        
        print('y train and test before vectorization -- shapes', y_train.shape, y_test.shape)
        print('y train and test before vectorization -- types', type(y_train), type(y_test))
        print('y_train df\n', y_train); 
        
        if vectorizer in ['bow', 'tfidf']:
            X_train, X_test, new_feat_names = self.text_vectorize(X_train, X_test, vectorizer, ngram_range)
        elif vectorizer in ['word2vec', 'glove', 'cbow', 'skipgram']:
            X_train, X_test, new_feat_names = self.text_embed(X_train, X_test, nlp, vectorizer)
        elif vectorizer == 'fasttext':
            assert False
            #df = pd.read_csv('/mnt/c/Users/khasi/Downloads/Cleaned_Indian_Food_Dataset.csv')
            X_train, X_test, new_feat_names = self.text_vectorize_fasttext(X_train, X_test, nlp)
        else:
            raise Exception('Unsupported vectorizer ' + str(vectorizer) +
                            ' in function test_classify')
        
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        print('after vectorization -- shapes:', X_train.shape, X_test.shape); 
        print('after vectorization -- types:', type(X_train), type(X_test)); 
        print(X_train)
        
        #'''
        print('new_feature names', new_feat_names)
           
        #X_train = pd.DataFrame(X_train).drop(columns=[text_colname], inplace=True)
        #X_test = pd.DataFrame(X_test).drop(columns=[text_colname], inplace=True)
        
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        print('y vectors after converting to df -- shapes', y_train.shape, y_test.shape)
        print('y vectors after converting to df -- types', type(y_train), type(y_test))
        print('y_train df\n', y_train); print('y_test df\n', y_test); 
        #'''
        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)
        #T = pd.concat([T_train, T_test], axis=0)
        print('full X and y -- shapes:', X.shape, y.shape)
        print('full X and y -- types:', type(X), type(y))
        print('X\n', X); print('y\n', y)
        print('concat\n', pd.concat([X, y], axis=1))
        
        return pd.concat([X, y], axis=1), new_feat_names
    
           