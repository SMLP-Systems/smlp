# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

# Spacy
import spacy
from spacy.symbols import ORTH
from spacy.tokens import Span # a fragment of a text
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS

from smlp_py.smlp_utils import str_to_bool #, str_to_int_tuple, lists_union_order_preserving_without_duplicates

class SmlpNlp:
    def __init__(self):
        self._DEF_NLP_SPACY_BLANK = False
        self._DEF_NLP_SPACY_LEMMATIZER = True
        self._DEF_NLP_SPACY_TAGGER = True
        self._DEF_NLP_SPACY_RULER = True
        self._DEF_NLP_SPACY_SENTER = True
        self._DEF_NLP_SPACY_PARSER = False
        self._DEF_NLP_SPACY_TOK2VEC = False
        self._DEF_NLP_SPACY_NER = False
        self._DEF_NLP_SPACY_MORPHOLOGIZER = False
        self._DEF_NLP_SPACY_CORE = 'en_core_web_sm'
        
        self.nlp_blank = self._DEF_NLP_SPACY_BLANK #nlp_blank
        self.nlp_lemmatizer = self._DEF_NLP_SPACY_LEMMATIZER #nlp_lemmatizer
        self.nlp_tagger = self._DEF_NLP_SPACY_TAGGER #nlp_tagger
        self.nlp_ruler = self._DEF_NLP_SPACY_RULER #nlp_ruler
        self.nlp_senter = self._DEF_NLP_SPACY_SENTER #nlp_senter
        self.nlp_parser = self._DEF_NLP_SPACY_PARSER #nlp_parser
        self.nlp_tok2vec = self._DEF_NLP_SPACY_TOK2VEC #nlp_tok2vec
        self.nlp_ner = self._DEF_NLP_SPACY_NER #nlp_ner
        self.nlp_morphologizer = self._DEF_NLP_SPACY_MORPHOLOGIZER #nlp_morphologizer
        self.nlp_spacy_core = self._DEF_NLP_SPACY_CORE
        
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
            'nlp_spacy_core': {'abbr':'nlp_spacy_core', 'default':self._DEF_NLP_SPACY_CORE, 'type':str, 
                'help':'Spacy core pre-trained language model; it should be downloaded using a command like:  ' + 
                    '"python -m spacy download en_core_web_sm", where "en_core_web_sm" is Spacy small English ' +
                    'language model. Two more English language models with medium and large sizes are respectively ' +
                     '"en_core_web_md" and "en_core_web_lg", and these two also contain pre-trained word embeddings ' +
                    '[default {}]'.format(str(self._DEF_NLP_SPACY_CORE))}
        }
        
    # set logger from a caller script
    def set_logger(self, logger):
        self._nlp_logger = logger 

    def set_nlp_params(self, nlp_blank:bool, nlp_lemmatizer:bool, nlp_tagger:bool, nlp_ruler:bool, 
            nlp_senter:bool, nlp_parser:bool, nlp_tok2vec:bool, nlp_ner:bool, nlp_morphologizer:bool,
            nlp_spacy_core:str):
        self.nlp_blank = nlp_blank
        self.nlp_lemmatizer = nlp_lemmatizer
        self.nlp_tagger = nlp_tagger
        self.nlp_ruler = nlp_ruler
        self.nlp_senter = nlp_senter
        self.nlp_parser = nlp_parser
        self.nlp_tok2vec = nlp_tok2vec
        self.nlp_ner = nlp_ner
        self.nlp_morphologizer = nlp_morphologizer
        self.nlp_spacy_core = nlp_spacy_core
    
    # blank=True and opt_load=True version does not work, due to an issue in spacy.
    # Suggested fix like installing spacy-lookups-data does not work either:
    # https://github.com/explosion/spaCy/discussions/9512
    # A workarroud is to use blank=True and opt_load=True, but then there are
    # issues with senter and morpholizer.
    # Currently one should use blank=False, and then disable stages that are not
    # used: an observation suggests that for a better performance one should not
    # include in nlp the pipe stages (nlp_pipe) that are not used -- see:
    # https://github.com/explosion/spaCy/discussions/8402
    def create_nlp(self):
        self._nlp_logger.info('Creating NLP instance: start')
        assert self.nlp_spacy_core in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

        opt1 = False # does not work due to internal issue in spacy mentioned above
        if self.nlp_blank:
            if opt1:
                nlp = spacy.blank('en')
                nlp.add_pipe('senter')
                nlp.add_pipe('lemmatizer')
                nlp.add_pipe('tagger')
                nlp.add_pipe('attribute_ruler', config={'validate':True})
            else:
                source_nlp = spacy.load(self.nlp_spacy_core) #"en_core_web_sm"
                # source_nlp's nlp_pipe is ['tok2vec', 'tagger', 'parser',
                # 'attribute_ruler', 'lemmatizer', 'ner']; thus it does not
                # include 'senter' and "morphologizer" from the list below.
                #print('nlp_pipe', source_nlp.pipe_names)
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
            nlp = spacy.load(self.nlp_spacy_core) #'en_core_web_sm'
            # nlp's nlp_pipe is ['tok2vec', 'tagger', 'parser',
            # 'attribute_ruler', 'lemmatizer', 'ner']; thus it does not
            # include 'senter' and "morphologizer" from the list below.
            print(nlp.pipe_names); print('senter' in nlp.pipe_names)
            if not self.nlp_ner and "ner" in nlp.pipe_names:
                nlp.disable_pipe("ner")
            print(nlp.pipe_names); print('senter' in nlp.pipe_names)
            if not self.nlp_senter and "senter" in nlp.pipe_names:
                nlp.disable_pipe("senter")
            elif self.nlp_senter and not "senter" in nlp.pipe_names:
                try:
                    nlp.add_pipe("senter")
                except:
                    print('senter already exists in the pipeline')
                    try:
                        nlp.add_pipe("sentencizer")
                    except:
                        print('sentencizer already exists in the pipeline')
            if not self.nlp_lemmatizer and "lemmatizer" in nlp.pipe_names:
                nlp.disable_pipe("lemmatizer")
            if not self.nlp_tagger and "tagger" in nlp.pipe_names:
                nlp.disable_pipe("tagger")
            if not self.nlp_ruler and "attribute_ruler" in nlp.pipe_names:
                nlp.disable_pipe("attribute_ruler")
            if not self.nlp_parser and "parser" in nlp.pipe_names:
                nlp.disable_pipe("parser")
            if not self.nlp_tok2vec and "tok2vec" in nlp.pipe_names:
                nlp.disable_pipe("tok2vec")
            if not self.nlp_morphologizer and "morphologizer" in nlp.pipe_names:
                nlp.disable_pipe("morphologizer")
        #nlp.initialize()
        
        self._nlp_logger.info('NLP pipe: ' + ', '.join(nlp.pipe_names))
        self._nlp_logger.info('Creating NLP instance: end')
        return nlp

    # TODO extend this function to support more text processing heuristics
    # Currently we just apply lemmatization and drop the stop words and
    # punctuation marks as well as tokens of categories ['SPACE', 'X', 'PUNCT'].
    # Examples of stop words: to, for, over, a, from, had, not.
    # STOP_WORDS from spacy.lang.en.stop_words contains all English stop words.
    # For chat-bot/Q&R system, sentiment classification, and language
    # translation tasks, removing stop words could be a bad idea,
    # e.g. dropping stop word "not" in sentiment classification task.
    # TODO !!! Spacy cannot deal with texts longer than 100,000 characters
    def nlp_preprocess(self, text, nlp):
        if len(text) > 100000:
            self._nlp_logger.warning('Spacy library cannot support processing text \
                longer than 100,000 characters; ignoring characters in the text beyond that limit')
            text = text[:100000]
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
