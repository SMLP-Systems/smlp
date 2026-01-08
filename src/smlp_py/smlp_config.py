# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

#import textwrap
import os, argparse, json
from smlp_py.smlp_utils import str_to_bool

class SmlpConfig:
    def __init__(self):
        self.report_file_prefix = None
        self.model_file_prefix = None
        self.model_rerun_config = None
        self.wordvec_file_prefix = None
        self.config = None
        
        self._DEF_LABELED_DATA = None
        self._DEF_TEXT_DATA = None
        self._DEF_ANALYTICS_MODE = None #'train'
        self._DEF_SAVE_CONFIGURATION = False
        self._DEF_LOG_FILE_PREFIX = None
        self._DEF_OUTPUT_DIRECTORY = None
        self._DEF_INTERACTIVE_PLOTS = True
        self._DEF_SEED = None
        self._DEF_LOAD_CONFIGURATION = None
        
        self.config_params_dict = {
            'labeled_data': {'abbr':'data', 'default':self._DEF_LABELED_DATA, 'type':str, 
                'help':'Path, possibly excluding the .csv, or including gz or bz2 suffix, to input ' +
                    ' training data file containing labels [default {}]'.format(str(self._DEF_LABELED_DATA))},
            'text_data': {'abbr':'text', 'default':self._DEF_TEXT_DATA, 'type':str, 
                'help':'Path to input training text data file for finetune and RAG modes. ' +
                    "In finetune mode, the expected structure varies by task: the text-generation task requires " +
                    "and the BERT-style QA requires 'question', 'context', 'answer' fields. " +
                    "In RAG mode: PDF for LangChain based RAG, and PDF/JSON/CSV for HuggingFace based RAG. " +
                    '[default {}]'.format(str(self._DEF_TEXT_DATA))},
            'analytics_mode': {'abbr':'mode', 'default':self._DEF_ANALYTICS_MODE, 'type':str, 
                'help':'What kind of analysis should be performed; the supported modes are: '+
                    '"train", "predict", "subgroups", "doe", "discretize", "optimize", "verify", "query", "optsyn" ' +
                    '[default: {}]'.format(str(self._DEF_ANALYTICS_MODE))},
            'interactive_plots': {'abbr':'plots', 'default':self._DEF_INTERACTIVE_PLOTS, 'type':str_to_bool, 
                'help':'Should plots be displayed interactively (or only saved)?'+
                    '[default: {}]'.format(str(self._DEF_INTERACTIVE_PLOTS))},
            'seed': {'abbr':'seed', 'default':self._DEF_SEED, 'type':int, 
                'help':'Initial random seed [default {}]'.format(str(self._DEF_SEED))},
            'log_files_prefix': {'abbr':'pref', 'default':self._DEF_LOG_FILE_PREFIX, 'type':str, 
                'help':'String to be used as prefix for the output files ' + 
                    '[default: {}]'.format(str(self._DEF_LOG_FILE_PREFIX))},
            'output_directory': {'abbr':'out_dir', 'default':self._DEF_OUTPUT_DIRECTORY, 'type':str, 
                'help':'Output directory where all reports and output files will be written '+
                    '[default: the same directory from which data is loaded]'},
            'save_configuration': {'abbr':'save_config', 'default':self._DEF_SAVE_CONFIGURATION, 'type':str_to_bool, 
                'help':'Should tool run parameters be saved into a a configuration file? ' +
                    '[default: {}]'.format(str(self._DEF_SAVE_CONFIGURATION))},
            'load_configuration': {'abbr':'config', 'default':self._DEF_LOAD_CONFIGURATION, 'type':str, 
                'help':'Json config file name, to load tool parameter values from, or None. ' +
                    'Paramters specified through command line will override the correponding '
                    'config file values if they are specified there as well ' +
                    '[default: {}]'.format(str(self._DEF_LOAD_CONFIGURATION))}
        }
    
    # Compute prefix report_name_prefix to be used in all report / log file names of an SMLP run; 
    # as well as prefix model_name_prefix to be used in the names of all output files that are required 
    # to save a trained model info and re-run the saved model on new data (without re-training).
    # The argument model_name is used for loading a pre-trained model and saving a newly trained model.
    # In the former use, full_path to the location of model files as well as the model name need to be
    # passed to argument model_name, and in the latter case only the model name needs to be passed, and
    # model files are saved in the output directory, along with all other logs / report files.
    def args_get_report_name_prefix(self, data_file_prefix:str, run_prefix:str, output_directory:str=None, 
            new_data_file_prefix:str=None, model_name:str=None, doe_spec_file_prefix=None, text_file_prefix=None, 
            wordvec_model=None):
        if not data_file_prefix is None:
            data_dir, data_name_prefix = os.path.split(data_file_prefix)
        else:
            data_dir, data_name_prefix = None, None
        
        # Define _model_dir and _model_name_prefix. 
        if not model_name is None:
            model_dir, model_name_prefix = os.path.split(model_name)
        else:
            model_dir, model_name_prefix = None, None

        if not doe_spec_file_prefix is None:
            doe_spec_dir, doe_spec_name_prefix = os.path.split(doe_spec_file_prefix)
        else:
            doe_spec_dir, doe_spec_name_prefix = None, None
        
        if not text_file_prefix is None:
            text_dir, text_name_prefix = os.path.split(text_file_prefix)
        else:
            text_dir, text_name_prefix = None, None
        
        out_dir = output_directory
        if out_dir is None:
            if not data_dir is None:
                out_dir = data_dir
            elif not text_dir is None:
                out_dir = text_dir
            elif not model_dir is None:
                out_dir = model_dir
            elif not doe_spec_dir is None:
                out_dir = doe_spec_dir
            else:
                raise Exception('A training data file, a model or doe spec file should be provided')   
        
        if data_name_prefix is not None:
            input_data_name_prefix = data_name_prefix.removesuffix('.bz2')
            input_data_name_prefix = input_data_name_prefix.removesuffix('.gz')
            input_data_name_prefix = input_data_name_prefix.removesuffix('.csv')
        elif text_name_prefix is not None:
            input_data_name_prefix = text_name_prefix.removesuffix('.json').\
                removesuffix('.jsonl').removesuffix('.txt').removesuffix('.pdf').removesuffix('.csv')
        elif doe_spec_name_prefix is not None:
            input_data_name_prefix = doe_spec_name_prefix.removesuffix('.csv')
        else:
            assert model_name is not None
            input_data_name_prefix = None

        # record model_name 
        #model_name = model_name

        # define model_name_prefix to be used as suffix in names of all files used to save the model related info
        #run_prefix = run_prefix
        if model_name is None:
            assert not (data_name_prefix is None and doe_spec_dir is None and text_name_prefix is None)
            model_name_prefix = os.path.join(out_dir, run_prefix + '_' + input_data_name_prefix)
        else:
            model_name_prefix = os.path.join(out_dir, model_name)

        # define _report_name_prefix to be used as a prefix in SMLP report filenames
        if input_data_name_prefix is None:
            assert not model_name is None
            _, model_name = os.path.split(model_name)
            report_name_prefix = os.path.join(out_dir, run_prefix + '_' + model_name)
        else:
            report_name_prefix = os.path.join(out_dir, run_prefix + '_' + input_data_name_prefix)

        # if new_data is not None, its name is added to self._filename_prefix
        if not new_data_file_prefix is None:
            new_data_file_prefix = new_data_file_prefix.removesuffix('.bz2')
            new_data_file_prefix = new_data_file_prefix.removesuffix('.gz')
            new_data_file_prefix = new_data_file_prefix.removesuffix('.csv')
            _, new_data_fname = os.path.split(new_data_file_prefix)
            report_name_prefix = report_name_prefix + '_' + new_data_fname
        
        # name of word vector embedding model -- user-trained or pre-trained
        if wordvec_model is None:
            wordvec_name_prefix = None
        else:
            _, wordvec_name = os.path.split(wordvec_model)
            wordvec_name_prefix = os.path.join(out_dir, wordvec_name)
        
        return report_name_prefix, model_name_prefix, wordvec_name_prefix

        
    # args parser to which some of the arguments are added explicitly in a regular way
    # and in addition it adds additional arguments from args_dict defined elsewhere;
    # As of now args_dict includes model training hyperparameters from ML packages
    # sklearm caret, keras -- model_params_dict = keras_dict | sklearn_dict | caret_dict, 
    # as well as data and logger related parameters: data_params_dict and logger_params_dict
    def args_dict_parse(self, argv, args_dict):
        if not argv:
            argv = ["run_smlp.py"]

        parser = argparse.ArgumentParser(prog=argv[0])

        for p, v in args_dict.items():
            if 'default' in v:
                parser.add_argument('-'+v['abbr'], '--'+p, default=v['default'], 
                                    type=v['type'], help=v['help'])
            else:
                parser.add_argument('-'+v['abbr'], '--'+p, type=v['type'], help=v['help'])

        # Initial parse
        args = parser.parse_args(argv[1:])

        # Handle config loading
        if args.load_configuration is not None:
            with open(args.load_configuration, 'r') as f:
                parser.set_defaults(**json.load(f))
            args = parser.parse_args(argv[1:])  # RE-parse with same argv (not sys.argv)

        # Compute and save report_file_prefix and model_file_prefix as part of self
        self.report_file_prefix, self.model_file_prefix, self.wordvec_file_prefix = \
            self.args_get_report_name_prefix(
                args.labeled_data, args.log_files_prefix, args.output_directory, args.new_data,
                args.model_name, args.doe_spec_file, args.text_data, args.wordvec_model
            )

        # Save config if requested
        if args.save_configuration:
            args_config_file = self.report_file_prefix + '_args_config.json'
            tmp_args = vars(args).copy()
            del tmp_args['save_configuration'] # Do not dump value of this option
            del tmp_args['load_configuration'] # Values already loaded
            self.config = tmp_args
            with open(args_config_file, 'w') as f:
                f.write(json.dumps(tmp_args, indent=4, sort_keys=True))

        # Save model rerun config if requested
        if args.save_model_rerun_configuration:
            if not vars(args)['save_model']:
                return args
            model_args = vars(args).copy()
            # assign false to save_model since we are using an already saved model
            model_args['save_model'] = 'false'
            # assign true to use_model since we want to use a saved model
            model_args['use_model'] = 'true'
            # new data set must be provided, we are not using new data from config file
            model_args['new_data'] = None
            # training (labeled) data set from which model was built is not required
            model_args['labeled_data'] = None
            # the log file prefix used to create model is not needed
            model_args['log_files_prefix'] = None
            # prefix model_name with the output directory
            model_args['model_name'] = None
            self.model_rerun_config = model_args

        return args
