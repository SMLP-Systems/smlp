# Fitting sklearn regression tree models
from sklearn.tree import DecisionTreeRegressor
#from sklearn.tree import _tree
from sklearn import tree, ensemble

# Fitting sklearn polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# general
import numpy as np
import pandas as pd
#import pickle

# SMLP
from smlp.smlp_plot import *
from smlp.formula_sklearn import SklearnFormula
from utils_common import str_to_bool, lists_union_order_preserving_without_duplicates


# Methods for training and predction, results reproting with SKLEARN package   
# Currently 'rf', 'dt', 'et', 'poly' are supported
# When addig new models self._KERAS_MODELS = ['nn'] needs to be updated
class ModelSklearn:
    def __init__(self):
        #data_logger = logging.getLogger(__name__)
        self._sklearn_logger = None
        self._SKLEARN_MODELS = ['dt', 'et', 'rf', 'poly']
        self.SMLP_SKLEARN_MODELS = [self._algo_name_local2global(m) for m in self._SKLEARN_MODELS]
        self._instFormula = SklearnFormula()
        
        # trees (rf, dt, et) common
        self._DEF_MIN_SAMPLES_SPLIT = 2
        self._DEF_MIN_SAMPLES_LEAF = 1
        self._DEF_MAX_DEPTH = None
        self._DEF_RAND_STATE = None
        self._DEF_CRITERION = 'squared_error'
        self._DEF_MAX_LEAF_NODES = None
        self._DEF_MIN_WEIGHT_FRACTION_LEAF = 0.0
        self._DEF_MIN_IMPURITY_DECREASE = 0.0
        self._DEF_CCP_ALPHA = 0.0
        # dt
        self._DEF_MAX_FEATURS_DT = None
        # rf
        self._DEF_N_ESTIMATORS = 100
        self._DEF_SPLITTER_RF = 'best'
        self._DEF_MAX_FEATURS_RF = 1.0
        self._DEF_BOOTSTRAP = True
        self._DEF_VERBOSE_RF = 0
        self._DEF_WARM_START = False
        self._DEF_MAX_SAMPLES = None
        # et
        self._DEF_SPLITTER_ET = 'random'
        self._DEF_MAX_FEATURS_ET = 1.0
        # linear / polynomial rgeression
        self._DEF_POLY_DEGREE = 2
        self._DEF_FIT_INTERCEPT = True
        self._DEF_COPY_X = True
        self._DEF_N_JOBS = None
        self._DEF_POSITIVE = False

        # TODO !!!: Is there a way to get this dictionary through SKLEARN API?
        # So this will work if hyperparameters are chenged in the future, and
        # also there will not be to do copy/paste hyperparamer names, defaults
        # and descriptions from the documentation. !!!!!!!!!!!!!!!!!!!!!!!
        # hyperparameters that are exactly the same to rf, dt and et
        self._trees_hyperparam_dict = {
            'criterion': {'abbr':'criterion', 'default': self._DEF_CRITERION, 'type':str,
                'help': 'The function to measure the quality of a split. Supported criteria are ' +
                        '“squared_error” for the mean squared error, which is equal to variance ' +
                        'reduction as feature selection criterion and minimizes the L2 loss using ' +
                        'the mean of each terminal node, “friedman_mse”, which uses mean squared error ' +
                        'with Friedman’s improvement score for potential splits, “absolute_error” for the ' +
                        'mean absolute error, which minimizes the L1 loss using the median of each terminal ' +
                        'node, and “poisson” which uses reduction in Poisson deviance to find splits. ' + 
                        'Training using “absolute_error” is slower than when using “squared_error”. ' +
                        '[default: ' + str(self._DEF_CRITERION) + ']'},
            'max_depth': {'abbr':'max_depth', 'default': self._DEF_MAX_DEPTH, 'type':int,
                'help': 'The maximum depth of the tree. If None, then nodes are expanded until all ' + 
                        'leaves are pure or until all leaves contain less than min_samples_split samples. ' +
                        '[default: ' + str(self._DEF_MAX_DEPTH) + ']'},
            'min_samples_split': {'abbr':'min_samples_split', 'default': self._DEF_MIN_SAMPLES_SPLIT, 'type':str, # can be int or float
                'help': 'The minimum number of samples required to split an internal node.' + 
                        'If int, then consider min_samples_split as the minimum number. If float, ' +
                        'min_samples_split is a fraction and ceil(min_samples_split * n_samples) ' +
                        'is the minimum number of samples for each split. ' +
                        '[default: ' + str(self._DEF_MIN_SAMPLES_SPLIT) + ']'},    
            'min_samples_leaf': {'abbr':'min_samples_leaf', 'default': self._DEF_MIN_SAMPLES_LEAF, 'type':str, # can be int or float
                'help': 'The minimum number of samples required to be at a leaf node. ' +
                        'If int, then consider min_samples_leaf as the minimum number. If float, ' +
                        'min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) ' +
                        'is the minimum number of samples for each node. '+
                        ' [default: ' + str(self._DEF_MIN_SAMPLES_LEAF) + ']'},
            'min_weight_fraction_leaf': {'abbr':'min_weight_fraction_leaf', 'default': self._DEF_MIN_WEIGHT_FRACTION_LEAF, 'type':float,
                'help': 'The minimum weighted fraction of the sum total of weights ' +
                        '(of all the input samples) required to be at a leaf node. ' +
                        'Samples have equal weight when sample_weight is not provided. ' +
                        ' [default: ' + str(self._DEF_MIN_WEIGHT_FRACTION_LEAF) + ']'},
            'max_leaf_nodes': {'abbr':'max_leaf_nodes', 'default': self._DEF_MAX_LEAF_NODES, 'type':int,
                'help': 'Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are ' + 
                        'defined as relative reduction in impurity. If None then unlimited ' + 
                        'number of leaf nodes [default: ' + str(self._DEF_MAX_LEAF_NODES) + ']'},
            'min_impurity_decrease': {'abbr':'min_impurity_decrease', 'default': self._DEF_MIN_IMPURITY_DECREASE, 'type':float,
                'help': 'A node will be split if this split induces a decrease of the impurity ' +
                        'greater than or equal to this value ' + 
                        'N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity), ' +
                        'where N is the total number of samples, N_t is the number of samples at the current node, ' +
                        'N_t_L is the number of samples in the left child, and N_t_R is the number of samples '
                        'in the right child. N, N_t, N_t_R and N_t_L all refer to the weighted sum, ' +
                        'if sample_weight is passed. [default: ' + str(self._DEF_MIN_IMPURITY_DECREASE) + ']'},
            'ccp_alpha': {'abbr':'ccp_alpha', 'default': self._DEF_CCP_ALPHA, 'type':float,
                'help': 'Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with ' + 
                        'the largest cost complexity that is smaller than ccp_alpha will be chosen. ' +
                        'By default, no pruning is performed. [default: ' + str(self._DEF_CCP_ALPHA) + ']'}
            }

        # parameters that are unique to dt or usage is slightly different from et, rf
        self._dt_hyperparam_dict = {
            'splitter': {'abbr':'splitter', 'default': self._DEF_SPLITTER_RF, 'type':str,
                'help': 'The strategy used to choose the split at each node. Supported strategies are ' +
                        '“best” to choose the best split and “random” to choose the best random split ' + 
                        '[default: ' + str(self._DEF_SPLITTER_RF) + ']'},
            'max_features': {'abbr':'max_features', 'default': self._DEF_MAX_FEATURS_DT, 'type':str, # can be int, float or constant string
                'help': 'The number of features to consider when looking for the best split: ' +
                        'If int, then consider max_features features at each split. ' +
                        'If float, max_features is a fraction and max(1, int(max_features * n_features_in_)) ' +
                        'features are considered at each split. ' + 
                        'If “sqrt”, then max_features=sqrt(n_features). ' +
                        'If “log2”, then max_features=log2(n_features). ' +
                        'If None, then max_features=n_features. [default: ' + str(self._DEF_MAX_FEATURS_DT) + ']'},
            'random_state': {'abbr':'rand_state', 'default': self._DEF_RAND_STATE, 'type':int,
                'help': 'Controls the randomness of the estimator. The features are always ' + 
                        'randomly permuted at each split, even if splitter is set to "best". ' +
                        'When max_features < n_features, the algorithm will select max_features ' +
                        'at random at each split before finding the best split among them. ' +
                        'But the best found split may vary across different runs, even if ' +
                        'max_features=n_features. That is the case, if the improvement of the ' +
                        'criterion is identical for several splits and one split has to be selected ' +
                        'at random. To obtain a deterministic behaviour during fitting, random_state ' +
                        'has to be fixed to an integer. [default: ' + str(self._DEF_RAND_STATE) + ']'}
            }

        # parameters that are unique to rf or usage is slightly different from et, dt
        self._rf_hyperparam_dict = {
            'n_estimators': {'abbr':'n_estimators', 'default': self._DEF_N_ESTIMATORS, 'type':int,
                'help': 'The number of trees in the forest. [default: ' + str(self._DEF_N_ESTIMATORS) + ']'},
            'max_features': {'abbr':'max_features', 'default': self._DEF_MAX_FEATURS_RF, 'type':str, # can be int, float or constant string
                'help': 'The number of features to consider when looking for the best split: ' +
                        'If int, then consider max_features features at each split. ' +
                        'If float, max_features is a fraction and max(1, int(max_features * n_features_in_)) ' +
                        'features are considered at each split, where n_features_in_ is the number of features seen during fit. ' + 
                        'If “sqrt”, then max_features=sqrt(n_features). ' +
                        'If “log2”, then max_features=log2(n_features). ' +
                        'If None or 1.0, then max_features=n_features. [default: ' + str(self._DEF_MAX_FEATURS_RF) + ']'},
            'bootstrap': {'abbr':'bootstrap', 'default': self._DEF_BOOTSTRAP, 'type':str_to_bool,
                'help': 'Whether bootstrap samples are used when building trees. If False, the whole ' +
                        'dataset is used to build each tree [default: ' + str(self._DEF_BOOTSTRAP) + ']'},
            'verbose': {'abbr':'verbose', 'default': self._DEF_VERBOSE_RF, 'type':str_to_bool,
                'help': 'Controls the verbosity when fitting and predicting. [default: ' + str(self._DEF_VERBOSE_RF) + ']'},
            'warm_start': {'abbr':'warm_start', 'default': self._DEF_WARM_START, 'type':str_to_bool,
                'help': 'When set to True, reuse the solution of the previous call to fit and add more ' +
                        'estimators to the ensemble, otherwise, just fit a whole new forest ' +
                        '[default: ' + str(self._DEF_WARM_START) + ']'},
            'max_samples': {'abbr':'max_samples', 'default': self._DEF_MAX_SAMPLES, 'type':str,
                'help': 'If bootstrap is True, the number of samples to draw from X to train each base estimator. ' +
                        'If None (default), then draw X.shape[0] samples. ' +
                        'If int, then draw max_samples samples.' +
                        'If float, then draw max(round(n_samples * max_samples), 1) samples. ' +
                        'Thus, max_samples should be in the interval (0.0, 1.0]. ' +
                        '[default: ' + str(self._DEF_MAX_SAMPLES) + ']'},
            'random_state': {'abbr':'rand_state', 'default': self._DEF_RAND_STATE, 'type':int,
                'help': 'Controls both the randomness of the bootstrapping of the samples used when building ' +
                        'trees (if bootstrap=True) and the sampling of the features to consider when ' +
                        'looking for the best split at each node (if max_features < n_features). ' +
                        '[default: ' + str(self._DEF_RAND_STATE) + ']'}
            }

        # parameters that are unique to et or usage is slightly different from dt, rf
        self._et_hyperparam_dict = {
            'splitter': {'abbr':'splitter', 'default': self._DEF_SPLITTER_ET, 'type':str,
                'help': 'The strategy used to choose the split at each node. Supported strategies are ' +
                        '“best” to choose the best split and “random” to choose the best random split ' + 
                        '[default: ' + str(self._DEF_SPLITTER_ET) + ']'},
            'max_features': {'abbr':'max_features', 'default': self._DEF_MAX_FEATURS_ET, 'type':str, # can be int, float or constant string
                'help': 'The number of features to consider when looking for the best split: ' +
                        'If int, then consider max_features features at each split. ' +
                        'If float, max_features is a fraction and max(1, int(max_features * n_features_in_)) ' +
                        'features are considered at each split, where n_features_in_ is the number of features seen during fit. ' + 
                        'If “sqrt”, then max_features=sqrt(n_features). ' +
                        'If “log2”, then max_features=log2(n_features). ' +
                        'If None, then max_features=n_features. [default: ' + str(self._DEF_MAX_FEATURS_ET) + ']'},
            'random_state': {'abbr':'rand_state', 'default': self._DEF_RAND_STATE, 'type':int,
                'help': 'Used to pick randomly the max_features used at each split. ' + 
                        'Note that the mere presence of random_state doesn’t mean that randomization ' +
                        'is always used, as it may be dependent on another parameter, e.g. shuffle, being set. ' +
                        '[default: ' + str(self._DEF_RAND_STATE) + ']'}
            }


        # hyper params dictionary for sklearn model training
        self._poly_hyperparam_dict = {
            'degree': {'abbr':'degree', 'default': self._DEF_POLY_DEGREE, 'type':int,
                'help': 'Degree of the polynomial to train [default: ' + str(self._DEF_POLY_DEGREE) + ']'},
            'fit_intercept': {'abbr':'fit_intercept', 'default': self._DEF_FIT_INTERCEPT, 'type':str_to_bool,
                'help': 'Whether to calculate the intercept for this model. If set to False, ' +
                        'no intercept will be used in calculations (i.e. data is expected to be centered). ' +
                        '[default: ' + str(self._DEF_FIT_INTERCEPT) + ']'},
            'copy_X': {'abbr':'copy_X', 'default': self._DEF_COPY_X, 'type':str_to_bool,
                'help': 'If True, X will be copied; else, it may be overwritten. [default: ' + str(self._DEF_COPY_X) + ']'},
            'n_jobs': {'abbr':'n_jobs', 'default': self._DEF_N_JOBS, 'type':int,
                'help': 'The number of jobs to use for the computation. This will only provide speedup ' +
                        'in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly ' + 
                        'X is sparse or if positive is set to True. ' +
                        'None means 1 unless in a joblib.parallel_backend context. ' +
                        '-1 means using all processors [default: ' + str(self._DEF_N_JOBS) + ']'},
            'positive': {'abbr':'positive', 'default': self._DEF_POSITIVE, 'type':str_to_bool,
                'help': 'When set to True, forces the coefficients to be positive. ' +
                        'This option is only supported for dense arrays. ' +
                        '[default: ' + str(self._DEF_POSITIVE) + ']'}
            }

        
        self.sklearn_hparam_dict = self.get_sklearn_hparam_default_dict()
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._sklearn_logger = logger
        self._instFormula.set_logger(logger)
    
    # local names for model are 'dt', 'rf', ..., while global names are 'dt_sklearn'
    # 'rf_sklearn', to distinguish dt, rf, ... implementation in different packages
    def _algo_name_local2global(self, algo):
        return algo+'_sklearn'
    
    # local name of hyper parameter (as in sklearn package) to global name;
    # the global name is obtained from local name, say 'max_depth', by prefixing it
    # with the global name of the algorithm, which results in 'dt_sklearn_max_depth'
    def _hparam_name_local_to_global(self, hparam, algo):
        #print('hparam global name', hparam, algo)
        return self._algo_name_local2global(algo) + '_' + hparam
        
    # given training algo name like dt and the hyper parameter dictionary param_dict  
    # for that algo in the python package used in this class), this function returns  
    # a modified dictionary obtained from param_dictby by adds algo name like dt_sklearn
    # (where sklearn is the name of the package used) to the parameter name and its
    # correponding abbriviated name in param_dict.
    def _param_dict_with_algo_name(self, param_dict, algo):
        #print('param_dict', param_dict)
        result_dict = {}
        for k, v in param_dict.items():
            v_updated = v.copy()
            v_updated['abbr'] = self._hparam_name_local_to_global(v['abbr'], algo) # algo + '_' + v['abbr']
            #print('updated abbrv', v_updated['abbr'])
            #print('updated key', self._hparam_name_local_to_global(k, algo))
            result_dict[self._hparam_name_local_to_global(k, algo)] = v_updated #algo + '_' + k
        #raise Exception('tmp')
        return result_dict
    
    # local hyper params dictionary
    def get_sklearn_hparam_default_dict(self):
        dt_sklearn_hyperparam_dict = self._param_dict_with_algo_name(self._dt_hyperparam_dict | self._trees_hyperparam_dict, 'dt')
        rf_sklearn_hyperparam_dict = self._param_dict_with_algo_name(self._rf_hyperparam_dict | self._trees_hyperparam_dict, 'rf')
        et_sklearn_hyperparam_dict = self._param_dict_with_algo_name(self._et_hyperparam_dict | self._trees_hyperparam_dict, 'et')
        poly_sklearn_hyperparam_dict = self._param_dict_with_algo_name(self._poly_hyperparam_dict, 'poly')
        sklearn_hparam_dict = poly_sklearn_hyperparam_dict | dt_sklearn_hyperparam_dict | \
            rf_sklearn_hyperparam_dict | et_sklearn_hyperparam_dict
        return sklearn_hparam_dict

    # train decision tree regression model with sklearn
    def dt_regr_train(self, feature_names, resp_names, algo,
            X_train, X_test, y_train, y_test, seed, weights):
        # Fit the regressor, set max_depth = 3
        regr = DecisionTreeRegressor(max_depth=15, random_state=seed)
        model = regr.fit(X_train, y_train, sample_weight=weights)
        assert(regr == model)

        # print text representation of the tree model
        text_representation = tree.export_text(model)
        print(text_representation)

        '''
        # visualaize tree TODO !!!!!!!!!!!! does not work 
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(regr)
        plt.show()
        plt.clf()
        '''  
        
        return model

    # train random forest regression model with sklearn
    def rf_regr_train(self, feature_names, resp_names, algo,
            X_train, X_test, y_train, y_test, seed, weights):
        # Fit the regressor, set max_depth = 3
        model = ensemble.RandomForestRegressor(max_depth=15, random_state=seed)
        model = model.fit(X_train, y_train, sample_weight=weights)
        #assert regr == model

        return model

    # train extra trees regression model with sklearn
    def et_regr_train(self, feature_names, resp_names, algo,
            X_train, X_test, y_train, y_test, seed, weights):
        # Fit the regressor, set max_depth = 3
        model = ensemble.ExtraTreesRegressor(max_depth=15, random_state=seed)
        model = model.fit(X_train, y_train, sample_weight=weights)
        #assert regr == model
        #assert regr.estimators_[0].tree_.value.shape[1] == 1 # no support for multi-output
        #self._instFormula.trees_to_rules(inst, regr.estimators_, feature_names, resp_names, None, True, True)

        return model

    # prints basic statistics of each column of df.
    # similar to pd.descrie() but this functions only prints a few columns when df has many columns
    def df_cols_summary(self, df):
        for col in df.columns.tolist():
            print(col, 
                  'min', df[col].min(), 
                  'max', df[col].max(),
                  'mean', df[col].mean(),
                  'std', df[col].std(), 
                  'z_min', (df[col].min() - df[col].mean())/df[col].std(), 
                  'z_max', (df[col].max() - df[col].mean())/df[col].std())  

    # train polynomial regression model with sklearn
    def poly_train(self, input_names, resp_names, degree,
            X_train, X_test, y_train, y_test, seed, weights):
        #print('poly_degree', degree); print('weigts', weights);     
        poly_reg = PolynomialFeatures(degree)
        #dummy = df_cols_summary(X_train) 
        X_train = poly_reg.fit_transform(X_train)
        X_test = poly_reg.transform(X_test)
        #print('X_train', X_train.shape, '\n', X_train)
        #print('transformed X_train data')
        #print(pd.DataFrame(X_train).describe())
        #dummy = df_cols_summary(pd.DataFrame(X_train))
        #raise Exception('tmp')
        pol_reg = LinearRegression()
        model = pol_reg.fit(X_train, y_train, sample_weight=weights)

        ''' # writing spec file
        model_domain = training_data_to_domain_spec(X_train, input_names)
        domain_file = open(inst._report_name_prefix + "_poly_domain.txt"), "w")
        domain_file.write(model_domain)
        domain_file.close()
        '''

        assert(model == pol_reg)
        return model, poly_reg #, X_train, X_test
        
    # model for sklearn poly model is in fact a pair (linear_model, poly_reg), where
    # poly_reg is transformer that creates polynomial terems (like x^2) from the original
    # features (like x), and linear_model is a linear regeression model trained on the
    # extended dataset obtained from original features X by applying transformer poly_reg.
    # Therefore, to apply model to dataset we apply model[0] to model[1].transform(X).
    def poly_sklearn_predict(self, model, X):
        return model[0].predict(model[1].transform(X))
        
    def _sklearn_train_multi_response(self, inst, feat_names, resp_names, algo,
            X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
            seed, sample_weights_vect):
        if algo in ['dt', 'et', 'rf']:
            if algo == 'dt':
                model = self.dt_regr_train(feat_names, resp_names, algo,
                    X_train, X_test, y_train, y_test, seed, sample_weights_vect)
                tree_estimators = [model]
            elif algo == 'rf':
                model = self.rf_regr_train(feat_names, resp_names, algo,
                    X_train, X_test, y_train, y_test, seed, sample_weights_vect)
                tree_estimators = model.estimators_
            elif algo == 'et':
                model = self.et_regr_train(feat_names, resp_names, algo,
                    X_train, X_test, y_train, y_test, seed, sample_weights_vect)
                tree_estimators = model.estimators_
            else:
                assert False
                
            # save tree model as rules
            rules_report_name = inst.get_model_name_prefix(resp_names, self._algo_name_local2global(algo)) + '_tree_rules.txt'
            print('rules_report_name', rules_report_name); print('feat_names', feat_names); print('resp_names', resp_names); 
            self._instFormula.trees_to_rules(tree_estimators, feat_names, resp_names, 
                None, True, rules_report_name)
            return model
        elif algo == 'poly':
            degree = hparam_dict[self._hparam_name_local_to_global('degree', 'poly')]
            #model, poly_reg, X_train, X_test 
            linear_model, poly_reg = self.poly_train(feat_names, resp_names, degree,
                X_train, X_test, y_train, y_test, seed, sample_weights_vect)
            for resp_id in range(len(resp_names)):
                #formula_report_name = inst._report_name_prefix + '_' + str(algo) + '_' + resp_names[resp_id] + "_poly_formula.txt"
                formula_report_name = inst.get_model_name_prefix(resp_names, self._algo_name_local2global(algo)) + '_formula.txt'
                model_formula = self._instFormula.poly_model_to_formula(feat_names, resp_names, linear_model.coef_, 
                    poly_reg.powers_, resp_id, True, formula_report_name)
            return linear_model, poly_reg #, X_train, X_test
        else:
            raise Exception('Unsupported model type ' + str(algo) + ' in function tree_main')
        
    def sklearn_main(self, inst, feat_names_dict, resp_names, algo,
            X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
            seed, sample_weights_vect, model_per_response):
        # train a separate models for each response, pack into a dictionary with response names
        # as keys and the correponding models as values
        if model_per_response:
            model = {}
            for rn in resp_names:
                rn_model = self._sklearn_train_multi_response(inst, feat_names_dict[rn], [rn], algo,
                    X_train, X_test, y_train[[rn]], y_test[[rn]], hparam_dict, interactive_plots, 
                    seed, sample_weights_vect)
                model[rn] = rn_model
            return model
        else:
            #union_feat_names = list(set(sum(list(feat_names_dict.values()), []))) - the ordering is affected 
            union_feat_names = lists_union_order_preserving_without_duplicates(list(feat_names_dict.values()))
            model = self._sklearn_train_multi_response(inst, union_feat_names, resp_names, algo,
                X_train, X_test, y_train, y_test, hparam_dict, interactive_plots, 
                seed, sample_weights_vect)
            return model

'''
    # generates domain file in current format for running the solvers.
    # TODO: this is for temporary usage, amd also does not belong to this file
    def training_data_to_domain_spec(X_poly, input_names):
        X_poly_df = pd.DataFrame(X_poly)
        print('X_poly_df\n', X_poly_df)
        X_test_scaled_df = X_poly_df[range(1,len(input_names)+1)]
        X_test_scaled_df.columns = input_names
        print('X_test_scaled_df\n', X_test_scaled_df)
        domain_spec = ''
        for col in X_test_scaled_df.columns.tolist():
            print(col, '\n', X_test_scaled_df[col].sort_values())
            domain_spec = domain_spec + col + ' -- ' + '[' + str(X_test_scaled_df[col].min()) + ',' + str(X_test_scaled_df[col].max()) + ']\n'
        print('domain_spec\n', domain_spec)
        return domain_spec
'''
