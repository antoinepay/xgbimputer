import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from .exception import *
from .grid_param import GridParam

class XGBImputer:
    def __init__(self, with_cv=True, verbose=0, random_seed=21):
        """
        Imputer from XGBoost algorithm

        Parameters
        ----------
        with_cv : boolean
            Enable Cross-Validation for imputing
        verbose : int
            Verbosity of the imputation process
        random_seed : scalar, default 21
            Keep the same results after multiple imputations
        """
        self.with_cv = with_cv
        self.verbose = verbose
        self.random_seed = random_seed
        self.gbm = None

    def check_params(self, data, missing_values_variable, features):
        """
        Check parameters when trying to fit the imputation model

        Parameters
        ----------
        data : array-like
            data to be checked
        missing_values_variable : string
            Column name of the variable to impute
        features : array-like
            Features to be used to predict missing data within the variable
        """
        if not isinstance(data, pd.DataFrame):
            raise WrongArgumentException('data is not a dataframe')

        if missing_values_variable not in data.columns:
            raise WrongArgumentException('%s not found in data columns' % missing_values_variable)

        if not set(features).issubset(data.columns):
            raise WrongArgumentException(
                'variables : %s not found in data' % list(set(features).difference(data.columns))
            )

        if data[missing_values_variable].isna().sum() == 0:
            raise NotNecessaryImputationException('No NAs in %s variable' % missing_values_variable)

    def fit(self, data, missing_values_variable, features, params=None, n_jobs=1, n_iter=5):
        """
        Fit the model to be used during the imputation

        Parameters
        ----------
        data : array-like
            data to fit the model
        missing_values_variable : string
            Column name of the variable to impute
        features : array-like
            Features to be used to predict missing data within the variable
        params : dict
            Parameters to be passed into the fitting
        n_jobs : int
            Number of different threads used for Cross-Validation
        n_iter : int
            Number of different parameters set to use in Cross-Validation
        """
        self.check_params(data, missing_values_variable, features)

        params = params if params else {}

        train_set = data.dropna(subset=[missing_values_variable])

        X_train, y_train = train_set[features], train_set[missing_values_variable]

        gbm, scoring = self.get_best_model(data, missing_values_variable, params)

        if not self.with_cv:
            gbm.fit(X_train, y_train)

        self.gbm = self.fit_with_cv(gbm, X_train, y_train, scoring, n_jobs, n_iter, params) if self.with_cv else gbm

        return self

    def transform(self, data, missing_values_variable):
        """
        Impute the variable containing missing data

        Parameters
        ----------
        data : array-like
            data to be imputed
        missing_values_variable : string
            Column name of the variable to impute
        """
        if not self.gbm:
            raise ModelNotFittedException('You must class fit method first')

        X_fill = data[data[missing_values_variable].isnull()]

        predictions = pd.Series(
            self.gbm.predict(X_fill.drop(columns=missing_values_variable)),
            index=X_fill.index
        )

        data[missing_values_variable] = data[missing_values_variable].fillna(predictions)

        return self

    def fit_transform(self, data, missing_values_variable, features, params=None, n_jobs=1, n_iter=5):
        """
        Fit the model and impute the dataset

        Parameters
        ----------
        data : array-like
            data to be copied and imputed
        missing_values_variable : string
            Column name of the variable to impute
        features : array-like
            Features to be used to predict missing data within the variable
        params : dict
            Parameters to be passed into the fitting
        n_jobs : int
            Number of different threads used for Cross-Validation
        n_iter : int
            Number of different parameters set to use in Cross-Validation
        """

        data_to_be_imputed = data.copy()

        self.fit(
            data,
            missing_values_variable,
            features,
            params,
            n_jobs,
            n_iter
        ).transform(
            data_to_be_imputed,
            missing_values_variable
        )

        return data_to_be_imputed

    def get_best_model(self, data, missing_values_variable, params):
        """
        Select the best model according to the target variable
            - XGBoost regression for continuous variable
            - XGBoost binary logistic for binary categorical variable
            - XGBoost multi softmax for multiple classes categorical variable

        Parameters
        ----------
        data : array-like
            data to be analysed
        missing_values_variable : string
            Column name of the variable to impute
        params : dict
            Parameters to be passed into the fitting
        """
        if data.dtypes[missing_values_variable] != 'object':
            gbm = xgb.XGBRegressor(**params).set_params(random_state=self.random_seed)
            return gbm, 'neg_mean_squared_error'

        gbm = xgb.XGBClassifier(**params).set_params(random_state=self.random_seed)

        classes = pd.unique(data.dropna(subset=[missing_values_variable])[missing_values_variable])

        n_classes = len(classes)

        if n_classes == 2:
            gbm.set_params(objective='binary:logistic')
            return gbm, 'roc_auc'
        else:
            gbm.set_params(objective='multi:softmax', n_classes=n_classes)
            return gbm, 'accuracy'

    def fit_with_cv(self, booster, X_train, y_train, scoring, n_jobs, n_iter, params):
        """
        Select the best model according to the target variable
            - XGBoost regression for continuous variable
            - XGBoost binary logistic for binary categorical variable
            - XGBoost multi softmax for multiple classes categorical variable

        Parameters
        ----------
        booster : Any
            Booster estimator to be used in Cross-Validation
        X_train : array
            Features to be used to predict missing data within the variable
        y_train : array
            Target variable used for training
        scoring : string
            Metric to be used within Cross-Validation
        n_jobs : int
            Number of different threads used for Cross-Validation
        n_iter : int
            Number of different parameters set to use in Cross-Validation
        params : dict
            Parameters to use initially for GridSearch
        """

        classifier = xgb.XGBClassifier()

        params = {
            'max_depth': GridParam(params['max_depth'] if 'max_depth' in params else
                                   classifier.max_depth, 2, 10, 2, 5).get_range(),
            'learning_rate': GridParam(params['learning_rate'] if 'learning_rate' in params else
                                       classifier.learning_rate, 0.05, 0.4, 0.005, 5).get_range(),
            'n_estimators': GridParam(params['n_estimators'] if 'n_estimators' in params else
                                      classifier.n_estimators, 50, 400, 20, 3).get_range(),
            'subsample': GridParam(params['subsample'] if 'subsample' in params else
                                   classifier.subsample, 0.1, 1, 0.5, 5).get_range(),
            'colsample_bytree': GridParam(params['colsample_bytree'] if 'colsample_bytree' in params else
                                          classifier.colsample_bytree, 0.1, 1, 0.5, 5).get_range()
        }

        grid_search = RandomizedSearchCV(
            estimator=booster,
            param_distributions=params,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=5,
            n_iter=n_iter,
            iid=False,
            verbose=self.verbose,
            random_state=self.random_seed
        )

        grid_search.fit(X_train, y_train)

        if not self.verbose == 0:
            print('Imputation with %s of %s' % (scoring, grid_search.best_score_))

        return grid_search.best_estimator_
