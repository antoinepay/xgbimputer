import pandas as pd
from .exception import *
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


class XGBImputer:
    def __init__(self, data, random_seed=21):
        """
        Imputer from XGBoost algorithm

        Parameters
        ----------
        data : array-like
            Contains the whole dataset
        random_seed : scalar, default 21
            Keep the same results after multiple imputations
        """
        self.data = data
        self.random_seed = random_seed
        self.gbm = None

    def check_params(self, missing_values_variable, features):
        """
        Check parameters when trying to fit the imputation model

        Parameters
        ----------
        missing_values_variable : string
            Column name of the variable to impute
        features : array
            Features to be used to predict missing data within the variable
        """
        if not isinstance(self.data, pd.DataFrame):
            raise WrongArgumentException('data is not a dataframe')

        if missing_values_variable not in self.data.columns:
            raise WrongArgumentException('%s not found in data columns' % missing_values_variable)

        if not set(features).issubset(self.data.columns):
            raise WrongArgumentException(
                'variables : %s not found in data' % list(set(features).difference(self.data.columns))
            )

        if self.data[missing_values_variable].isna().sum() == 0:
            raise NotNecessaryImputationException('No NAs in %s variable' % missing_values_variable)

    def fit(self, missing_values_variable, features, params=None, with_cv=True, n_jobs=1, n_iter=5):
        """
        Fit the model to be used during the imputation

        Parameters
        ----------
        missing_values_variable : string
            Column name of the variable to impute
        features : array
            Features to be used to predict missing data within the variable
        params : dict
            Parameters to be passed into the fitting
        with_cv : boolean
            (Recommended) Use Cross-Validation to auto-select the best model to impute
        n_jobs : int
            Number of different threads used for Cross-Validation
        n_iter : int
            Number of different parameters set to use in Cross-Validation
        """
        self.check_params(missing_values_variable, features)

        params = params if params else {}

        train_set = self.data.dropna(subset=[missing_values_variable])

        X_train, y_train = train_set[features], train_set[missing_values_variable]

        gbm, scoring = self.get_best_model(missing_values_variable, params)

        if not with_cv:
            gbm.fit(X_train, y_train)

        self.gbm = self.fit_with_cv(gbm, X_train, y_train, scoring, n_jobs, n_iter) if with_cv else gbm

        return self

    def transform(self, missing_values_variable):
        """
        Impute the variable containing missing data

        Parameters
        ----------
        missing_values_variable : string
            Column name of the variable to impute
        """
        if not self.gbm:
            raise ModelNotFittedException('You must class fit method first')

        X_fill = self.data[self.data[missing_values_variable].isnull()]

        predictions = pd.Series(
            self.gbm.predict(X_fill.drop(columns=missing_values_variable)),
            index=X_fill.index
        )

        self.data[missing_values_variable] = self.data[missing_values_variable].fillna(predictions)

        return self

    def fit_transform(self, missing_values_variable, features, params=None, with_cv=True, n_jobs=1, n_iter=5):
        """
        Fit the model and impute the dataset

        Parameters
        ----------
        missing_values_variable : string
            Column name of the variable to impute
        features : array
            Features to be used to predict missing data within the variable
        params : dict
            Parameters to be passed into the fitting
        with_cv : boolean
            (Recommended) Use Cross-Validation to auto-select the best model to impute
        n_jobs : int
            Number of different threads used for Cross-Validation
        n_iter : int
            Number of different parameters set to use in Cross-Validation
        """
        return self.fit(missing_values_variable, features, params, with_cv, n_jobs, n_iter)\
            .transform(missing_values_variable)

    def get_best_model(self, missing_values_variable, params):
        """
        Select the best model according to the target variable
            - XGBoost regression for continuous variable
            - XGBoost binary logistic for binary categorical variable
            - XGBoost multi softmax for multiple classes categorical variable

        Parameters
        ----------
        missing_values_variable : string
            Column name of the variable to impute
        params : dict
            Parameters to be passed into the fitting
        """
        if self.data.dtypes[missing_values_variable] != 'object':
            return xgb.XGBRegressor(**params), 'neg_mean_squared_error'

        gbm = xgb.XGBClassifier(**params)

        classes = pd.unique(self.data.dropna(subset=[missing_values_variable])[missing_values_variable])

        n_classes = len(classes)

        if n_classes == 2:
            gbm.set_params(objective='binary:logistic')
            return gbm, 'roc_auc'
        else:
            gbm.set_params(objective='multi:softmax', n_classes=n_classes)
            return gbm, 'accuracy'

    def fit_with_cv(self, booster, X_train, y_train, scoring, n_jobs, n_iter):
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
        """
        params = {
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.5, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.8, 1]
        }

        grid_search = RandomizedSearchCV(
            estimator=booster,
            param_distributions=params,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=5,
            n_iter=n_iter,
            iid=False
        )

        grid_search.fit(X_train, y_train)

        print('Imputation with %s of %s' % (scoring, grid_search.best_score_))

        return grid_search.best_estimator_
