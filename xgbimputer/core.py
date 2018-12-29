import pandas as pd
from .exception import *
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


class XGBImputer:
    def __init__(self, data, missing_values_variable, features, random_seed=21):
        self.data = data
        self.missing_values_variable = missing_values_variable
        self.features = features
        self.random_seed = random_seed

        self.check_params()

    def check_params(self):
        if not isinstance(self.data, pd.DataFrame):
            raise WrongArgumentException('data is not a dataframe')

        if self.missing_values_variable not in self.data.columns:
            raise WrongArgumentException('%s not found in data columns' % self.missing_values_variable)

        if not set(self.features).issubset(self.data.columns):
            raise WrongArgumentException(
                'variables : %s not found in data' % list(set(self.features).difference(self.data.columns))
            )

        if self.data[self.missing_values_variable].isna().sum() == 0:
            raise NotNecessaryImputationException('No NAs in %s variable' % self.missing_values_variable)

    def imputation_accuracy(self, params=None, with_cv=True, n_jobs=1, n_iter=5):
        pass

    def impute(self, params=None, with_cv=True, n_jobs=1, n_iter=5):

        gbm = None
        scoring = ''

        params = params if params else {}

        train_set = self.data.dropna(subset=[self.missing_values_variable])

        X_train, y_train = train_set.drop(columns=[self.missing_values_variable]), train_set[self.missing_values_variable]

        X_fill = self.data[self.data[self.missing_values_variable].isnull()]

        if self.data.dtypes[self.missing_values_variable] == 'object':
            gbm = xgb.XGBClassifier(**params)

            classes = pd.unique(self.data.dropna(subset=[self.missing_values_variable])[self.missing_values_variable])

            n_classes = len(classes)

            if n_classes == 2:
                gbm.set_params(objective='binary:logistic')
                scoring = 'roc_auc'
            else:
                gbm.set_params(objective='multi:softmax', n_classes=n_classes)
                scoring = 'accuracy'
        else:
            gbm = xgb.XGBRegressor(**params)
            scoring = 'neg_mean_squared_error'

        if with_cv:
            params = {
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.05, 0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'subsample': [0.5, 0.8, 1.0],
                'colsample_bytree': [0.5, 0.8, 1]
            }

            grid_search = RandomizedSearchCV(
                estimator=gbm,
                param_distributions=params,
                scoring=scoring,
                n_jobs=n_jobs,
                cv=5,
                n_iter=n_iter,
                iid=False
            )

            grid_search.fit(X_train, y_train)

            gbm = grid_search.best_estimator_

            print('Imputation with %s of %s' % (scoring, grid_search.best_score_))
        else:
            gbm.fit(X_train, y_train)

        predictions = pd.Series(
            gbm.predict(X_fill.drop(columns=self.missing_values_variable)),
            index=X_fill.index
        )

        self.data[self.missing_values_variable] = self.data[self.missing_values_variable].fillna(predictions)

        return self.data
