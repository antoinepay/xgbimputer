import random
import pandas as pd
import numpy as np
import pytest
import string

from xgbimputer import *


class TestImputation:
    def test_check_params_success(self):
        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        XGBImputer().check_params(data, 'a', ['b', 'c', 'd', 'e'])

    def test_check_params_wrong_arguments(self):
        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        pytest.raises(WrongArgumentException, "XGBImputer().check_params(data, 'f', ['b', 'c', 'd', 'e'])")

    def test_check_params_not_necessary(self):
        data = pd.DataFrame({
            'a': np.arange(0, 100, 1),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        pytest.raises(NotNecessaryImputationException, "XGBImputer().check_params(data, 'a', ['b', 'c', 'd', 'e'])")

    def test_reg_imputation_shape(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        orig_shape = data.shape

        XGBImputer().fit(data, 'a', ['b', 'c', 'd', 'e']).transform(data, 'a')

        assert data.shape == orig_shape

    def test_reg_imputation_shape_with_cv(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        orig_shape = data.shape

        XGBImputer(with_cv=True).fit(data, 'a', ['b', 'c', 'd', 'e']).transform(data, 'a')

        assert data.shape == orig_shape

    def test_reg_no_na_after_imputation(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        assert data['a'].isna().sum() > 0

        XGBImputer(with_cv=True).fit(data, 'a', ['b', 'c', 'd', 'e']).transform(data, 'a')

        assert data['a'].isna().sum() == 0

    def test_reg_no_na_after_fit_transform(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        assert data['a'].isna().sum() > 0

        data_imputed = XGBImputer(with_cv=True).fit_transform(data, 'a', ['b', 'c', 'd', 'e'], n_iter=5)

        assert data_imputed['a'].isna().sum() == 0

    def test_bin_log_no_na_after_fit_transform(self):

        data = pd.DataFrame({
            'a': [random.choice(string.ascii_lowercase[:2]) for i in range(98)] + [np.nan, np.nan],
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        assert data['a'].isna().sum() > 0

        data_imputed = XGBImputer(with_cv=True).fit_transform(data, 'a', ['b', 'c', 'd', 'e'], n_iter=5)

        assert data_imputed['a'].isna().sum() == 0

    def test_multi_no_na_after_fit_transform(self):

        data = pd.DataFrame({
            'a': [random.choice(string.ascii_lowercase[:10]) for i in range(98)] + [np.nan, np.nan],
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        assert data['a'].isna().sum() > 0

        data_imputed = XGBImputer(with_cv=True).fit_transform(data, 'a', ['b', 'c', 'd', 'e'], n_iter=5)

        assert data_imputed['a'].isna().sum() == 0


