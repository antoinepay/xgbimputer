import random
import pandas as pd
import numpy as np
import pytest

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

        XGBImputer(data).check_params('a', ['b', 'c', 'd', 'e'])

    def test_check_params_wrong_arguments(self):
        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        pytest.raises(WrongArgumentException, "XGBImputer(data).check_params('f', ['b', 'c', 'd', 'e'])")

    def test_check_params_not_necessary(self):
        data = pd.DataFrame({
            'a': np.arange(0, 100, 1),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        pytest.raises(NotNecessaryImputationException, "XGBImputer(data).check_params('a', ['b', 'c', 'd', 'e'])")

    def test_imputation_shape(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        orig_shape = data.shape

        XGBImputer(data).impute('a', ['b', 'c', 'd', 'e'])

        assert data.shape == orig_shape

    def test_imputation_shape_with_cv(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        orig_shape = data.shape

        XGBImputer(data).impute('a', ['b', 'c', 'd', 'e'], with_cv=True)

        assert data.shape == orig_shape

    def test_no_na_after_imputation(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        assert data['a'].isna().sum() > 0

        XGBImputer(data).impute('a', ['b', 'c', 'd', 'e'], with_cv=True)

        assert data['a'].isna().sum() == 0
