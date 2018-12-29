import random
import pandas as pd
import numpy as np

from xgbimputer import XGBImputer


class TestImputation:
    def test_imputation_shape(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        data_imputed = XGBImputer(data, 'a', ['b', 'c', 'd', 'e']).impute()

        assert data_imputed.shape == data.shape

    def test_imputation_shape_with_cv(self):

        data = pd.DataFrame({
            'a': np.concatenate((np.arange(0, 98, 1), [np.nan, np.nan])),
            'b': [random.randint(1, 50) for i in range(100)],
            'c': [random.randint(1, 50) for i in range(100)],
            'd': [random.randint(1, 50) for i in range(100)],
            'e': [random.randint(1, 50) for i in range(100)]
        })

        data_imputed = XGBImputer(data, 'a', ['b', 'c', 'd', 'e']).impute(with_cv=True)

        assert data_imputed.shape == data.shape
