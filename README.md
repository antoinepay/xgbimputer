# XGBoost Imputer

Impute missing values with XGBoost algorithm from choosen features of your dataset. 

Note : This process shall be used only with features where missing values are not MAR (Missing at random)

### Usage

The dataset must be a pandas Dataframe

##### Classic

```python
import pandas as pd
import numpy as np
from xgbimputer import XGBImputer

data = pd.DataFrame({
    'F1': [1, 2, 3, 4],
    'F2': [1, 2, 3, np.nan],
    'F3': [3, 2, 3, 4]
})

data_imputed = XGBImputer(with_cv=True).fit_transform(
    data=data,
    missing_values_variable='F2',
    features=['F1', 'F2'],
    params={'learning_rate':0.3},
    n_jobs=4, # Parallelizing Cross-validation
    n_iter=10 # Random choices among parameters grid
)
```

##### Inplace

```python
import pandas as pd
import numpy as np
from xgbimputer import XGBImputer

data = pd.DataFrame({
    'F1': [1, 2, 3, 4],
    'F2': [1, 2, 3, np.nan],
    'F3': [3, 2, 3, 4]
})

imputer = XGBImputer(with_cv=True)
imputer.fit(
    data=data,
    missing_values_variable='F2',
    features=['F1', 'F2'],
    params={'learning_rate':0.3},
    n_jobs=4, # Parallelizing Cross-validation
    n_iter=10 # Random choices among parameters grid
)
imputer.transform(data, 'F1')
```

### Cross-validation

You can pass parameters by reference to the fit method in order to customize XGBoost training. 
If you choose to use cross-validation for the training, an automatic range will be built around the value you specified or the default one.

### sklearn API

This library is compatible with scikit-learn API. 






