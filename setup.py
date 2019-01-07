from setuptools import setup, find_packages

setup(
    name='xgbimputer',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/antoinepay/xgbimputer',
    author='antoinepayan',
    author_email='antoine.payan@hotmail.fr',
    description='Imputation with XGBoost',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'pytest'
    ],
)
