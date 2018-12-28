from setuptools import setup, find_packages

setup(
    name='xgbimputer',
    version='',
    packages=find_packages(),
    url='',
    license='',
    author='antoinepayan',
    author_email='antoine.payan@hotmail.fr',
    description='Imputation with XGBoost',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost'
    ],
)
