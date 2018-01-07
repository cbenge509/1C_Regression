# 1C Regression ML Challenge - December, 2017
Future Sales Prediction Challenge - Kaggle

**This submssion from Dec. 2017 was good for a 0.94x private leaderboard score and 13th place by the close of CY 2017.  Final submission was a bagged version of XGB, LightGBM, and ExtraTreesRegressor (each individually built then averaged by a separate utility script).**

**Start with the submission.py file to see how the final submissions are created; this script calls contest.py (where all of the feature engineering and data cleansing takes place) and build.py (where ML models are constructed and parameterized).  If you want to use the pickle files, which is much faster than loading the CSV files, refer to the pickle.py file.**

[email](mailto:cbenge509@gmail.com)-  cbenge509, 2017

## Links

- **Competition:**
    - Challenge description:  [1C - Predict Future Sales ](https://www.kaggle.com/c/competitive-data-science-final-project)
- **Documentation**
    - All documentation covered in the Jupyter Notebook files (3 x EDA and 1 x validation analysis).
- **Models used**
    - SciKit-Learn:
        - [ensemble methods](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
            - [ExtraTreesRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
        - [generalized linear models](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
            - [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
    - [XGBoost](https://github.com/dmlc/xgboost)
        - [XGBoostRegressor](http://xgboost.readthedocs.io/en/latest/python/python_api.html)
    - [Microsoft LightGBM](https://github.com/Microsoft/LightGBM)
        - [LGBMRegressor](http://lightgbm.readthedocs.io/en/latest/Python-API.html)
    - [MLXTend](https://github.com/rasbt/mlxtend)
        - [StackingRegressorCV](https://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/)
            - this was used in the stacking branch of code (see submission.py); did not result in improved scores.
    
## License

- This project is released under a permissive MIT open source license ([LICENSE-MIT.txt](https://github.com/cbenge509/DataScienceCapstone_Oct2017/blob/master/LICENSE-MIT.txt)).  There is no warranty; not even for merchantability or fitness for a particular solution.

