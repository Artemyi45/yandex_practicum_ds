import sklearn
import phik
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyRegressor
from sklearn.dummy import DummyClassifier


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (make_scorer, roc_auc_score)


train_job_sr = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Проекты/HR/train_job_satisfaction_rate.csv')
test_features = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Проекты/HR/test_features.csv')
test_target_job_sr = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Проекты/HR/test_target_job_satisfaction_rate.csv')

train_quit = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Проекты/HR/train_quit.csv')
test_target_quit = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Проекты/HR/test_target_quit.csv')

test_features = test_features.replace(' ', np.nan)
test_features = test_features.replace('sinior', 'senior')
train_quit = train_quit.replace('sinior', 'senior')

hr_test = test_features.merge(test_target_job_sr, on = 'id', how = 'inner')
hr_test_no_id = hr_test.drop(columns = ['id'])

ohe_columns = ['dept']
ord_columns = ['level', 'workload', 'last_year_promo', 'last_year_violations']
num_columns = ['employment_years', 'supervisor_evaluation', 'salary']

ordinal_categories = {
    'level': ['junior', 'middle', 'senior'],
    'workload': ['low', 'medium', 'high'],
    'last_year_promo': ['no', 'yes'],
    'last_year_violations': ['no', 'yes']}

RANDOM_STATE = 42

ohe_pipe = Pipeline([
    ('simpleImputer', SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False, drop = 'first'))])

ord_pipe = Pipeline([
    ('simpleImputer_before_ord', SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')),
    ('ord', OrdinalEncoder(categories=[ordinal_categories[col] for col in ord_columns], 
        handle_unknown = 'use_encoded_value',
        unknown_value=np.nan)),
    ('simpleImputer_after_ord', SimpleImputer(missing_values=np.nan, strategy = 'most_frequent'))])

num_pipe = Pipeline([
    ('simpleImputer_le', SimpleImputer(missing_values=np.nan, strategy = 'median')),
    ('scaler', StandardScaler())])

data_preprocessor = ColumnTransformer([
    ('ohe', ohe_pipe, ohe_columns),
    ('ord', ord_pipe, ord_columns),
    ('num', num_pipe, num_columns)],
    remainder = 'passthrough')

pipe_rf = Pipeline([
    ('preprocessor', data_preprocessor),
    ('model', RandomForestRegressor(random_state=RANDOM_STATE))])

train_job_sr_no_id = train_job_sr.drop(columns = ['id'])
train_job_sr_no_id = train_job_sr_no_id.drop_duplicates().reset_index(drop = True)

X_train_one = train_job_sr_no_id.drop(columns = ['job_satisfaction_rate'], axis = 1)
y_train_one = train_job_sr_no_id['job_satisfaction_rate']

X_test_one = hr_test.drop(columns = ['id', 'job_satisfaction_rate'])
y_test_one = hr_test['job_satisfaction_rate']

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    
    smape_value = 100 * np.mean(np.divide(numerator, denominator, 
                                        out=np.zeros_like(numerator), 
                                        where=denominator!=0))
    return smape_value
smape_scorer = make_scorer(
    smape,
    greater_is_better=False
)
# третья модель, RandomForest

rf_parameters = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 15, 20, None],
    'model__min_samples_leaf': [1, 2, 5],
    'model__max_features': ['sqrt', 'log2']
}

rf_cv = RandomizedSearchCV(
    pipe_rf,
    param_distributions = rf_parameters,
    n_jobs = -1,
    cv = 5,
    scoring = smape_scorer,
    n_iter = 10,
    random_state = 42,
    refit=True,
    return_train_score=True,
    verbose = 1
)

rf_cv.fit(X_train_one, y_train_one);

best_model_one = rf_cv.best_estimator_
y_pred_test_one = best_model_one.predict(X_test_one)
test_smape = smape(y_test_one, y_pred_test_one)


train_quit['job_satisfaction_rate'] = best_model_one.predict(train_quit.drop(columns = ['id', 'quit']))

ohe_columns = ['dept']
ord_columns = ['level', 'workload', 'last_year_promo', 'last_year_violations']
num_columns = ['employment_years', 'supervisor_evaluation', 'salary', 'job_satisfaction_rate'] 

ordinal_categories = {
    'level': ['junior', 'middle', 'senior'],
    'workload': ['low', 'medium', 'high'],
    'last_year_promo': ['no', 'yes'],
    'last_year_violations': ['no', 'yes']}

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

ohe_pipe = Pipeline([
    ('simpleImputer', SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False, drop = 'first'))])

ord_pipe = Pipeline([
    ('simpleImputer_before_ord', SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')),
    ('ord', OrdinalEncoder(categories = [ordinal_categories[col] for col in ord_columns],
        handle_unknown = 'use_encoded_value', unknown_value = np.nan)),
    ('simpleImputer_after_ord', SimpleImputer(missing_values = np.nan, strategy = 'most_frequent'))])

data_preprocessor = ColumnTransformer([
    ('ohe', ohe_pipe, ohe_columns),
    ('ord', ord_pipe, ord_columns),
    ('num', num_pipe, num_columns)])

pipe_rf = Pipeline([
    ('preprocessor', data_preprocessor),
    ('model', RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_depth=10))])

hr_test_two = test_features.merge(test_target_quit, on = 'id', how = 'inner')

X_train_two = train_quit.drop(columns=['quit', 'id'])
y_train_two = train_quit['quit']

X_test_two = hr_test_two.drop(columns=['id'])
y_test_two = hr_test_two['quit']

le = LabelEncoder()
le.classes_ = np.array(['no', 'yes'])

le.fit(y_train_two)  

y_train_two = le.transform(y_train_two)
y_test_two = le.transform(y_test_two)

# RandomForest
param_grid_rf = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, 15, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__class_weight': ['balanced', None]
}

rf_grid_cv = RandomizedSearchCV(
    pipe_rf,
    param_distributions = param_grid_rf,
    n_jobs = -1,
    cv = 5,
    scoring = 'roc_auc',
    n_iter = 10,
    random_state = 42,
    refit=True,
    return_train_score=True 
)

X_test_two['job_satisfaction_rate'] = best_model_one.predict(X_test_two)

rf_grid_cv.fit(X_train_two, y_train_two)
rf_model = rf_grid_cv.best_estimator_

y_pred_test_two = rf_model.predict(X_test_two)
y_proba_test_two = rf_model.predict_proba(X_test_two)[:, 1]
test_roc_auc = roc_auc_score(y_test_two, y_proba_test_two)

print(test_smape, test_roc_auc)