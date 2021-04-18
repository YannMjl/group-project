# Project

# %%
# Load the patient data from "healthcare_data.csv" file.

import pandas as pd
from sklearn.pipeline import Pipeline

patients = pd.read_csv('healthcare_data.csv').fillna(0)
patients.describe()

# %%
# Regularize the data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

categorical_columns = [
    'gender',
    'hypertension',
    'heart_disease',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status',
    ]
numerical_columns = ['age', 'avg_glucose_level', 'bmi']

one_hot = OneHotEncoder(drop='first')

numerical_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('standardizer', StandardScaler()),
])

column_tx = ColumnTransformer([
    ('Categorical', one_hot, categorical_columns),
    ('Numerical', numerical_pipeline, numerical_columns),],
    remainder='drop')

column_tx.fit(patients)

stroke_patients = patients[patients.stroke == 1]
stroke_patients = column_tx.transform(stroke_patients)

column_titles = column_tx.named_transformers_['Categorical'].get_feature_names(categorical_columns).tolist()

column_titles += numerical_columns

stroke_patients = pd.DataFrame(
    stroke_patients,
    columns=column_titles)

test_patients = pd.DataFrame(column_tx.transform(patients), columns=column_titles)

training_strokes = patients[patients.stroke == 1].loc[:, 'stroke']
expected_strokes = patients.loc[:, 'stroke']

# %%
# model it

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def print_metrics(y_true, y_predict):
    print(f'R2 Score: {r2_score(y_true, y_predict)}')
    print(f'MSE: {mean_squared_error(y_true, y_predict)}')
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = (tp) / (tp + fn)
    precision = (tp) / (tp + fp)
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print()

model = svm.OneClassSVM()
model.fit(stroke_patients, training_strokes)

predicted_strokes = model.predict(test_patients)
predicted_strokes = list(map(
    lambda x: max(x, 0),
    predicted_strokes))

print_metrics(expected_strokes, predicted_strokes)

strokes = patients.loc[:, "stroke"]
standardized_patients = column_tx.transform(patients)

model = svm.SVC(
    gamma='auto',
    kernel='rbf',
    C=10)
model.fit(standardized_patients, strokes)
y_predict = model.predict(standardized_patients)

print_metrics(strokes, y_predict)

X_train, X_test, y_train, y_test = train_test_split(
    standardized_patients,
    strokes,
    test_size=.4,
    random_state=4331)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print_metrics(y_test, y_predict)

# %%
