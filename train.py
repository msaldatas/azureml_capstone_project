# AzureML imports
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
# Scikit Learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Other imports
import pandas as pd
import numpy as np
import argparse
import os
import joblib

url = r'https://raw.githubusercontent.com/msaldatas/pima_data/main/pima_diabetes.csv'
ds = TabularDatasetFactory.from_delimited_files(url, validate=True, include_path=False, infer_column_types=True, set_column_types=None, separator=',', header=True, partition_format=None, support_multi_line=False, empty_as_string=False, encoding='utf8')
df = ds.to_pandas_dataframe()

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

df.shape

impute_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in impute_columns:
    mean = df[column].mean()
    df[column].fillna(value=mean, inplace=True)

X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]

y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(X_train, y_train)
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')
    
    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
