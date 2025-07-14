# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005..
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


# Paso previo. Librerías necesarias
import gzip
import json
import os
import pickle
import zipfile
from glob import glob

import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (  # type: ignore
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

# Paso 1. Cargar archivo
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, compression='zip')

# Paso 2.Limpieza del dataset
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df = new_df.rename(columns={'default payment next month': 'default'})
    new_df = new_df.drop(columns=['ID'])
    new_df = new_df.loc[new_df["MARRIAGE"] != 0]
    new_df = new_df.loc[new_df["EDUCATION"] != 0]
    new_df["EDUCATION"] = new_df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    return new_df

# Paso 3. Crear pipeline
def create_pipeline() -> Pipeline:
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# Paso 4. Optimizar hiperparametros
def create_estimator(pipeline: Pipeline) -> GridSearchCV:
    param_grid = {
	    'classifier__n_estimators': [50, 100, 200],
	    'classifier__max_depth': [None, 5, 10, 20],
	    'classifier__min_samples_split': [2, 5, 10],
	    'classifier__min_samples_leaf': [1, 2, 4]
    }

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2,
        refit=True
    )

# Paso 5. Guardar modelo


def save_model(path: str, estimator: GridSearchCV):
    print(f"Saving model to: {os.path.abspath(path)}")  # Debug: Print absolute path
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
    with gzip.open(path, 'wb') as f:
        pickle.dump(estimator, f)
    print(f"Model saved successfully: {os.path.exists(path)}")  # Debug: Confirm file exists


# Paso 6.Calcular métricas de precisión
def calculate_precision_metrics(dataset_name: str, y, y_pred) -> dict: 
    return {
        'type': 'metrics',
        'dataset': dataset_name,
	    'precision': precision_score(y, y_pred, zero_division=0),
	    'balanced_accuracy': balanced_accuracy_score(y, y_pred),
	    'recall': recall_score(y, y_pred, zero_division=0),
	    'f1_score': f1_score(y, y_pred, zero_division=0)
    }

# Paso 7.Calcular matriz de confusión
def calculate_confusion_metrics(dataset_name: str, y, y_pred) -> dict:
    cm = confusion_matrix(y, y_pred)
    return {
        'type': 'cm_matrix',
	    'dataset': dataset_name,
	    'true_0': {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
	    'true_1': {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
    }
    
#Paso final. Ejecutar 
def run():
    input_files_path = 'files/input/'
    models_files_path = 'files/models/'
    os.makedirs(models_files_path, exist_ok=True)

    test_df = load_dataset(os.path.join(input_files_path, 'test_data.csv.zip'))
    train_df = load_dataset(os.path.join(input_files_path, 'train_data.csv.zip'))

    test_df = clean_dataset(test_df)
    train_df = clean_dataset(train_df)

    x_test = test_df.drop(columns=['default'])
    y_test = test_df['default']

    x_train = train_df.drop(columns=['default'])
    y_train = train_df['default']

    pipeline = create_pipeline()

    estimator = create_estimator(pipeline)
    estimator.fit(x_train, y_train)

    save_model(
        os.path.join(models_files_path, 'model.pkl.gz'),
        estimator,
    )

    

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = calculate_precision_metrics(
        'test',
        y_test,
        y_test_pred
    )
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = calculate_precision_metrics(
        'train',
        y_train,
        y_train_pred
    )

    test_confusion_metrics = calculate_confusion_metrics('test', y_test, y_test_pred)
    train_confusion_metrics = calculate_confusion_metrics('train', y_train, y_train_pred)

    with open('files/output/metrics.json', 'w') as file:
        file.write(json.dumps(train_precision_metrics)+'\n')
        file.write(json.dumps(test_precision_metrics)+'\n')
        file.write(json.dumps(train_confusion_metrics)+'\n')
        file.write(json.dumps(test_confusion_metrics)+'\n')


if __name__ == "__main__":
    run()