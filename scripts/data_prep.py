
# Importar bibliotecas
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

# Rutas de archivos
final_file_path = "data/processed_data/aapl_final_data.csv"
macro_file_path = "data/external/Bases_de_datos.xlsx"  # Ajusta esta ruta según la ubicación del archivo

# Cargar datos de mercado
df = pd.read_csv(final_file_path)

# Asegurarse de que la columna 'Date' está en formato de fecha si existe
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Cargar datos macroeconómicos
try:
    # Intentar cargar el archivo macroeconómico
    macro_df = pd.read_excel(macro_file_path, sheet_name='macro_data_combined')
    if 'Date' in macro_df.columns:
        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
        df = pd.merge(df, macro_df, on='Date', how='left')
    else:
        print("Error: La columna 'Date' no se encuentra en macro_data_combined.")
except FileNotFoundError:
    print("Archivo Bases de datos.xlsx no encontrado en la ruta especificada.")
except Exception as e:
    print(f"Ocurrió un error al cargar los datos macroeconómicos: {e}")

# Verificar la estructura del DataFrame combinado
print("Columnas en el DataFrame combinado:", df.columns)
print("Primeras filas del DataFrame combinado:", df.head())

# Dividir datos en características y objetivo
X = df.drop(columns=['Target', 'Date'], errors='ignore')  # Eliminar 'Date' y 'Target'
y = df['Target']

# División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular el peso de las clases
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
scale_pos_weight = class_weights[0] / class_weights[1]

# Configurar el modelo XGBoost y la búsqueda de hiperparámetros
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Definir los parámetros para GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [scale_pos_weight, 1.0, 1.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2.0]
}

# Realizar la búsqueda de hiperparámetros
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Ajustar el modelo
grid_search.fit(X_train, y_train)

# Evaluar el mejor modelo
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Mostrar métricas finales
print("Precisión del modelo en el conjunto de prueba:", best_model.score(X_test, y_test))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred, zero_division=0))

# Crear archivo para TradingView
try:
    # Restaurar la columna 'Date' en X_test para el archivo de señales
    test_data = X_test.copy()
    test_data['Date'] = df.loc[X_test.index, 'Date']
    test_data['Prediction'] = y_pred
    test_data['Buy_Signal'] = np.where(test_data['Prediction'] == 1, test_data['Date'], None)
    test_data['Sell_Signal'] = np.where(test_data['Prediction'] == 0, test_data['Date'], None)
    signals_file_path = "data/processed_data/predicciones_para_tradingview.csv"
    test_data[['Date', 'Buy_Signal', 'Sell_Signal']].to_csv(signals_file_path, index=False)
    print(f"Archivo de señales guardado en: {signals_file_path}")
except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Ocurrió un error al generar el archivo de señales: {e}")

# Guardar el modelo entrenado
joblib.dump(best_model, "models/trained_model_xgboost_optimized.pkl")
