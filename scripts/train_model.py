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

# Cargar datos macroeconómicos
try:
    # Intentar cargar el archivo macroeconómico
    macro_df = pd.read_excel(macro_file_path, sheet_name='macro_data_combined')
    
    # Asegurarse de que las columnas 'Date' están en ambos DataFrames y son de tipo fecha
    if 'Date' in macro_df.columns:
        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    else:
        print("Error: La columna 'Date' no se encuentra en macro_data_combined.")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print("Error: La columna 'Date' no se encuentra en aapl_final_data.csv.")

    # Unir ambos DataFrames usando la columna 'Date' si ambas columnas están presentes
    if 'Date' in df.columns and 'Date' in macro_df.columns:
        df = pd.merge(df, macro_df, on='Date', how='left')
    else:
        print("La unión no se realizó porque falta la columna 'Date' en uno de los archivos.")
    
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

# Guardar el modelo entrenado
joblib.dump(best_model, "models/trained_model_xgboost_optimized.pkl")

# ---- Exportar señales para TradingView ----
# Crear DataFrame de resultados con fechas y precios de cierre de X_test
test_data = X_test.copy()
test_data['Date'] = df.loc[X_test.index, 'Date']
test_data['Close'] = df.loc[X_test.index, 'Close']
test_data['Signal'] = y_pred  # 1 para compra, 0 para venta

# Guardar las señales en un archivo CSV para TradingView
output_file_path = 'data/processed_data/predicciones_para_tradingview.csv'
test_data[['Date', 'Close', 'Signal']].to_csv(output_file_path, index=False)
print(f"Archivo de señales guardado en: {output_file_path}")
