
import conexion_proyect
import pandas as pd
import tensorflow as tf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")



# Obtenemos los datos
data_estadofinanciero = conexion_proyect.api_estado_financiero
estado_financiero = data_estadofinanciero.get()
data_macro = conexion_proyect.api_datos_macro
datos_macro = data_macro.get()
data_patrimonio = conexion_proyect.api_patrimonio
patrimonio = data_patrimonio.get()
data_movimientos = conexion_proyect.api_movimientos
movimientos = data_movimientos.get()
data_indicadores = conexion_proyect.api_indicadores
indicadores = data_indicadores.get()

# Convertimos las variables a DataFrames si no lo son
estado_financiero_df = pd.DataFrame(estado_financiero)
datos_macro_df = pd.DataFrame(datos_macro)
patrimonio_df = pd.DataFrame(patrimonio)
movimientos_df = pd.DataFrame(movimientos)
indicadores_df = pd.DataFrame(indicadores)

# Convertir las fechas a un formato común (YYYY-MM)
for df in [estado_financiero_df, datos_macro_df, patrimonio_df, movimientos_df, indicadores_df]:
    df['fecha'] = pd.to_datetime(df['fecha']).dt.to_period('M').dt.start_time

# Ordenar DataFrames por fecha
estado_financiero_df = estado_financiero_df.sort_values('fecha')
datos_macro_df = datos_macro_df.sort_values('fecha')
patrimonio_df = patrimonio_df.sort_values('fecha')
movimientos_df = movimientos_df.sort_values('fecha')
indicadores_df = indicadores_df.sort_values('fecha')

# Eliminar columnas duplicadas y renombrar las que causan conflictos
estado_financiero_df = estado_financiero_df.drop(columns=['updated_at', 'created_at'], errors='ignore')
datos_macro_df = datos_macro_df.drop(columns=['updated_at', 'created_at'], errors='ignore')
patrimonio_df = patrimonio_df.drop(columns=['updated_at', 'created_at'], errors='ignore')
movimientos_df = movimientos_df.drop(columns=['updated_at', 'created_at'], errors='ignore')
indicadores_df = indicadores_df.drop(columns=['updated_at', 'created_at'], errors='ignore')

# Renombrar columnas para evitar conflictos
estado_financiero_df.rename(columns={'id_empresa': 'id_empresa_ef'}, inplace=True)
datos_macro_df.rename(columns={'id_empresa': 'id_empresa_dm'}, inplace=True)
patrimonio_df.rename(columns={'id_empresa': 'id_empresa_p'}, inplace=True)
movimientos_df.rename(columns={'id_empresa': 'id_empresa_m'}, inplace=True)
indicadores_df.rename(columns={'id_empresa': 'id_empresa_i'}, inplace=True)

# Realizar merge_asof para combinar DataFrames
df_completo = pd.merge_asof(estado_financiero_df, datos_macro_df, on='fecha', direction='nearest')
df_completo = pd.merge_asof(df_completo, patrimonio_df, on='fecha', direction='nearest')
df_completo = pd.merge_asof(df_completo, movimientos_df, on='fecha', direction='nearest')
df_completo = pd.merge_asof(df_completo, indicadores_df, on='fecha', direction='nearest')


# Ajustamos las opciones de visualización
pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)  # Ajusta el ancho de la salida
pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de cada columna




# Eliminar columnas con solo valores nulos
df_completo = df_completo.drop(columns=['id_empresa_ef', 'id_empresa_p', 'id_empresa_m', 'id_empresa_i','id_estado','id_macro','id_patrimonio','id_iug','id_indicador'])
#empezamos la seleccion de caracteristicas con el modelo lasso
# Filtrar las columnas de tipo datetime
date_columns = df_completo.select_dtypes(include=['datetime']).columns
df_sin_fechas = df_completo.drop(columns=date_columns)

# Inicializar un diccionario para almacenar los resultados
resultados = {}

# Definir un rango de valores para alpha
alphas = np.logspace(-3, 0, 100)  # Esto genera 100 valores de alpha entre 1e-3 y 1e0

# Iterar sobre cada variable objetivo (columnas), excluyendo las fechas
for target_column in df_sin_fechas.columns:
    # Eliminar la columna objetivo actual de las características (X)
    X = df_sin_fechas.drop(columns=[target_column])  # Características
    y = df_sin_fechas[target_column]  # Variable objetivo actual

    # Dividir el conjunto de datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ajustar el modelo Lasso usando validación cruzada con un rango de valores de alpha
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000).fit(X_train_scaled, y_train)

    # Extraer el mejor valor de alpha
    best_alpha = lasso_cv.alpha_
    #print(f"Mejor valor de alpha para la columna {target_column}: {best_alpha}")

    # Ajustar el modelo Lasso con el mejor valor de alpha encontrado
    lasso = Lasso(alpha=best_alpha, max_iter=100000, tol=0.0001)
    lasso.fit(X_train_scaled, y_train)

    # Obtener los coeficientes
    coeficientes = pd.Series(lasso.coef_, index=X.columns)

    # Filtrar solo los coeficientes no nulos (aquellos seleccionados por LASSO)
    coeficientes_no_nulos = coeficientes[coeficientes != 0]

    # Guardar resultados para la columna actual
    resultados[target_column] = {
        'alpha': best_alpha,
        'coeficientes': coeficientes_no_nulos,
        'numero_variables_seleccionadas': len(coeficientes_no_nulos)
    }

    #print(f"Variables seleccionadas por LASSO para {target_column}: {coeficientes_no_nulos.index.tolist()}")

# Convertir el diccionario de resultados en un DataFrame para una mejor visualización
resultados_df = pd.DataFrame(resultados).T

# Añadir las columnas de fechas nuevamente al DataFrame final
resultados_df['fecha'] = df_completo[date_columns].iloc[:, 0]  # Asumiendo que hay una sola columna de fecha

# Generar un nuevo DataFrame solo con las variables seleccionadas por LASSO
# Primero, identificamos todas las variables seleccionadas en algún momento
variables_seleccionadas = set()

for target_column, resultado in resultados.items():
    variables_seleccionadas.update(resultado['coeficientes'].index)

# Crear un DataFrame nuevo solo con las variables seleccionadas
df_seleccionado = df_sin_fechas[list(variables_seleccionadas)]

# Añadir la columna de fecha al nuevo DataFrame
df_completo = pd.concat([df_seleccionado, df_completo[date_columns]], axis=1)
print(df_completo)

#Empezamos el MODELO ARIMA
df_completo['fecha'] = pd.to_datetime(df_completo['fecha'])
df_completo.set_index('fecha', inplace=True)

# Asegúrate de que los datos son de frecuencia mensual (MS) o la que necesites
df_completo = df_completo.asfreq('MS')


# Función para comprobar la estacionariedad
def test_estacionariedad(serie):
    resultado = adfuller(serie)
    print('Estadístico ADF:', resultado[0])
    print('p-valor:', resultado[1])
    print('Valores críticos:')
    for key, value in resultado[4].items():
        print(f'\t{key}: {value}')


# Iterar sobre cada columna en df_completo (exceptuando la columna de fecha)
for variable in df_completo.columns:
    print(f'\nEvaluando variable: {variable}')

    # Convertir a numérico, forzando errores a NaN
    df_completo[variable] = pd.to_numeric(df_completo[variable], errors='coerce')

    # Paso 1: Comprobar estacionariedad
    test_estacionariedad(df_completo[variable].dropna())

    # Si la serie no es estacionaria, aplica transformaciones
    # Por ejemplo, puedes diferenciarla
    df_completo[f'{variable}_diff'] = df_completo[variable].diff().dropna()

    # Ajustar el modelo ARIMA (p, d, q) - selecciona p, d, q basado en tu análisis
    modelo = ARIMA(df_completo[variable].dropna(), order=(1, 1, 1))  # Cambia los parámetros si es necesario
    modelo_fit = modelo.fit()

    # Mostrar el resumen del modelo
    print(modelo_fit.summary())

    # Paso 3: Obtener residuos
    residuos = modelo_fit.resid

    # Graficar los residuos
    plt.figure(figsize=(12, 6))
    plt.plot(residuos)
    plt.title(f'Residuos del Modelo ARIMA para {variable}')
    plt.xlabel('Fecha')
    plt.ylabel('Residuos')
    plt.axhline(0, color='red', linestyle='--')
    plt.grid()
    plt.show()

    # Realizar predicciones
    predicciones = modelo_fit.forecast(steps=12)

    # Crear un DataFrame para las predicciones
    fechas_futuras = pd.date_range(start=df_completo.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    predicciones_df = pd.DataFrame(predicciones, index=fechas_futuras, columns=['Predicción'])

    # Graficar las predicciones junto con la serie original
    plt.figure(figsize=(12, 6))
    plt.plot(df_completo[variable], label='Datos Históricos')
    plt.plot(predicciones_df, label='Predicciones', marker='o', color='orange')
    plt.title(f'Predicciones del Modelo ARIMA para {variable}')
    plt.xlabel('Fecha')
    plt.ylabel(variable)
    plt.legend()
    plt.grid()
    plt.show()