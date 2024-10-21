# Importamos librerías
import conexion_proyect
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.linear_model import Lasso
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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

# Imprimir el DataFrame combinado
print("DataFrame combinado:")
print(df_completo)



