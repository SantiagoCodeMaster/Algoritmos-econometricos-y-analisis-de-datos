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

# Creamos la clase Datosframe para convertir los datos recibidos en DataFrame
class Datosframe:
    def __init__(self, variable):
        self.variable = pd.DataFrame(variable)

    def convertir_frame(self):
        # Preparar datos
        self.variable['fecha'] = pd.to_datetime(self.variable['fecha']).dt.to_period('M').dt.start_time
        self.variable = self.variable.sort_values('fecha')
        self.variable = self.variable.drop(columns=['updated_at', 'created_at'], errors='ignore')
        pd.set_option('display.max_rows', None)  # Muestra todas las filas
        pd.set_option('display.max_columns', None)  # Muestra todas las columnas
        pd.set_option('display.width', None)  # Ajusta el ancho de la salida
        pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de cada columna

    def combinar_frames(self, otro_df):
        # Combina con otro DataFrame
        df_completo = pd.merge_asof(self.variable, otro_df, on='fecha', direction='nearest')
        return df_completo


if __name__ == "__main__":
    # Obtener datos
    data_estadofinanciero = conexion_proyect.api_estado_financiero.get()
    estado_financiero_df = Datosframe(data_estadofinanciero)
    estado_financiero_df.convertir_frame()  # Convertir pero mantener la instancia

    data_macro = conexion_proyect.api_datos_macro.get()
    macro_df = Datosframe(data_macro)
    macro_df.convertir_frame()

    data_patrimonio = conexion_proyect.api_patrimonio.get()
    patrimonio_df = Datosframe(data_patrimonio)
    patrimonio_df.convertir_frame()

    data_movimientos = conexion_proyect.api_movimientos.get()
    movimientos_df = Datosframe(data_movimientos)
    movimientos_df.convertir_frame()

    data_indicadores = conexion_proyect.api_indicadores.get()
    indicadores_df = Datosframe(data_indicadores)
    indicadores_df.convertir_frame()

    # Ejemplo de combinación de DataFrames
    df_estadofinanciero = estado_financiero_df.combinar_frames(macro_df.variable)
    df_patrimonio = patrimonio_df.combinar_frames(macro_df.variable)
    df_movimientos = movimientos_df.combinar_frames(macro_df.variable)
    df_indicadores = indicadores_df.combinar_frames(macro_df.variable)

    print(df_estadofinanciero)
    print(df_patrimonio)
    print(df_movimientos)
    print(df_indicadores)