#importamos las librerias

import conexion_proyect
import pandas as pd
import tensorflow as tf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
warnings.filterwarnings("ignore")

# Cargar df_definitivo
df_definitivo = pd.read_csv("df_definitivo.csv", index_col=0)

# Cargar residuos_df
residuos_df = pd.read_csv("residuos_df.csv", index_col=0)

class RNNModel:
    def __init__(self, df_definitivo, residuos_df, window_size=12, epochs=50, batch_size=16):
        # Inicialización de los datos y parámetros
        self.df_definitivo = df_definitivo
        self.residuos_df = residuos_df
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size

        # Inicialización de los normalizadores
        self.scaler_definitivo = MinMaxScaler()
        self.scaler_residuos = MinMaxScaler()

        # Espacio para los datos procesados
        self.X = None
        self.y = None
        self.model = None

    def normalize_data(self):
        # Normalizar los datos definitivos y residuos
        self.df_definitivo_scaled = self.scaler_definitivo.fit_transform(self.df_definitivo)
        self.residuos_df_scaled = self.scaler_residuos.fit_transform(self.residuos_df)

    def prepare_data(self):
        # Listas para almacenar los datos de entrenamiento
        X = []
        y = []

        # Crear secuencias con la ventana de tiempo
        for i in range(len(self.df_definitivo_scaled) - self.window_size):
            if i + self.window_size < len(self.residuos_df_scaled):  # Asegurar que no exceda el límite
                X.append(self.df_definitivo_scaled[i:i + self.window_size])
                y.append(self.residuos_df_scaled[i + self.window_size])

        # Convertir las listas en arrays numpy
        self.X = np.array(X)
        self.y = np.array(y)

        # Verificar las formas de X e y antes de continuar
        print("Forma de X antes de reshaping:", self.X.shape)
        print("Forma de y:", self.y.shape)

        if self.X.ndim == 3:
            print("self.X ya tiene la forma adecuada.")
        else:
            self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))

        print("Forma final de X después de preparar datos:", self.X.shape)
        print("Forma final de y después de preparar datos:", self.y.shape)

    def build_model(self, units=50, dropout_rate=0.2):
        # Definir la estructura de la RNN con LSTM
        self.model = Sequential()
        self.model.add(LSTM(units, activation='relu', input_shape=(self.window_size, self.X.shape[2])))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dense(self.y.shape[1]))  # Capa de salida para las predicciones

        # Compilar el modelo
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self):
        # Entrenar el modelo RNN
        self.history = self.model.fit(self.X, self.y, epochs=self.epochs,
                                      batch_size=self.batch_size, validation_split=0.2, verbose=1)

    def evaluate_model(self):
        # Evaluar el modelo en los datos de entrenamiento
        loss = self.model.evaluate(self.X, self.y)
        print("Pérdida en el conjunto de entrenamiento:", loss)
        # Gráfica del historial de entrenamiento
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Pérdida en Entrenamiento')
        plt.plot(self.history.history['val_loss'], label='Pérdida en Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida (MSE)')
        plt.title('Pérdida durante el Entrenamiento y Validación')
        plt.legend()
        plt.grid(True)
        plt.show()

        return loss

    def predict(self):
        # Realizar predicciones con el modelo
        y_pred = self.model.predict(self.X)
        return y_pred

    def inverse_transform_predictions(self, y_pred):
        # Desnormalizar las predicciones y valores reales
        y_pred_rescaled = self.scaler_residuos.inverse_transform(y_pred)
        y_rescaled = self.scaler_residuos.inverse_transform(self.y)
        return y_pred_rescaled, y_rescaled

    def hyperparameter_tuning(self, param_grid):
        best_model = None
        best_loss = float('inf')

        for params in ParameterGrid(param_grid):
            print(f"Entrenando con parámetros: {params}")
            self.build_model(units=params['units'], dropout_rate=params['dropout_rate'])
            self.train_model()
            loss = self.evaluate_model()

            if loss < best_loss:
                best_loss = loss
                best_model = self.model

        print("Mejor modelo encontrado con pérdida:", best_loss)
        self.model = best_model  # Actualizar el modelo con el mejor encontrado

    def evaluate_and_correct_model(self):
        # Dividir los datos en conjuntos de entrenamiento, validación y prueba
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Entrenar el modelo con el conjunto de entrenamiento
        self.X, self.y = X_train, y_train  # Asignar datos de entrenamiento
        self.train_model()

        # Evaluar el modelo en el conjunto de entrenamiento
        train_loss = self.evaluate_model()

        # Evaluar el modelo en el conjunto de validación
        self.X, self.y = X_val, y_val  # Asignar datos de validación
        val_loss = self.evaluate_model()

        # Evaluar el modelo en el conjunto de prueba
        self.X, self.y = X_test, y_test  # Asignar datos de prueba
        test_loss = self.evaluate_model()

        # Comparar las pérdidas
        print(f"Pérdida de Entrenamiento: {train_loss}")
        print(f"Pérdida de Validación: {val_loss}")
        print(f"Pérdida de Prueba: {test_loss}")

        # Detectar sobreajuste
        if val_loss > train_loss:
            print("El modelo puede estar sobreajustado. Ajustando hiperparámetros...")

            # Ajuste de hiperparámetros
            param_grid = {
                'units': [50, 100],  # Ejemplo de unidades en la capa LSTM
                'dropout_rate': [0.2, 0.3]  # Ejemplo de tasa de abandono
            }

            # Reentrenar el modelo con los nuevos hiperparámetros
            self.hyperparameter_tuning(param_grid)
            print("Hiperparámetros ajustados. Reevaluando el modelo...")

            # Evaluar de nuevo
            self.X, self.y = X_train, y_train
            new_train_loss = self.evaluate_model()

            self.X, self.y = X_val, y_val
            new_val_loss = self.evaluate_model()

            print(f"Nueva Pérdida de Entrenamiento: {new_train_loss}")
            print(f"Nueva Pérdida de Validación: {new_val_loss}")

            if new_val_loss < val_loss:
                print("Los ajustes han mejorado el rendimiento del modelo.")
            else:
                print("Los ajustes no mejoraron el rendimiento del modelo.")

        else:
            print("El modelo no muestra signos de sobreajuste.")


# Definir los parámetros para el ajuste
param_grid = {
    'units': [50, 100],  # Número de unidades en la capa LSTM
    'dropout_rate': [0.2, 0.3]  # Tasa de abandono
}

# Instanciar la clase con los DataFrames
rnn_model = RNNModel(df_definitivo, residuos_df)

# Ejecutar los métodos paso a paso
rnn_model.normalize_data()         # Paso 1: Normalizar datos
rnn_model.prepare_data()           # Paso 2: Preparar los datos para la RNN
rnn_model.hyperparameter_tuning(param_grid)  # Paso 3: Ajustar hiperparámetros
loss = rnn_model.evaluate_model()  # Paso 4: Evaluar el modelo
rnn_model.evaluate_and_correct_model() #Paso 5 : verificar el sobreajuste

# Realizar predicciones y desnormalizarlas
y_pred = rnn_model.predict()
y_pred_rescaled, y_rescaled = rnn_model.inverse_transform_predictions(y_pred)

# Imprimir predicciones y variables objetivo
print("Predicciones desnormalizadas:", y_pred_rescaled)
print("Valores reales desnormalizados:", y_rescaled)
print("Variables que el modelo está prediciendo:", residuos_df.columns.tolist())


# Convertir `y_pred_rescaled` a un DataFrame y asignar los nombres de las variables
df_predicciones = pd.DataFrame(data=y_pred_rescaled, columns=residuos_df.columns)

# Limitar el número de decimales a 2
df_predicciones = df_predicciones.round(3)


# Ajustamos las opciones de visualización
pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)  # Ajusta el ancho de la salida
pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de cada columna

# Obtener la última fecha de los datos originales
ultima_fecha = df_definitivo.index[-1]

# Asegurarse de que `ultima_fecha` sea un objeto datetime
if isinstance(ultima_fecha, str):
    ultima_fecha = pd.to_datetime(ultima_fecha)

# Determinar cuántas predicciones tienes
num_predicciones = len(df_predicciones)

# Generar las fechas mensuales basadas en la última fecha de los datos originales
fechas_predicciones = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), periods=num_predicciones, freq='MS')

# Asignar las fechas generadas como una nueva columna en df_predicciones
df_predicciones['fecha'] = fechas_predicciones

# Reorganizar el DataFrame para que la fecha esté a la izquierda
df_predicciones = df_predicciones[['fecha'] + [col for col in df_predicciones.columns if col != 'fecha']]

# Mostrar el DataFrame con las fechas de predicción
print("Predicciones desnormalizadas con fechas mensuales:\n", df_predicciones)







