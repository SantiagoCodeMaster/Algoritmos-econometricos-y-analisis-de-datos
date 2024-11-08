#importamos la librerias
import joblib
import os
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from keras.models import load_model
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

df_completo.to_excel("df_datosanalisis.xlsx", index=False)
print(df_completo.columns)

#iniciamos modelo LASSO

# Definir el directorio donde se guardarán los modelos
os.makedirs("modelos_lasso", exist_ok=True)

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

    # Guardar el modelo Lasso entrenado
    nombre_modelo = f"modelos_lasso/lasso_{target_column}.pkl"
    joblib.dump(lasso, nombre_modelo)
    print(f"Modelo LASSO para {target_column} guardado en {nombre_modelo}")

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

    print(f"Variables seleccionadas por LASSO para {target_column}: {coeficientes_no_nulos.index.tolist()}")

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


#EMPEZAMOS red nueronal recurrente# Empezamos el MODELO ARIMA
#indice fecha como incio
df_completo['fecha'] = pd.to_datetime(df_completo['fecha'])
df_completo.set_index('fecha', inplace=True)
#datos mensuales
df_completo = df_completo.asfreq('MS')

# Crear un directorio para guardar los modelos
os.makedirs("modelos_arima", exist_ok=True)

# Función para comprobar la estacionariedad
def test_estacionariedad(serie):
    resultado = adfuller(serie)
    print('Estadístico ADF:', resultado[0])
    print('p-valor:', resultado[1])
    print('Valores críticos:')
    for key, value in resultado[4].items():
        print(f'\t{key}: {value}')

# Función para optimizar y entrenar el modelo ARIMA
def optimizar_arima(serie):
    modelo = auto_arima(serie, start_p=0, max_p=3, start_q=0, max_q=3,
                        seasonal=True, m=12, trace=True, error_action='ignore',
                        suppress_warnings=True, stepwise=True)
    return modelo

# Función para evaluar el modelo ARIMA
def evaluar_modelo_arima(predicciones, reales):
    mae = mean_absolute_error(reales, predicciones)
    mse = mean_squared_error(reales, predicciones)
    mape = mean_absolute_percentage_error(reales, predicciones)
    return {"MAE": mae, "MSE": mse, "MAPE": mape}

# Inicializar un diccionario para almacenar los resultados
resultados_evaluacion = {}
residuos_dict = {}  # Para guardar los residuos de cada variable

for variable in df_completo.columns:
    print(f'\nEvaluando variable: {variable}')

    # Asegurarse de que los datos son numéricos
    df_completo[variable] = pd.to_numeric(df_completo[variable], errors='coerce')

    # Paso 1: Comprobar estacionariedad
    test_estacionariedad(df_completo[variable].dropna())

    # Aplicar diferenciación si es necesario
    if adfuller(df_completo[variable].dropna())[1] > 0.05:  # Si no es estacionaria
        df_completo[f'{variable}_diff'] = df_completo[variable].diff().dropna()
        serie_a_usar = df_completo[f'{variable}_diff'].dropna()
    else:
        serie_a_usar = df_completo[variable].dropna()

    # Paso 2: Optimizar y entrenar el modelo ARIMA
    modelo_arima = optimizar_arima(serie_a_usar)
    modelo_fit = modelo_arima.fit(serie_a_usar)
    print(modelo_fit.summary())

    # Guardar el modelo ARIMA entrenado
    nombre_modelo = f"modelos_arima/arima_{variable}.pkl"
    joblib.dump(modelo_fit, nombre_modelo)
    print(f"Modelo ARIMA para {variable} guardado en {nombre_modelo}")

    # Paso 3: Obtener y graficar los residuos
    residuos = modelo_fit.resid()
    residuos = np.array(residuos)
    residuos_dict[variable] = residuos

    # Paso 4: Realizar predicciones
    n_periods = 12
    predicciones = modelo_fit.predict(n_periods=n_periods)

    # Obtener fechas para el futuro y valores reales
    fechas_futuras = pd.date_range(start=df_completo.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    predicciones_df = pd.DataFrame(predicciones, index=fechas_futuras, columns=['Predicción'])
    valores_reales = df_completo[variable][-n_periods:]

    # Paso 5: Evaluar el modelo y almacenar las métricas
    metricas = evaluar_modelo_arima(predicciones, valores_reales)
    resultados_evaluacion[variable] = metricas
    print(f"Evaluación para {variable}: {metricas}")

# Seleccionar variables con mejor desempeño
umbral_mae = 0.5  # Ajusta este umbral según tu contexto
variables_seleccionadas = [var for var, metrics in resultados_evaluacion.items() if metrics['MAE'] < umbral_mae]

print("\nVariables Seleccionadas basadas en el MAE:")
print(variables_seleccionadas)

# Crear el DataFrame df_definitivo con las variables seleccionadas
df_definitivo = df_completo[variables_seleccionadas].copy()

# Revisar longitudes de los residuos
longitudes_residuos = {var: len(residuos) for var, residuos in residuos_dict.items()}
print("Longitudes de los residuos para cada variable:")
print(longitudes_residuos)

# Opcional: Para hacer que todas las longitudes sean iguales
# Encuentra la longitud mínima
min_length = min(longitudes_residuos.values())

# Recortar o rellenar los residuos
for var in residuos_dict.keys():
    if len(residuos_dict[var]) > min_length:
        residuos_dict[var] = residuos_dict[var][:min_length]  # Recortar
    elif len(residuos_dict[var]) < min_length:
        # Rellenar con NaN (opcional)
        residuos_dict[var] = np.pad(residuos_dict[var], (0, min_length - len(residuos_dict[var])), 'constant', constant_values=np.nan)

# Crear el DataFrame de residuos
residuos_df = pd.DataFrame(residuos_dict)
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

    def save_model(self, filename='rnn_model.h5'):
        # Guardar el modelo entrenado
        self.model.save(filename)
        print(f"Modelo guardado como {filename}")

    def load_model(self, filename='rnn_model.h5'):
        # Cargar el modelo previamente guardado
        self.model = load_model(filename)
        print(f"Modelo cargado desde {filename}")

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
rnn_model.evaluate_and_correct_model() # Paso 5: Verificar el sobreajuste

# Realizar predicciones y desnormalizarlas
y_pred = rnn_model.predict()
y_pred_rescaled, y_rescaled = rnn_model.inverse_transform_predictions(y_pred)

# Guardar el modelo
rnn_model.save_model()