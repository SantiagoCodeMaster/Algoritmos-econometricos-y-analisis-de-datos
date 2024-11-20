import requests
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.linear_model import Lasso
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Ruta a los modelos entrenados
path_modelos_lasso = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\modelos_lasso"
path_modelos_arima = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\modelos_arima"
path_modelo_rnn = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\rnn_model.h5"

# Función para obtener los datos de la API de Laravel
def obtener_datos_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Suponiendo que los datos están en formato JSON
    else:
        raise Exception(f"Error al obtener los datos: {response.status_code}")

# Función para limpiar los datos (esto es genérico y debe adaptarse a tu formato de datos)
def limpiar_datos(datos):
    df = pd.DataFrame(datos)
    # Eliminar columnas innecesarias o valores nulos
    df = df.dropna(axis=1, how='any')  # Elimina columnas con valores nulos
    # Otras limpiezas pueden incluir escalado, transformación, etc.
    return df

# Cargar el modelo Lasso
def cargar_modelo_lasso():
    modelos_lasso = {}
    for archivo in os.listdir(path_modelos_lasso):
        if archivo.endswith(".pkl"):
            with open(os.path.join(path_modelos_lasso, archivo), 'rb') as f:
                modelo_name = archivo.split('.')[0]  # Nombre del modelo basado en el archivo
                modelos_lasso[modelo_name] = pickle.load(f)
    return modelos_lasso

# Cargar el modelo ARIMA
def cargar_modelo_arima():
    modelos_arima = {}
    for archivo in os.listdir(path_modelos_arima):
        if archivo.endswith(".pkl"):
            with open(os.path.join(path_modelos_arima, archivo), 'rb') as f:
                modelo_name = archivo.split('.')[0]  # Nombre del modelo basado en el archivo
                modelos_arima[modelo_name] = pickle.load(f)
    return modelos_arima

# Cargar el modelo RNN
def cargar_modelo_rnn():
    return load_model(path_modelo_rnn)

# Selección de características usando Lasso
def seleccionar_caracteristicas_lasso(df, modelo_lasso):
    X = df.values  # Convertir a array de numpy
    predicciones = modelo_lasso.predict(X)
    return predicciones

# Modelo ARIMA para residuos
def predecir_arima(df, modelo_arima):
    X = df.values
    arima_model = ARIMA(X, order=(1, 1, 1))  # Modelo ARIMA básico
    arima_model_fit = arima_model.fit()
    residuos = arima_model_fit.resid
    return residuos

# Predecir con la RNN
def predecir_rnn(df, modelo_rnn):
    X = df.values
    X_scaled = StandardScaler().fit_transform(X)  # Escalar los datos
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))  # Formato RNN
    predicciones = modelo_rnn.predict(X_scaled)
    return predicciones

# Función principal
def procesar_datos_y_predicciones(url_api):
    # Obtener los datos desde la API de Laravel
    datos = obtener_datos_api(url_api)

    # Limpiar los datos
    df_limpio = limpiar_datos(datos)

    # Cargar los modelos
    modelos_lasso = cargar_modelo_lasso()
    modelos_arima = cargar_modelo_arima()
    modelo_rnn = cargar_modelo_rnn()

    # Procesar cada empresa (suponiendo que cada conjunto de datos corresponde a una empresa)
    predicciones_totales = {}
    for empresa, data in df_limpio.groupby('empresa_id'):
        # Selección de características con Lasso
        modelo_lasso_actual = modelos_lasso.get(f"lasso_{empresa}")
        if modelo_lasso_actual:
            variables_lasso = seleccionar_caracteristicas_lasso(data, modelo_lasso_actual)

            # Predicción con ARIMA usando los residuos
            modelo_arima_actual = modelos_arima.get(f"arima_{empresa}")
            if modelo_arima_actual:
                residuos_arima = predecir_arima(data, modelo_arima_actual)

                # Predicción con RNN usando los residuos y variables seleccionadas
                data_residuos = pd.DataFrame(residuos_arima, columns=['residuos'])
                data_final = pd.concat([data[[ 'activos_biologicos', 'activos_no_corrientes_venta', 'activos_por_derechos',
    'activo_por_impuesto_diferido', 'capital_emitido', 'ciclo_caja', 'ciclo_operacion',
    'cobertura_deuda', 'cobertura_intereses', 'costos_venta', 'costo_deuda_financiera',
    'costo_pasivo_total', 'costo_patrimonio', 'deudores_comerciales', 'deudores_comerciales_no_corriente',
    'diferencia_cambio_activos_pasivos', 'diferencia_cambio_activos_pasivos_no_operativos', 'dividendos',
    'efectivo_equivalentes', 'gastos_administracion', 'gastos_financieros', 'gastos_produccion',
    'gastos_venta', 'impuestos_por_pagar_corriente', 'impuesto_renta_corriente', 'impuesto_renta_diferido',
    'inflacion', 'ingresos_financieros', 'inventarios', 'inversiones_asociadas_negocios', 'kwn',
    'margen_utilidad_antes_impuestos', 'margen_utilidad_bruta', 'margen_utilidad_neta', 'margen_utilidad_operacional',
    'nivel_apalancamiento', 'nivel_endeudamiento', 'nivel_endeudamiento_corto_plazo', 'nivel_endeudamiento_largo_plazo',
    'obligaciones_financieras_corrientes', 'obligaciones_financieras_no_corriente', 'operaciones_discontinuadas',
    'otros_activos_corrientes', 'otros_activos_intangibles', 'otros_activos_no_corrientes', 'otros_ingresos_netos',
    'otro_resultado_integral', 'participaciones_no_controladoras', 'participacion_asociadas_negocios',
    'pasivos_por_derecho_corriente', 'pasivos_por_derecho_no_corriente', 'pasivo_beneficios_empleados_corriente',
    'pasivo_beneficios_empleados_no_corriente', 'pasivo_por_impuesto_diferido', 'patrimonio_atributable_controladoras',
    'pib', 'plusvalia', 'prima_emision_capital', 'propiedades_inversion', 'propiedades_planta_equipo',
    'proveedores_cuentas_pagar_corriente', 'proveedores_cuentas_pagar_no_corriente', 'provisiones_corriente',
    'provisiones_no_corriente', 'prueba_acida', 'razon_corriente', 'rentabilidad_operativa', 'rentabilidad_patrimonio',
    'reservas_utilidades_acumuladas', 'roi', 'rotacion_activos', 'rotacion_ctas_x_cobrar', 'rotacion_inventarios',
    'rotacion_proveedores', 'tasa_desempleo', 'tasa_interes', 'utilidad_antes_impuestos', 'utilidad_bruta',
    'utilidad_neta_periodo', 'utilidad_operativa', 'utilidad_periodo', 'utilidad_periodo_operaciones_continuadas']], data_residuos], axis=1)
                predicciones_rnn = predecir_rnn(data_final, modelo_rnn)

                # Guardar las predicciones
                predicciones_totales[empresa] = predicciones_rnn
            else:
                print(f"Modelo ARIMA no encontrado para la empresa {empresa}")
        else:
            print(f"Modelo Lasso no encontrado para la empresa {empresa}")

    # Enviar predicciones a Laravel
    url_predicciones = "http://tu-laravel-api.com/predicciones"
    response = requests.post(url_predicciones, json=predicciones_totales)
    if response.status_code == 200:
        print("Predicciones enviadas correctamente a Laravel")
    else:
        print(f"Error al enviar las predicciones: {response.status_code}")

# Llamada a la función principal
url_api = "http://tu-laravel-api.com/datos"
procesar_datos_y_predicciones(url_api)
