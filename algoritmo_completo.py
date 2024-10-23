# Importamos librerías
import conexion_proyect
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.linear_model import Lasso
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


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


numeric_columns = df_completo.columns[1:]  # Esto selecciona todas las columnas desde la segunda

# Función para convertir a numérico y manejar errores
def to_numeric(series):
    return pd.to_numeric(series, errors='coerce')  # Usa 'coerce' para convertir valores no convertibles a NaN

# Aplicamos la conversión a las columnas numéricas
df_completo= df_completo.copy()  # Hacemos una copia del DataFrame original para limpiar
df_completo[numeric_columns] = df_completo[numeric_columns].apply(to_numeric)
print(df_completo.dtypes)

# Imprimir el DataFrame combinado
print("DataFrame combinado:")
print(df_completo)

# Ver las primeras filas del DataFrame
print(df_completo.head())

# Ver la información del DataFrame
print(df_completo.info())

# Ver estadísticas descriptivas
print(df_completo.describe())


# Lista de columnas sin las que contienen "id"
columnas = ['efectivo_equivalentes', 'deudores_comerciales', 'inventarios', 'activos_biologicos',
            'otros_activos_corrientes', 'activos_no_corrientes_venta', 'deudores_comerciales_no_corriente',
            'inversiones_asociadas_negocios', 'propiedades_planta_equipo', 'activos_por_derechos',
            'propiedades_inversion', 'plusvalia', 'otros_activos_intangibles', 'activo_por_impuesto_diferido',
            'otros_activos_no_corrientes', 'obligaciones_financieras_corrientes', 'pasivos_por_derecho_corriente',
            'proveedores_cuentas_pagar_corriente', 'impuestos_por_pagar_corriente', 'pasivo_beneficios_empleados_corriente',
            'provisiones_corriente', 'obligaciones_financieras_no_corriente', 'pasivos_por_derecho_no_corriente',
            'proveedores_cuentas_pagar_no_corriente', 'pasivo_beneficios_empleados_no_corriente',
            'pasivo_por_impuesto_diferido', 'provisiones_no_corriente', 'pib', 'inflacion', 'tasa_interes',
            'tasa_desempleo', 'capital_emitido', 'prima_emision_capital', 'reservas_utilidades_acumuladas',
            'otro_resultado_integral', 'utilidad_periodo', 'patrimonio_atributable_controladoras',
            'participaciones_no_controladoras', 'costos_venta', 'utilidad_bruta', 'gastos_administracion',
            'gastos_venta', 'gastos_produccion', 'diferencia_cambio_activos_pasivos', 'otros_ingresos_netos',
            'utilidad_operativa', 'ingresos_financieros', 'gastos_financieros', 'dividendos',
            'diferencia_cambio_activos_pasivos_no_operativos', 'participacion_asociadas_negocios',
            'utilidad_antes_impuestos', 'impuesto_renta_corriente', 'impuesto_renta_diferido',
            'utilidad_periodo_operaciones_continuadas', 'operaciones_discontinuadas', 'utilidad_neta_periodo',
            'razon_corriente', 'kwn', 'prueba_acida', 'rotacion_ctas_x_cobrar', 'rotacion_inventarios',
            'ciclo_operacion', 'rotacion_proveedores', 'ciclo_caja', 'rotacion_activos',
            'rentabilidad_operativa', 'roi', 'rentabilidad_patrimonio', 'margen_utilidad_bruta',
            'margen_utilidad_operacional', 'margen_utilidad_antes_impuestos', 'margen_utilidad_neta',
            'nivel_endeudamiento', 'nivel_endeudamiento_corto_plazo', 'nivel_endeudamiento_largo_plazo',
            'nivel_apalancamiento', 'cobertura_intereses', 'cobertura_deuda', 'costo_pasivo_total',
            'costo_deuda_financiera', 'costo_patrimonio']


# Convertir columnas a numéricas
for col in columnas:
    df_completo[col] = pd.to_numeric(df_completo[col], errors='coerce')

# Verificar si hay NaN
nan_counts = df_completo[columnas].isnull().sum()


# Comprobar valores únicos en las columnas que generaron NaN
for col in columnas:
    if nan_counts[col] > 0:
        print(f"Valores únicos en {col} (conversión fallida):")
        print(df[col].unique())

#Empezamos el MODELO ARIMA


# creamos el el modelo
df_completo.set_index('fecha', inplace=True)  # Aseguramos que la columna de fecha sea el índice
df_completo = df_completo.asfreq('MS')  # Establecemos la frecuencia de los datos de forma mensual

# Desplazamiento de ceros: Añadimos un pequeño valor constante (por ejemplo, 0.01) para evitar problemas con log(0)
shift_value = 0.01  # Este valor puede ser ajustado según convenga
df_completo[columnas] = df_completo[columnas] + shift_value


# Definir la función para verificar estacionaridad usando KPSS
def verificar_estacionaridad_kpss(serie):
    resultado = kpss(serie, regression='c')  # 'c' para la prueba con constante
    print('Estadística KPSS:', resultado[0])
    print('Valor p:', resultado[1])
    print('Número de lags utilizados:', resultado[2])
    print('Valores críticos:')
    for key, value in resultado[3].items():
        print(f'   {key}: {value}')


# Verificar estacionaridad y ejecutar ARIMA para cada columna
for columna in columnas:
    print(f'\nVisualizando y verificando estacionaridad para la columna: {columna}')

    # Visualizar la serie de tiempo
    plt.figure(figsize=(10, 6))
    plt.plot(df_completo.index, df_completo[columna], label=f'Datos Reales ({columna})', color='blue')
    plt.title(f'Serie de Tiempo para {columna}')
    plt.xlabel('Fecha')
    plt.ylabel(columna)
    plt.legend()
    plt.show()

    # Verificar estacionaridad
    verificar_estacionaridad_kpss(df_completo[columna])

    try:
        # Ajustamos el modelo ARIMA para cada columna
        print(f'Ejecutando ARIMA para la columna: {columna}')
        mod = ARIMA(df_completo[columna], order=(3, 1, 4))  # Ajustar el orden según convenga
        res = mod.fit()

        # Mostramos el resumen del modelo
        print(res.summary())

        # Predicción para los próximos 12 periodos
        pred = res.predict(start=len(df_completo), end=len(df_completo) + 11,
                           typ='levels')  # Cambiado a 11 porque queremos 12 predicciones

        # Visualización de las predicciones
        plt.figure(figsize=(10, 6))
        plt.plot(df_completo.index, df_completo[columna], label=f'Datos Reales ({columna})', color='blue')
        plt.plot(pd.date_range(df_completo.index[-1], periods=12, freq='MS'), pred, label='Predicción', color='red')
        plt.title(f'Predicción con ARIMA para {columna}')
        plt.xlabel('Fecha')
        plt.ylabel(columna)
        plt.legend()
        plt.show()

    except Exception as e:
        print(f'Error al procesar la columna {columna}: {e}')