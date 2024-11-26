import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore", message="X has feature names, but Lasso was fitted without feature names")

# Rutas de los modelos entrenados
path_modelos_lasso = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\modelos_lasso"
path_modelos_arima = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\modelos_arima"
path_modelo_rnn = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\rnn_model.h5"

# Lista base de características esperadas por los modelos Lasso
columnas_esperadas_base = [
    'activos_biologicos', 'activos_no_corrientes_venta', 'activos_por_derechos',
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
    'utilidad_neta_periodo', 'utilidad_operativa', 'utilidad_periodo', 'utilidad_periodo_operaciones_continuadas'
]

app = Flask(__name__)

@app.route('/api/algoritmo', methods=['POST'])
def ejecutar_algoritmo():
    try:
        # Obtener los datos desde Laravel
        data = request.json
        estado_financiero = data.get('estado_financiero', [])
        datos_macro = data.get('datos_macro', [])
        patrimonio = data.get('patrimonio', [])
        movimientos = data.get('movimientos', [])
        indicadores = data.get('indicadores', [])
        # Convertir los datos a DataFrames
        estado_financiero_df = pd.DataFrame(estado_financiero)
        datos_macro_df = pd.DataFrame(datos_macro)
        patrimonio_df = pd.DataFrame(patrimonio)
        movimientos_df = pd.DataFrame(movimientos)
        indicadores_df = pd.DataFrame(indicadores)

        # Procesar las fechas
        for df in [estado_financiero_df, datos_macro_df, patrimonio_df, movimientos_df, indicadores_df]:
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha']).dt.to_period('M').dt.start_time

        # Limpiar columnas innecesarias antes de las combinaciones
        for df in [estado_financiero_df, datos_macro_df, patrimonio_df, movimientos_df, indicadores_df]:
            df.drop(columns=['updated_at', 'created_at', 'id_empresa'], errors='ignore', inplace=True)

        # Ordenar los DataFrames por 'fecha'
        estado_financiero_df.sort_values('fecha', inplace=True)
        datos_macro_df.sort_values('fecha', inplace=True)
        patrimonio_df.sort_values('fecha', inplace=True)
        movimientos_df.sort_values('fecha', inplace=True)
        indicadores_df.sort_values('fecha', inplace=True)

        # Combinar los DataFrames con sufijos personalizados
        df_completo = pd.merge_asof(estado_financiero_df, datos_macro_df, on='fecha', direction='nearest', suffixes=('', '_datos_macro'))
        df_completo = pd.merge_asof(df_completo, patrimonio_df, on='fecha', direction='nearest', suffixes=('', '_patrimonio'))
        df_completo = pd.merge_asof(df_completo, movimientos_df, on='fecha', direction='nearest', suffixes=('', '_movimientos'))
        df_completo = pd.merge_asof(df_completo, indicadores_df, on='fecha', direction='nearest', suffixes=('', '_indicadores'))

        # Normalizar los datos según las columnas esperadas
        df_completo = df_completo.reindex(columns=columnas_esperadas_base, fill_value=0)

        # Ajustamos las opciones de visualización
        pd.set_option('display.max_rows', None)  # Muestra todas las filas
        pd.set_option('display.max_columns', None)  # Muestra todas las columnas
        pd.set_option('display.width', None)  # Ajusta el ancho de la salida
        pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de cada columna
        print(df_completo)
        variables_seleccionadas = []
        # Lasso - Selección de Características
        for columna in df_completo.columns:
            modelo_lasso_path = os.path.join(path_modelos_lasso, f"lasso_{columna}.pkl")
            if os.path.exists(modelo_lasso_path):
                modelo_lasso = joblib.load(modelo_lasso_path)
                df_actual = df_completo.copy()

                # Reindexar según las columnas esperadas
                df_actual = df_actual[columnas_esperadas_base]

                # Si el número de columnas excede el esperado, eliminar columnas sobrantes automáticamente
                if df_actual.shape[1] > 81:
                    columnas_eliminadas = df_actual.columns[81:]  # Seleccionar columnas sobrantes
                    df_actual.drop(columns=columnas_eliminadas, inplace=True)  # Eliminar las columnas sobrantes
                    print(f"Se eliminaron columnas sobrantes: {columnas_eliminadas.tolist()}")

                try:
                    # Calcular la importancia utilizando el modelo Lasso
                    importancia = modelo_lasso.predict(df_actual)

                    # Si la importancia es mayor a 0, añadir la columna a la lista de seleccionadas
                    if importancia[0] > 0:
                        variables_seleccionadas.append(columna)

                except Exception as e:
                    print(f"Hubo un error al predecir con el modelo Lasso para la columna {columna}: {e}")

        # Filtrar las columnas seleccionadas
        df_lasso = df_completo[variables_seleccionadas]

        # Convertir a tipo numérico y eliminar filas con NaN
        df_lasso = df_lasso.apply(pd.to_numeric, errors='coerce').dropna()

        residuos_arima = []

        for columna in df_lasso.columns:
            print(f"Columna: {columna}")
            print(f"Número de observaciones: {df_lasso[columna].dropna().shape[0]}")
            print(f"Valores únicos: {df_lasso[columna].nunique()}")
            print(f"Primeros valores:\n{df_lasso[columna].head()}\n")

            # Conversión de datos a numéricos
            df_lasso[columna] = pd.to_numeric(df_lasso[columna], errors='coerce')

            # En caso de que la columna tenga NaNs, los reemplazamos con ceros
            if df_lasso[columna].isnull().any():
                print(f"La columna '{columna}' contiene valores NaN. Reemplazando por ceros.")
                df_lasso[columna] = df_lasso[columna].fillna(0)

            # Si la columna está vacía después de eliminar los NaN, la omitimos
            if df_lasso[columna].dropna().empty:
                print(f"La columna '{columna}' está vacía después de eliminar los NaN y será omitida.")
                continue

            modelo_arima_path = os.path.join(path_modelos_arima, f"arima_{columna}.pkl")

            if os.path.exists(modelo_arima_path):
                modelo_arima = joblib.load(modelo_arima_path)

                try:
                    # Si la columna tiene poca variación o valores extremos, aplicar una transformación logarítmica
                    # para asegurar que no haya problemas con la escala
                    if df_lasso[columna].var() == 0:
                        print(f"Varianza de la columna '{columna}' es 0, aplicando transformación logarítmica.")
                        df_lasso[columna] = np.log1p(df_lasso[columna])


                    # Ajustar el modelo ARIMA sin preocuparse por la estacionariedad (sin pruebas previas)
                    arima_model = ARIMA(df_lasso[columna], order=(5, 1, 0))  # Ajustar el orden según sea necesario
                    arima_fitted = arima_model.fit()

                    # Guardar los residuos de ARIMA para cada columna
                    residuos_arima.append(arima_fitted.resid)


                except Exception as e:
                    print(f"Hubo un error al ajustar ARIMA para la columna '{columna}': {e}")


        # Convertir residuos a un arreglo de Numpy (transponer para que tenga la forma correcta)
        residuos_arima = np.array(residuos_arima).T  # Transponer para alinear los residuos con el df_lasso

        # Concatenar los residuos con df_lasso (asegurándonos de que tengan las mismas dimensiones)
        df_combinado = np.concatenate([residuos_arima, df_lasso.values], axis=1)
        # Preparar los datos para la RNN
        # Aquí estamos utilizando df_combinado como entrada tanto para las características como para el objetivo
        # El generador de series de tiempo puede tomar estos datos para crear secuencias de longitud 10
        generador_rnn = TimeseriesGenerator(df_combinado, df_combinado, length=10, batch_size=32)

        # Cargar el modelo de la RNN
        modelo_rnn = load_model(path_modelo_rnn)

        try:
         # Realizar las predicciones con la RNN
         predicciones_rnn = modelo_rnn.predict(generador_rnn)
        except  Exception as e:
            print(f"error grave en : {e}")


        # Retornar las predicciones
        return jsonify({
            "mensaje": "Algoritmo ejecutado correctamente",
            "predicciones": predicciones_rnn.tolist()
        })
    except Exception as e:
        return jsonify({"mensaje": f"Error: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)



