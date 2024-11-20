import pandas as pd
import json
import os
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Función para cargar todos los modelos .pkl en una carpeta
def cargar_modelos(carpeta_lasso, carpeta_arima):
    modelos_lasso = {}  # Diccionario para almacenar los modelos Lasso
    modelos_arima = {}  # Diccionario para almacenar los modelos ARIMA

    # Cargar los modelos de la carpeta de Lasso
    for archivo in os.listdir(carpeta_lasso):
        if archivo.endswith(".pkl"):
            ruta_completa = os.path.join(carpeta_lasso, archivo)
            try:
                modelo = joblib.load(ruta_completa)
                modelos_lasso[archivo] = modelo
                print(f"Modelo Lasso '{archivo}' cargado correctamente.")
            except Exception as e:
                print(f"Error al cargar el modelo Lasso '{archivo}': {e}")

    # Cargar los modelos de la carpeta de ARIMA
    for archivo in os.listdir(carpeta_arima):
        if archivo.endswith(".pkl"):
            ruta_completa = os.path.join(carpeta_arima, archivo)
            try:
                modelo = joblib.load(ruta_completa)
                modelos_arima[archivo] = modelo
                print(f"Modelo ARIMA '{archivo}' cargado correctamente.")
            except Exception as e:
                print(f"Error al cargar el modelo ARIMA '{archivo}': {e}")

    return modelos_lasso, modelos_arima

# Función para crear el df_completo
def crear_df_completo(estado_financiero, datos_macro, patrimonio, movimientos, indicadores):
    # Convertir las fechas a un formato común (YYYY-MM)
    for df in [estado_financiero, datos_macro, patrimonio, movimientos, indicadores]:
        df['fecha'] = pd.to_datetime(df['fecha']).dt.to_period('M').dt.start_time

    # Ordenar DataFrames por fecha
    for df in [estado_financiero, datos_macro, patrimonio, movimientos, indicadores]:
        df.sort_values('fecha', inplace=True)

    # Eliminar columnas duplicadas y renombrar las que causan conflictos
    for df in [estado_financiero, datos_macro, patrimonio, movimientos, indicadores]:
        df.drop(columns=['updated_at', 'created_at'], errors='ignore', inplace=True)

    # Renombrar columnas para evitar conflictos
    estado_financiero.rename(columns={'id_empresa': 'id_empresa_ef'}, inplace=True)
    datos_macro.rename(columns={'id_empresa': 'id_empresa_dm'}, inplace=True)
    patrimonio.rename(columns={'id_empresa': 'id_empresa_p'}, inplace=True)
    movimientos.rename(columns={'id_empresa': 'id_empresa_m'}, inplace=True)
    indicadores.rename(columns={'id_empresa': 'id_empresa_i'}, inplace=True)

    # Realizar merge_asof para combinar DataFrames
    df_completo = pd.merge_asof(estado_financiero, datos_macro, on='fecha', direction='nearest')
    df_completo = pd.merge_asof(df_completo, patrimonio, on='fecha', direction='nearest')
    df_completo = pd.merge_asof(df_completo, movimientos, on='fecha', direction='nearest')
    df_completo = pd.merge_asof(df_completo, indicadores, on='fecha', direction='nearest')


    # Eliminar columnas con solo valores nulos o no necesarias
    columnas_no_necesarias = ['id_empresa_ef', 'id_empresa_dm', 'id_empresa_p', 'id_empresa_m', 'id_empresa_i',
                              'id_estado', 'id_macro', 'id_patrimonio', 'id_iug', 'id_indicador','activos_biologicos']
    df_completo = df_completo.drop(columns=columnas_no_necesarias, errors='ignore')

    # Filtrar las columnas de tipo datetime (si las hay)
    date_columns = df_completo.select_dtypes(include=['datetime']).columns
    df_completo = df_completo.drop(columns=date_columns)
    # Asegurarse de que el número de columnas coincida con lo esperado por el modelo

    print(f"Columnas en df_completo: {df_completo.columns.tolist()}")
    print(f"Total de columnas: {len(df_completo.columns)}")


    columnas_esperadas = 81  # Ajusta según las necesidades del modelo
    if df_completo.shape[1] > columnas_esperadas:
        columnas_a_eliminar = df_completo.shape[1] - columnas_esperadas
        df_completo = df_completo.iloc[:, :-columnas_a_eliminar]  # Elimina las columnas extras

    # Verificar el número de características después del ajuste
    print(f"Total de características en df_completo después de ajustar: {df_completo.shape[1]}")





    return df_completo
# Función para ajustar las columnas del df_completo según los modelos Lasso y ARIMA
# Función para ajustar las columnas del df_completo según los modelos Lasso y ARIMA
def ajustar_columnas(df_completo, modelos_lasso, modelos_arima):
    diccionario_columnas_lasso = {
        'lasso_activos_biologicos.pkl': 'activos_biologicos',
        'lasso_lasso_activos_no_corrientes_venta.pkl': 'activos_no_corrientes_venta',
        'lasso_activos_por_derechos.pkl': 'activos_por_derechos',
        'lasso_activo_por_impuesto_diferido.pkl': 'activo_por_impuesto_diferido',
        'lasso_capital_emitido.pkl': 'capital_emitido',
        'lasso_ciclo_caja.pkl': 'ciclo_caja',
        'lasso_ciclo_operacion.pkl': 'ciclo_operacion',
        'lasso_cobertura_deuda.pkl': 'cobertura_deuda',
        'lasso_cobertura_intereses.pkl': 'cobertura_intereses',
        'lasso_costos_venta.pkl': 'costos_venta',
        'lasso_costo_deuda_financiera.pkl': 'costo_deuda_financiera',
        'lasso_costo_pasivo_total.pkl': 'costo_pasivo_total',
        'lasso_costo_patrimonio.pkl': 'costo_patrimonio',
        'lasso_deudores_comerciales.pkl': 'deudores_comerciales',
        'lasso_deudores_comerciales_no_corriente.pkl': 'deudores_comerciales_no_corriente',
        'lasso_diferencia_cambio_activos_pasivos.pkl': 'diferencia_cambio_activos_pasivos',
        'lasso_diferencia_cambio_activos_pasivos_no_operativos.pkl': 'diferencia_cambio_activos_pasivos_no_operativos',
        'lasso_dividendos.pkl': 'dividendos',
        'lasso_efectivo_equivalentes.pkl': 'efectivo_equivalentes',
        'lasso_gastos_administracion.pkl': 'gastos_administracion',
        'lasso_gastos_financieros.pkl': 'gastos_financieros',
        'lasso_gastos_produccion.pkl': 'gastos_produccion',
        'lasso_gastos_venta.pkl': 'gastos_venta',
        'lasso_impuestos_por_pagar_corriente.pkl': 'impuestos_por_pagar_corriente',
        'lasso_impuesto_renta_corriente.pkl': 'impuesto_renta_corriente',
        'lasso_impuesto_renta_diferido.pkl': 'impuesto_renta_diferido',
        'lasso_inflacion.pkl': 'inflacion',
        'lasso_ingresos_financieros.pkl': 'ingresos_financieros',
        'lasso_inventarios.pkl': 'inventarios',
        'lasso_inversiones_asociadas_negocios.pkl': 'inversiones_asociadas_negocios',
        'lasso_kwn.pkl': 'kwn',
        'lasso_margen_utilidad_antes_impuestos.pkl': 'margen_utilidad_antes_impuestos',
        'lasso_margen_utilidad_bruta.pkl': 'margen_utilidad_bruta',
        'lasso_margen_utilidad_neta.pkl': 'margen_utilidad_neta',
        'lasso_margen_utilidad_operacional.pkl': 'margen_utilidad_operacional',
        'lasso_nivel_apalancamiento.pkl': 'nivel_apalancamiento',
        'lasso_nivel_endeudamiento.pkl': 'nivel_endeudamiento',
        'lasso_nivel_endeudamiento_corto_plazo.pkl': 'nivel_endeudamiento_corto_plazo',
        'lasso_nivel_endeudamiento_largo_plazo.pkl': 'nivel_endeudamiento_largo_plazo',
        'lasso_obligaciones_financieras_corrientes.pkl': 'obligaciones_financieras_corrientes',
        'lasso_obligaciones_financieras_no_corriente.pkl': 'obligaciones_financieras_no_corriente',
        'lasso_operaciones_discontinuadas.pkl': 'operaciones_discontinuadas',
        'lasso_otros_activos_corrientes.pkl': 'otros_activos_corrientes',
        'lasso_otros_activos_intangibles.pkl': 'otros_activos_intangibles',
        'lasso_otros_activos_no_corrientes.pkl': 'otros_activos_no_corrientes',
        'lasso_otros_ingresos_netos.pkl': 'otros_ingresos_netos',
        'lasso_otro_resultado_integral.pkl': 'otro_resultado_integral',
        'lasso_participaciones_no_controladoras.pkl': 'participaciones_no_controladoras',
        'lasso_participacion_asociadas_negocios.pkl': 'participacion_asociadas_negocios',
        'lasso_pasivos_por_derecho_corriente.pkl': 'pasivos_por_derecho_corriente',
        'lasso_pasivos_por_derecho_no_corriente.pkl': 'pasivos_por_derecho_no_corriente',
        'lasso_pasivo_beneficios_empleados_corriente.pkl': 'pasivo_beneficios_empleados_corriente',
        'lasso_pasivo_beneficios_empleados_no_corriente.pkl': 'pasivo_beneficios_empleados_no_corriente',
        'lasso_pasivo_por_impuesto_diferido.pkl': 'pasivo_por_impuesto_diferido',
        'lasso_patrimonio_atributable_controladoras.pkl': 'patrimonio_atributable_controladoras',
        'lasso_pib.pkl': 'pib',
        'lasso_plusvalia.pkl': 'plusvalia',
        'lasso_prima_emision_capital.pkl': 'prima_emision_capital',
        'lasso_propiedades_inversion.pkl': 'propiedades_inversion',
        'lasso_propiedades_planta_equipo.pkl': 'propiedades_planta_equipo',
        'lasso_proveedores_cuentas_pagar_corriente.pkl': 'proveedores_cuentas_pagar_corriente',
        'lasso_proveedores_cuentas_pagar_no_corriente.pkl': 'proveedores_cuentas_pagar_no_corriente',
        'lasso_provisiones_corriente.pkl': 'provisiones_corriente',
        'lasso_provisiones_no_corriente.pkl': 'provisiones_no_corriente',
        'lasso_prueba_acida.pkl': 'prueba_acida',
        'lasso_razon_corriente.pkl': 'razon_corriente',
        'lasso_rentabilidad_operativa.pkl': 'rentabilidad_operativa',
        'lasso_rentabilidad_patrimonio.pkl': 'rentabilidad_patrimonio',
        'lasso_reservas_utilidades_acumuladas.pkl': 'reservas_utilidades_acumuladas',
        'lasso_roi.pkl': 'roi',
        'lasso_rotacion_activos.pkl': 'rotacion_activos',
        'lasso_rotacion_ctas_x_cobrar.pkl': 'rotacion_ctas_x_cobrar',
        'lasso_rotacion_inventarios.pkl': 'rotacion_inventarios',
        'lasso_rotacion_proveedores.pkl': 'rotacion_proveedores',
        'lasso_tasa_desempleo.pkl': 'tasa_desempleo',
        'lasso_tasa_interes.pkl': 'tasa_interes',
        'lasso_utilidad_antes_impuestos.pkl': 'utilidad_antes_impuestos',
        'lasso_utilidad_bruta.pkl': 'utilidad_bruta',
        'lasso_utilidad_neta_periodo.pkl': 'utilidad_neta_periodo',
        'lasso_utilidad_operativa.pkl': 'utilidad_operativa',
        'lasso_utilidad_periodo.pkl': 'utilidad_periodo',
        'lasso_utilidad_periodo_operaciones_continuadas.pkl': 'utilidad_periodo_operaciones_continuadas',
    }

    # Diccionario con los nombres de los modelos como claves y las variables como valores
    diccionario_columnas_arima = {
        'arima_efectivo_equivalentes.pkl': 'efectivo_equivalentes',
        'arima_deudores_comerciales.pkl': 'deudores_comerciales',
        'arima_inventarios.pkl': 'inventarios',
        'arima_activos_biologicos.pkl': 'activos_biologicos',
        'arima_otros_activos_corrientes.pkl': 'otros_activos_corrientes',
        'arima_activos_no_corrientes_venta.pkl': 'activos_no_corrientes_venta',
        'arima_deudores_comerciales_no_corriente.pkl': 'deudores_comerciales_no_corriente',
        'arima_inversiones_asociadas_negocios.pkl': 'inversiones_asociadas_negocios',
        'arima_propiedades_planta_equipo.pkl': 'propiedades_planta_equipo',
        'arima_activos_por_derechos.pkl': 'activos_por_derechos',
        'arima_propiedades_inversion.pkl': 'propiedades_inversion',
        'arima_plusvalia.pkl': 'plusvalia',
        'arima_otros_activos_intangibles.pkl': 'otros_activos_intangibles',
        'arima_activo_por_impuesto_diferido.pkl': 'activo_por_impuesto_diferido',
        'arima_otros_activos_no_corrientes.pkl': 'otros_activos_no_corrientes',
        'arima_obligaciones_financieras_corrientes.pkl': 'obligaciones_financieras_corrientes',
        'arima_pasivos_por_derecho_corriente.pkl': 'pasivos_por_derecho_corriente',
        'arima_proveedores_cuentas_pagar_corriente.pkl': 'proveedores_cuentas_pagar_corriente',
        'arima_impuestos_por_pagar_corriente.pkl': 'impuestos_por_pagar_corriente',
        'arima_pasivo_beneficios_empleados_corriente.pkl': 'pasivo_beneficios_empleados_corriente',
        'arima_provisiones_corriente.pkl': 'provisiones_corriente',
        'arima_obligaciones_financieras_no_corriente.pkl': 'obligaciones_financieras_no_corriente',
        'arima_pasivos_por_derecho_no_corriente.pkl': 'pasivos_por_derecho_no_corriente',
        'arima_proveedores_cuentas_pagar_no_corriente.pkl': 'proveedores_cuentas_pagar_no_corriente',
        'arima_pasivo_beneficios_empleados_no_corriente.pkl': 'pasivo_beneficios_empleados_no_corriente',
        'arima_pasivo_por_impuesto_diferido.pkl': 'pasivo_por_impuesto_diferido',
        'arima_provisiones_no_corriente.pkl': 'provisiones_no_corriente',
        'arima_pib.pkl': 'pib',
        'arima_inflacion.pkl': 'inflacion',
        'arima_tasa_interes.pkl': 'tasa_interes',
        'arima_tasa_desempleo.pkl': 'tasa_desempleo',
        'arima_capital_emitido.pkl': 'capital_emitido',
        'arima_prima_emision_capital.pkl': 'prima_emision_capital',
        'arima_reservas_utilidades_acumuladas.pkl': 'reservas_utilidades_acumuladas',
        'arima_otro_resultado_integral.pkl': 'otro_resultado_integral',
        'arima_utilidad_periodo.pkl': 'utilidad_periodo',
        'arima_patrimonio_atributable_controladoras.pkl': 'patrimonio_atributable_controladoras',
        'arima_participaciones_no_controladoras.pkl': 'participaciones_no_controladoras',
        'arima_costos_venta.pkl': 'costos_venta',
        'arima_utilidad_bruta.pkl': 'utilidad_bruta',
        'arima_gastos_administracion.pkl': 'gastos_administracion',
        'arima_gastos_venta.pkl': 'gastos_venta',
        'arima_gastos_produccion.pkl': 'gastos_produccion',
        'arima_diferencia_cambio_activos_pasivos.pkl': 'diferencia_cambio_activos_pasivos',
        'arima_otros_ingresos_netos.pkl': 'otros_ingresos_netos',
        'arima_utilidad_operativa.pkl': 'utilidad_operativa',
        'arima_ingresos_financieros.pkl': 'ingresos_financieros',
        'arima_gastos_financieros.pkl': 'gastos_financieros',
        'arima_dividendos.pkl': 'dividendos',
        'arima_diferencia_cambio_activos_pasivos_no_operativos.pkl': 'diferencia_cambio_activos_pasivos_no_operativos',
        'arima_participacion_asociadas_negocios.pkl': 'participacion_asociadas_negocios',
        'arima_utilidad_antes_impuestos.pkl': 'utilidad_antes_impuestos',
        'arima_impuesto_renta_corriente.pkl': 'impuesto_renta_corriente',
        'arima_impuesto_renta_diferido.pkl': 'impuesto_renta_diferido',
        'arima_utilidad_periodo_operaciones_continuadas.pkl': 'utilidad_periodo_operaciones_continuadas',
        'arima_operaciones_discontinuadas.pkl': 'operaciones_discontinuadas',
        'arima_utilidad_neta_periodo.pkl': 'utilidad_neta_periodo',
        'arima_razon_corriente.pkl': 'razon_corriente',
        'arima_kwn.pkl': 'kwn',
        'arima_prueba_acida.pkl': 'prueba_acida',
        'arima_rotacion_ctas_x_cobrar.pkl': 'rotacion_ctas_x_cobrar',
        'arima_rotacion_inventarios.pkl': 'rotacion_inventarios',
        'arima_ciclo_operacion.pkl': 'ciclo_operacion',
        'arima_rotacion_proveedores.pkl': 'rotacion_proveedores',
        'arima_ciclo_caja.pkl': 'ciclo_caja',
        'arima_rotacion_activos.pkl': 'rotacion_activos',
        'arima_rentabilidad_operativa.pkl': 'rentabilidad_operativa',
        'arima_roi.pkl': 'roi',
        'arima_rentabilidad_patrimonio.pkl': 'rentabilidad_patrimonio',
        'arima_margen_utilidad_bruta.pkl': 'margen_utilidad_bruta',
        'arima_margen_utilidad_operacional.pkl': 'margen_utilidad_operacional',
        'arima_margen_utilidad_antes_impuestos.pkl': 'margen_utilidad_antes_impuestos',
        'arima_margen_utilidad_neta.pkl': 'margen_utilidad_neta',
        'arima_nivel_endeudamiento.pkl': 'nivel_endeudamiento',
        'arima_nivel_endeudamiento_corto_plazo.pkl': 'nivel_endeudamiento_corto_plazo',
        'arima_nivel_endeudamiento_largo_plazo.pkl': 'nivel_endeudamiento_largo_plazo',
        'arima_nivel_apalancamiento.pkl': 'nivel_apalancamiento',
        'arima_cobertura_intereses.pkl': 'cobertura_intereses',
        'arima_cobertura_deuda.pkl': 'cobertura_deuda',
        'arima_costo_pasivo_total.pkl': 'costo_pasivo_total',
        'arima_costo_deuda_financiera.pkl': 'costo_deuda_financiera',
        'arima_costo_patrimonio.pkl': 'costo_patrimonio'
    }

    # Ajustar las columnas para los modelos Lasso
    for modelo, columna in diccionario_columnas_lasso.items():
        if modelo in modelos_lasso:
            df_completo[columna] = modelos_lasso[modelo].predict(df_completo)

    # Ajustar las columnas para los modelos ARIMA
    for modelo, columna in diccionario_columnas_arima.items():
        if modelo in modelos_arima:
            df_completo[columna] = modelos_arima[modelo].forecast(len(df_completo))




    return df_completo
# Función de predicción usando Lasso, ARIMA y RNN
def predecir_con_modelos(df, modelos_lasso, modelos_arima, rnn_model):
    # Lasso: Seleccionamos las características de entrada para Lasso
    df_lasso = df[[modelo.split('.')[0] for modelo in modelos_lasso.keys()]]
    predicciones_lasso = np.array([modelo.predict(df_lasso) for modelo in modelos_lasso.values()])

    # ARIMA: Modelamos las series temporales y calculamos los residuos
    residuos_arima = []
    for column in df_lasso.columns:
        model = ARIMA(df[column], order=(1, 1, 1))  # Ajusta el orden de ARIMA según sea necesario
        model_fit = model.fit()
        residuos_arima.append(model_fit.resid)

    # Convertimos residuos en un arreglo numpy
    residuos_arima = np.array(residuos_arima).T  # Transponemos para que las dimensiones sean correctas

    # Selección de variables por ARIMA
    selected_features_arima = df_lasso.columns  # Suponemos que ARIMA selecciona todas las características de entrada
    df_arima = df[selected_features_arima]

    # RNN: Predicciones usando los residuos de ARIMA y las características seleccionadas por ARIMA
    predicciones_rnn = rnn_model.predict(residuos_arima)  # Usamos los residuos como entrada a la RNN

    # Imprimir las predicciones de la RNN
    print("\nPredicciones de la RNN (Red Neuronal Recurrente):")
    for idx, pred in enumerate(predicciones_rnn):
        print(f"Predicción RNN para el conjunto {idx}: {pred[:5]}")  # Muestra las primeras 5 predicciones

    return predicciones_rnn

def main():
    try:
        #cargar los modelos
        input_file = r"C:\xampp\htdocs\example-app\storage\app\temp_input_data.json"
        path_modelos_lasso = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\modelos_lasso"  # Ruta del modelo Lasso entrenado
        path_modelos_arima = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\modelos_arima"  # Ruta del modelo ARIMA entrenado
        path_modelo_rnn = r"C:\Users\USUARIO\Desktop\pyhton\proyecto_finance\rnn_model.h5"  # Ruta del modelo RNN entrenado

        modelos_lasso, modelos_arima = cargar_modelos(path_modelos_lasso, path_modelos_arima)
        rnn_model = load_model(path_modelo_rnn)  # Cargar modelo RNN

        # Cargar archivo de datos
        with open(input_file, 'r') as f:
            input_data = json.load(f)


        estado_financiero = pd.DataFrame(input_data['estadoFinanciero'])
        datos_macro = pd.DataFrame(input_data['datosMacro'])
        patrimonio = pd.DataFrame(input_data['patrimonio'])
        movimientos = pd.DataFrame(input_data['movimientos'])
        indicadores = pd.DataFrame(input_data['indicadores'])

        # Crear el DataFrame completo
        df_completo = crear_df_completo(estado_financiero, datos_macro, patrimonio, movimientos, indicadores)

        # Ajustar las columnas según los modelos
        df_completo = ajustar_columnas(df_completo, modelos_lasso, modelos_arima)

        # Predicciones
        predicciones_rnn = predecir_con_modelos(df_completo, modelos_lasso, modelos_arima, rnn_model)

        print("Predicciones realizadas correctamente.")

    except Exception as e:
        print(f"Error en el proceso: {e}")

if __name__ == "__main__":
    main()
