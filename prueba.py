# Importamos librerías
import conexion_proyect
import pandas as pd

class ProcesadorDatos:
    def __init__(self):
        self.df_completo = None  # Para almacenar el DataFrame combinado

    def obtener_datos(self):
        # Obtener los datos de las diferentes API
        self.estado_financiero = conexion_proyect.api_estado_financiero.get()
        self.datos_macro = conexion_proyect.api_datos_macro.get()
        self.patrimonio = conexion_proyect.api_patrimonio.get()
        self.movimientos = conexion_proyect.api_movimientos.get()
        self.indicadores = conexion_proyect.api_indicadores.get()

    def convertir_a_dataframe(self):
        # Convertir los datos a DataFrames si no lo son ya
        self.estado_financiero_df = pd.DataFrame(self.estado_financiero)
        self.datos_macro_df = pd.DataFrame(self.datos_macro)
        self.patrimonio_df = pd.DataFrame(self.patrimonio)
        self.movimientos_df = pd.DataFrame(self.movimientos)
        self.indicadores_df = pd.DataFrame(self.indicadores)

    def procesar_fechas(self):
        # Convertir las fechas a un formato común (YYYY-MM) y ordenar los DataFrames
        for df in [self.estado_financiero_df, self.datos_macro_df, self.patrimonio_df, self.movimientos_df, self.indicadores_df]:
            df['fecha'] = pd.to_datetime(df['fecha']).dt.to_period('M').dt.start_time
            df.sort_values('fecha', inplace=True)

    def limpiar_columnas(self):
        # Eliminar columnas duplicadas y renombrar las que causan conflictos
        for df in [self.estado_financiero_df, self.datos_macro_df, self.patrimonio_df, self.movimientos_df, self.indicadores_df]:
            df.drop(columns=['updated_at', 'created_at'], errors='ignore', inplace=True)

        # Renombrar columnas para evitar conflictos en los merges
        self.estado_financiero_df.rename(columns={'id_empresa': 'id_empresa_ef'}, inplace=True)
        self.datos_macro_df.rename(columns={'id_empresa': 'id_empresa_dm'}, inplace=True)
        self.patrimonio_df.rename(columns={'id_empresa': 'id_empresa_p'}, inplace=True)
        self.movimientos_df.rename(columns={'id_empresa': 'id_empresa_m'}, inplace=True)
        self.indicadores_df.rename(columns={'id_empresa': 'id_empresa_i'}, inplace=True)

    def combinar_dataframes(self):
        # Realizar merge_asof para combinar DataFrames
        self.df_completo = pd.merge_asof(self.estado_financiero_df, self.datos_macro_df, on='fecha', direction='nearest')
        self.df_completo = pd.merge_asof(self.df_completo, self.patrimonio_df, on='fecha', direction='nearest')
        self.df_completo = pd.merge_asof(self.df_completo, self.movimientos_df, on='fecha', direction='nearest')
        self.df_completo = pd.merge_asof(self.df_completo, self.indicadores_df, on='fecha', direction='nearest')

    def mostrar_dataframe(self):
        # Ajustar las opciones de visualización y mostrar el DataFrame combinado
        pd.set_option('display.max_rows', None)  # Muestra todas las filas
        pd.set_option('display.max_columns', None)  # Muestra todas las columnas
        pd.set_option('display.width', None)  # Ajusta el ancho de la salida
        pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de cada columna

        print("DataFrame combinado:")
        print(self.df_completo)


# Uso de la clase
if __name__ == "__main__":
    procesador = ProcesadorDatos()  # Instanciamos la clase
    procesador.obtener_datos()  # Obtenemos los datos de las APIs
    procesador.convertir_a_dataframe()  # Convertimos los datos a DataFrame
    procesador.procesar_fechas()  # Procesamos las fechas
    procesador.limpiar_columnas()  # Limpiamos y renombramos las columnas
    procesador.combinar_dataframes()  # Combinamos los DataFrames
    procesador.mostrar_dataframe()  # Mostramos el DataFrame final

