#importamos librerias
import conexion_proyect
import pandas as pd
import tensorflow as tf
import numpy as np

data_estadofinaciero= conexion_proyect.api_estado_financiero
estado_financiero = data_estadofinaciero.get()
data_macro = conexion_proyect.api_datos_macro
datos_macro = data_macro.get()
data_patrimonio = conexion_proyect.api_patrimonio
patrimonio = data_patrimonio.get()
data_movimientos = conexion_proyect.api_movimientos
movimientos = data_movimientos.get()
data_indicadores = conexion_proyect.api_indicadores
indicadores = data_indicadores.get()



