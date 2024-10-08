#importamos las librerias
import pandas as pd
import mysql.connector
from mysql.connector import Error

import mysql.connector
from mysql.connector import Error

# Datos de conexión (tomados de tu archivo .env de Laravel)
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'database': 'finance_model',
    'user': 'root',
    'password': 'Teledatos2024*'
}

try:
    # Crear la conexión a la base de datos
    connection = mysql.connector.connect(**db_config)

    if connection.is_connected():
        print("Conexión exitosa a la base de datos")

        # Crear un cursor para ejecutar consultas
        cursor = connection.cursor()

        # Consulta SQL para obtener los datos de la tabla estado_financiero
        query = "SELECT * FROM estado_financiero"
        cursor.execute(query)

        # Recuperar los resultados
        rows = cursor.fetchall()

        # Imprimir los resultados
        for row in rows:
            print(row)

except Error as e:
    print(f"Error al conectarse a la base de datos: {e}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("Conexión cerrada")
