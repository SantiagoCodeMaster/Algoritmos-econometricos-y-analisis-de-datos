#importamos las librerias
import requests
#creamos la clase para la hacer la conexion la API
class Api:
    def __init__(self,base_url):
      self.base_url = base_url

    #creamos el metodo get para pasar los datos a un metodo json
    def get(self):
        try:
             # Itera sobre cada URL y realiza la solicitud
             response = requests.get(self.base_url)
             # Verifica que la solicitud fue exitosa
             if response.status_code == 200:
                  #obtenemos los datos enviados por get a formato json
                  datos = response.json()
                  return datos
             else:
                print(f'Error en {self.base_url}: {response.status_code}')

        except requests.exceptions.RequestException as e:
                 print(f'Error al realizar la solicitud: {e}')

# Crear instancias de la clase Api fuera del bloque if __name__
api_estado_financiero = Api('http://127.0.0.1:8000/estado-financiero')
api_datos_macro = Api('http://127.0.0.1:8000/data_m')
api_patrimonio = Api('http://127.0.0.1:8000/patrimonio')
api_movimientos = Api('http://127.0.0.1:8000/movimientos')
api_indicadores = Api('http://127.0.0.1:8000/indicadores')

if __name__ == '__main__':
    estado_financiero = api_estado_financiero.get()
    datos_macro = api_datos_macro.get()
    patrimonio = api_patrimonio.get()
    movimientos = api_movimientos.get()
    indicadores = api_indicadores.get()









