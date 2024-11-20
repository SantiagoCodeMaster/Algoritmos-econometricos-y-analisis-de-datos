from flask import Flask, request, jsonify

app = Flask(__name__)


# Ruta para recibir los datos de Laravel
@app.route('/api/algoritmo', methods=['POST'])
def ejecutar_algoritmo():
    # Recibir los datos enviados por Laravel
    datos = request.json

    # Imprimir los datos recibidos en la consola
    print("Datos recibidos desde Laravel:")
    print(f"Email: {datos.get('email')}")
    print(f"Password: {datos.get('password')}")

    # Aquí obtienes los datos de la empresa o los resultados que necesitas procesar.
    # Como ejemplo, puedes agregar una lógica para el algoritmo
    # Enviar una respuesta de ejemplo
    return jsonify({
        "empresa": 123,
        "rentabilidad": 1000,
        "empleados": 50
    })


if __name__ == '__main__':
    # Inicia el servidor Flask en el puerto 5000 (por defecto)
    app.run(debug=True, port=5000)
