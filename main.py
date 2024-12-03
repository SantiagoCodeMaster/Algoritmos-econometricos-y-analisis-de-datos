from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/prueba/algo', methods=['POST'])
def prueba_algo():
    # Obtener los datos JSON del cuerpo de la solicitud
    data = request.get_json()

    # Verificar si los datos están presentes
    if not data:
        print({"error": "No data provided"}), 400

    # Verificar si todos los valores son números y sumarlos por 2
    if all(isinstance(x, (int, float)) for x in data):
        # Sumar 2 a cada número en la lista
        data = [x + 2 for x in data]
        return jsonify({"result": data}), 200
    else:
        print({"error": "All values must be numbers"}), 400

if __name__ == '__main__':
    app.run(debug=True)
