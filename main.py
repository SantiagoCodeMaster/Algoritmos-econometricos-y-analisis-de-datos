import joblib
import pandas as pd
import json
import os
from statsmodels.tsa.arima.model import ARIMA
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import sys


def cargar_modelos(path_lasso, path_arima):
    lasso_modelos = {}
    arima_modelos = {}
    try:
        for archivo in os.listdir(path_lasso):
            if archivo.endswith(".pkl"):
                variable = archivo.replace(".pkl", "")
                lasso_modelos[variable] = joblib.load(os.path.join(path_lasso, archivo))

        for archivo in os.listdir(path_arima):
            if archivo.endswith(".pkl"):
                variable = archivo.replace(".pkl", "")
                with open(os.path.join(path_arima, archivo), "rb") as f:
                    arima_modelos[variable] = joblib.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Error al cargar modelos: {e}"}))
    return lasso_modelos, arima_modelos


def predecir_con_modelos(df, lasso_modelos, arima_modelos, rnn_model):
    predicciones_lasso = {}
    residuos_arima = {}
    predicciones_rnn = {}

    for variable in df.columns:
        try:
            if variable in lasso_modelos:
                predicciones_lasso[variable] = lasso_modelos[variable].predict(
                    df[variable].values.reshape(-1, 1)).tolist()

            if variable in arima_modelos:
                arima_modelo = arima_modelos[variable]
                arima_modelo_fit = arima_modelo.fit(df[variable])
                residuos_arima[variable] = arima_modelo_fit.resid.tolist()

                scaler = MinMaxScaler(feature_range=(0, 1))
                residuos_arima_scaled = scaler.fit_transform(
                    np.array(residuos_arima[variable]).reshape(-1, 1)
                )

                predicciones_rnn[variable] = rnn_model.predict(residuos_arima_scaled).flatten().tolist()
        except Exception as e:
            print(json.dumps({"error": f"Error en predicción de {variable}: {e}"}))

    return predicciones_lasso, residuos_arima, predicciones_rnn


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Archivo de entrada no especificado"}))
        return

    path_modelos_lasso = sys.argv[2] if len(sys.argv) > 2 else "modelos_lasso"
    path_modelos_arima = sys.argv[3] if len(sys.argv) > 3 else "modelos_arima"
    path_modelo_rnn = "rnn_model.h5"
    input_file = sys.argv[1]

    try:
        lasso_modelos, arima_modelos = cargar_modelos(path_modelos_lasso, path_modelos_arima)
        rnn_model = load_model(path_modelo_rnn)

        with open(input_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        predicciones_lasso, residuos_arima, predicciones_rnn = predecir_con_modelos(df, lasso_modelos, arima_modelos,
                                                                                    rnn_model)
        resultados = {
            "predicciones_lasso": predicciones_lasso,
            "residuos_arima": residuos_arima,
            "predicciones_rnn": predicciones_rnn
        }
        print(json.dumps(resultados))

    except Exception as e:
        print(json.dumps({"error": f"Error en el procesamiento de datos: {e}"}))


if __name__ == "__main__":
    main()
