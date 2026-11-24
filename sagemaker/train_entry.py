"""Punto de entrada para SageMaker (script mode, contenedor scikit-learn).

SageMaker monta los datos en /opt/ml/input/data/train y espera el modelo en /opt/ml/model.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acuicultura.train import run

if __name__ == "__main__":
    data = os.path.join(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"), "hourly_ponds.csv")
    run(data, os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
