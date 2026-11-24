"""Cargador minimo que se empaqueta en el zip de las Lambdas.

Descarga handler.py y model.json de S3 (s3://<CODE_BUCKET>/code/lambda/) en cada arranque en frio,
asi el codigo se actualiza sin reconstruir el zip.
"""
import os
import sys

import boto3

b = os.environ["CODE_BUCKET"]
s3 = boto3.client("s3")
for f in ("handler.py", "model.json"):
    s3.download_file(b, "code/lambda/" + f, "/tmp/" + f)
sys.path.insert(0, "/tmp")
import handler

api, drift, retrain, promote = handler.api, handler.drift, handler.retrain, handler.promote
