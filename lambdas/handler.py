"""Funciones Lambda del proyecto (Python puro + boto3, sin numpy/sklearn).

api      -> POST /predict, GET /health, GET /model-info (Function URL)
drift    -> calcula PSI sobre las ultimas predicciones guardadas en S3 y lo publica en CloudWatch
retrain  -> lanza un job de entrenamiento en SageMaker (disparado por EventBridge)
promote  -> copia el modelo de un job terminado a s3://<bucket>/models/<job>/
"""
import base64
import json
import math
import os
import statistics
import uuid
from datetime import datetime, timedelta, timezone

import boto3

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "model.json"), encoding="utf-8") as f:
    M = json.load(f)

BUCKET = os.environ.get("BUCKET", "mlops-acuicultura-631603563100")
SM_BUCKET = os.environ.get("SM_BUCKET", "sagemaker-us-east-1-631603563100")
ROLE_ARN = os.environ.get("ROLE_ARN", "arn:aws:iam::631603563100:role/sageMakerEjecutor")
IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
NS = "Acuicultura/MLOps"
MIN_READINGS = 7
RANGES = {"temperature_c": (10, 40), "ph": (4, 11), "dissolved_oxygen_mgl": (0, 20)}
BASE = {"temperature_c": "temp", "ph": "ph", "dissolved_oxygen_mgl": "do"}


def _resp(code, obj):
    return {"statusCode": code, "headers": {"content-type": "application/json"}, "body": json.dumps(obj)}


def features(readings):
    """Mismas features que src/acuicultura/features.py, calculadas para la ultima lectura."""
    ts = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in readings]
    order = sorted(range(len(readings)), key=lambda i: ts[i])
    rs = [readings[i] for i in order]
    hour = ts[order[-1]].hour
    f = {"hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24)}
    for col, n in BASE.items():
        v = [float(r[col]) for r in rs]
        last6 = v[-6:]
        f[n] = v[-1]
        f[n + "_mean6"] = sum(last6) / 6
        f[n + "_std6"] = statistics.stdev(last6)
        f[n + "_diff1"] = v[-1] - v[-2]
    return f


def score(f):
    """Gradient boosting exportado a JSON: suma de arboles + sigmoide (mismo resultado que scikit-learn)."""
    z = M["baseline"]
    for tree in M["trees"]:
        i = 0
        while True:
            feat, thr, left, right, val = tree[i]
            if feat < 0:
                z += val
                break
            i = left if f[M["features"][feat]] <= thr else right
    return 1 / (1 + math.exp(-z))


def _validate(body):
    rs = body.get("readings") if isinstance(body, dict) else None
    if not isinstance(rs, list) or len(rs) < MIN_READINGS:
        return f"Se necesitan al menos {MIN_READINGS} lecturas en 'readings'"
    for i, r in enumerate(rs):
        try:
            datetime.fromisoformat(str(r["timestamp"]).replace("Z", "+00:00"))
            for col, (lo, hi) in RANGES.items():
                if not lo <= float(r[col]) <= hi:
                    return f"readings[{i}].{col} fuera de rango [{lo}, {hi}]"
        except (KeyError, TypeError, ValueError):
            return f"readings[{i}] invalida (timestamp, temperature_c, ph, dissolved_oxygen_mgl)"
    return None


def _metrics(items):
    try:
        boto3.client("cloudwatch").put_metric_data(
            Namespace=NS, MetricData=[{"MetricName": k, "Value": float(v)} for k, v in items.items()])
    except Exception as e:  # noqa: BLE001  el monitoreo nunca debe tumbar la prediccion
        print("metric error", e)


def api(event, context=None):
    http = event.get("requestContext", {}).get("http", {})
    method, path = http.get("method", "GET"), event.get("rawPath", "/")
    if path == "/health":
        return _resp(200, {"status": "ok"})
    if path == "/model-info":
        return _resp(200, {"model": M["model"], "do_threshold_mgl": M["do_threshold_mgl"],
                           "horizon_h": M["horizon_h"], "alert_threshold": M["threshold"], "test": M["test"]})
    if path == "/predict" and method == "POST":
        raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8")
        try:
            body = json.loads(raw)
        except ValueError:
            return _resp(422, {"detail": "JSON invalido"})
        err = _validate(body)
        if err:
            return _resp(422, {"detail": err})
        f = features(body["readings"])
        if f["do"] < M["do_threshold_mgl"] + M["early_margin"]:  # ya esta bajo: alerta directa
            p, alert, status = 1.0, True, "oxigeno_ya_bajo"
        else:
            p = score(f)
            alert, status = p >= M["threshold"], "modelo"
        out = {"risk_score": p, "alert": alert, "horizon_h": M["horizon_h"], "do_threshold_mgl": M["do_threshold_mgl"],
               "model_version": M["model"], "status": status}
        now = datetime.now(timezone.utc)
        try:  # registro de la prediccion para el monitoreo de drift
            boto3.client("s3").put_object(
                Bucket=BUCKET, Key=f"predictions/{now:%Y-%m-%d}/{now:%H%M%S}-{uuid.uuid4().hex[:8]}.json",
                Body=json.dumps({"ts": now.isoformat(), "features": f, **out}).encode())
        except Exception as e:  # noqa: BLE001
            print("log error", e)
        _metrics({"risk_score": p, "predicciones": 1})
        return _resp(200, out)
    return _resp(404, {"detail": "no existe"})


def _psi(actual_vals, deciles, props=None):
    edges = deciles[1:-1]
    counts = [0] * (len(edges) + 1)
    for v in actual_vals:
        counts[sum(v > e for e in edges)] += 1
    n = len(actual_vals)
    props = props or [1 / len(counts)] * len(counts)
    total = 0.0
    for c, e in zip(counts, props):
        e = max(e, 1e-6)
        a = max(c / n, 1e-6)
        total += (a - e) * math.log(a / e)
    return total


def drift(event=None, context=None, min_n=100, max_n=200):
    s3 = boto3.client("s3")
    keys = []
    now = datetime.now(timezone.utc)
    for day in sorted({now.strftime("%Y-%m-%d"), (now - timedelta(days=1)).strftime("%Y-%m-%d")}):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"predictions/{day}/"):
            keys += [(o["LastModified"], o["Key"]) for o in page.get("Contents", [])]
    keys = [k for _, k in sorted(keys)[-max_n:]]
    if len(keys) < min_n:
        print("pocas predicciones", len(keys))
        return {"n": len(keys), "skipped": True}
    rows = [json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())["features"] for k in keys]
    psis = {c: _psi([r[c] for r in rows], M["ref_deciles"][c], M.get("ref_props", {}).get(c)) for c in M["features"]}
    worst = max(psis, key=psis.get)
    _metrics({"max_psi": psis[worst], "n_predicciones_monitoreadas": len(rows)})
    print("max_psi", psis[worst], worst)
    return {"n": len(rows), "max_psi": psis[worst], "feature": worst}


def retrain(event=None, context=None):
    sm = boto3.client("sagemaker")
    name = "acuicultura-do-risk-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-auto"
    ep = ("import boto3;boto3.client('s3').download_file('%s','acuicultura/boot.py','/tmp/b.py');"  # noqa: UP031
          "exec(open('/tmp/b.py').read())" % SM_BUCKET)
    sm.create_training_job(
        TrainingJobName=name, RoleArn=ROLE_ARN,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File",
                               "ContainerEntrypoint": ["python", "-c", ep]},
        InputDataConfig=[{"ChannelName": "train", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{SM_BUCKET}/acuicultura/data/",
            "S3DataDistributionType": "FullyReplicated"}}}],
        OutputDataConfig={"S3OutputPath": f"s3://{SM_BUCKET}/acuicultura/output/"},
        ResourceConfig={"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 5},
        StoppingCondition={"MaxRuntimeInSeconds": 900},
        Tags=[{"Key": "project", "Value": "acuicultura-mlops"}, {"Key": "trigger", "Value": "eventbridge"}])
    print("job lanzado", name)
    return {"job": name}


def promote(event, context=None):
    d = event.get("detail", {})
    job = d.get("TrainingJobName", "")
    if not job.startswith("acuicultura-do-risk-") or d.get("TrainingJobStatus") != "Completed":
        return {"skipped": job}
    art = d["ModelArtifacts"]["S3ModelArtifacts"].replace("s3://", "")
    src_bucket, src_key = art.split("/", 1)
    boto3.client("s3").copy_object(Bucket=BUCKET, Key=f"models/{job}/model.tar.gz",
                                   CopySource={"Bucket": src_bucket, "Key": src_key})
    print("modelo copiado a models/" + job)
    return {"promoted": job}
