"""Monitoreo: drift de datos (PSI) y publicación de métricas a CloudWatch."""
import numpy as np

PSI_ALERT = 0.2  # regla común: PSI > 0.2 = cambio significativo


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index entre una muestra de referencia y una actual."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(X_ref, X_new, cols):
    """PSI por feature y bandera de drift."""
    scores = {c: psi(X_ref[c], X_new[c]) for c in cols}
    return {"psi": scores, "max_psi": max(scores.values()), "drift": max(scores.values()) > PSI_ALERT}


def publish_metrics(metrics: dict, namespace: str = "Acuicultura/MLOps", region: str = "us-east-1"):
    """Envía métricas personalizadas a CloudWatch (requiere credenciales de AWS)."""
    import boto3

    cw = boto3.client("cloudwatch", region_name=region)
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[{"MetricName": k, "Value": float(v)} for k, v in metrics.items()],
    )
