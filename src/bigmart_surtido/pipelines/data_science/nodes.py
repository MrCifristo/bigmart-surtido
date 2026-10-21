"""Nodos de entrenamiento y evaluación de modelos."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class BaselinePorGrupo:
    """Línea base: predice el promedio del objetivo de cada grupo.

    Con ``grupo = "Outlet_Type"`` equivale a la regla de negocio "cada producto
    vende lo que vende en promedio un producto en ese formato de tienda".
    """

    def __init__(self, grupo: str):
        """Guarda la columna que define los grupos."""
        self.grupo = grupo
        self.promedios_: pd.Series | None = None
        self.promedio_global_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaselinePorGrupo:
        """Calcula el promedio del objetivo por grupo."""
        self.promedios_ = y.groupby(X[self.grupo].to_numpy()).mean()
        self.promedio_global_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve el promedio del grupo; usa el global si el grupo es nuevo."""
        return (
            X[self.grupo].map(self.promedios_).fillna(self.promedio_global_).to_numpy()
        )


def dividir_datos(
    df: pd.DataFrame, parametros: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa features y objetivo en conjuntos de entrenamiento y prueba.

    Args:
        df: Dataset con features.
        parametros: Parámetros ``data_science`` (target, features, test_size,
            random_state).

    Returns:
        ``X_train``, ``X_test``, ``y_train`` y ``y_test``. Los objetivos se
        devuelven como DataFrame de una columna para guardarlos en CSV.
    """
    features = parametros["features_numericas"] + parametros["features_categoricas"]
    X = df[features]
    y = df[[parametros["target"]]]
    return tuple(
        train_test_split(
            X,
            y,
            test_size=parametros["test_size"],
            random_state=parametros["random_state"],
        )
    )


def entrenar_baseline(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    grupo: str,
) -> BaselinePorGrupo:
    """Entrena la línea base de promedio por grupo."""
    return BaselinePorGrupo(grupo).fit(X_train, y_train.iloc[:, 0])


def entrenar_modelo(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parametros: dict[str, Any],
) -> SkPipeline:
    """Entrena un Random Forest con codificación one-hot de las categóricas.

    Args:
        X_train: Features de entrenamiento.
        y_train: Objetivo de entrenamiento.
        parametros: Parámetros ``data_science``; usa ``features_categoricas``,
            ``random_state`` y los hiperparámetros de ``modelo``.

    Returns:
        Pipeline de scikit-learn entrenado.
    """
    preprocesador = ColumnTransformer(
        [
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                parametros["features_categoricas"],
            )
        ],
        remainder="passthrough",
    )
    modelo = SkPipeline(
        [
            ("preprocesador", preprocesador),
            (
                "regresor",
                RandomForestRegressor(
                    **parametros["modelo"],
                    random_state=parametros["random_state"],
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return modelo.fit(X_train, y_train.iloc[:, 0])


def _metricas(y_real: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_real, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_real, y_pred))), 4),
        "r2": round(float(r2_score(y_real, y_pred)), 4),
    }


def evaluar_modelos(
    modelo_baseline: BaselinePorGrupo,
    modelo: SkPipeline,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict[str, Any]:
    """Compara el modelo contra la línea base en el conjunto de prueba.

    Returns:
        Métricas (MAE, RMSE, R²) de ambos modelos y la mejora porcentual del
        MAE del modelo frente a la línea base.
    """
    y_real = y_test.iloc[:, 0]
    resultado = {
        "objetivo": y_test.columns[0],
        "baseline": _metricas(y_real, modelo_baseline.predict(X_test)),
        "modelo": _metricas(y_real, modelo.predict(X_test)),
    }
    mae_base = resultado["baseline"]["mae"]
    resultado["mejora_mae_vs_baseline_pct"] = round(
        100 * (mae_base - resultado["modelo"]["mae"]) / mae_base, 2
    )
    logger.info("Métricas de evaluación: %s", resultado)
    return resultado
