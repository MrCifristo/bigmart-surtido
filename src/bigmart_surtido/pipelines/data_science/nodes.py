"""Nodos de entrenamiento y evaluación de modelos."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# Colores de la gráfica SHAP (paleta categórica validada para daltonismo).
COLOR_TIENDA = "#2a78d6"
COLOR_PRODUCTO = "#eb6834"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_SECUNDARIO = "#52514e"
COLOR_FONDO = "#fcfcfb"


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


def construir_modelo(
    features_categoricas: list[str],
    hiperparametros: dict[str, Any],
    random_state: int,
) -> SkPipeline:
    """Arma (sin entrenar) un Random Forest con one-hot de las categóricas.

    Args:
        features_categoricas: Columnas que se codifican con one-hot; el resto
            pasa sin cambios.
        hiperparametros: Argumentos de ``RandomForestRegressor``.
        random_state: Semilla del bosque.

    Returns:
        Pipeline de scikit-learn sin entrenar.
    """
    preprocesador = ColumnTransformer(
        [
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                features_categoricas,
            )
        ],
        remainder="passthrough",
    )
    return SkPipeline(
        [
            ("preprocesador", preprocesador),
            (
                "regresor",
                RandomForestRegressor(
                    **hiperparametros,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


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
    modelo = construir_modelo(
        parametros["features_categoricas"],
        parametros["modelo"],
        parametros["random_state"],
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


def _resumir_folds(folds: list[dict[str, float]]) -> dict[str, Any]:
    tabla = pd.DataFrame(folds)
    return {
        "promedio": tabla.mean().round(4).to_dict(),
        "desviacion": tabla.std().round(4).to_dict(),
        "por_fold": folds,
    }


def validacion_cruzada(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parametros: dict[str, Any],
) -> dict[str, Any]:
    """Compara modelo y línea base con validación cruzada K-fold.

    Se hace sobre el conjunto de entrenamiento para dejar intacto el de
    prueba. En cada fold se entrenan desde cero la línea base y el modelo.

    Args:
        X_train: Features de entrenamiento.
        y_train: Objetivo de entrenamiento.
        parametros: Parámetros ``data_science``; usa ``cv_folds``,
            ``baseline_grupo``, ``features_categoricas``, ``modelo`` y
            ``random_state``.

    Returns:
        Promedio, desviación estándar y detalle por fold de MAE, RMSE y R²
        para ambos modelos, y en cuántos folds el modelo supera a la línea
        base en MAE.
    """
    y = y_train.iloc[:, 0]
    kfold = KFold(
        n_splits=parametros["cv_folds"],
        shuffle=True,
        random_state=parametros["random_state"],
    )
    folds_base, folds_modelo = [], []
    for idx_ent, idx_val in kfold.split(X_train):
        X_ent, X_val = X_train.iloc[idx_ent], X_train.iloc[idx_val]
        y_ent, y_val = y.iloc[idx_ent], y.iloc[idx_val]
        baseline = BaselinePorGrupo(parametros["baseline_grupo"]).fit(X_ent, y_ent)
        modelo = construir_modelo(
            parametros["features_categoricas"],
            parametros["modelo"],
            parametros["random_state"],
        ).fit(X_ent, y_ent)
        folds_base.append(_metricas(y_val, baseline.predict(X_val)))
        folds_modelo.append(_metricas(y_val, modelo.predict(X_val)))

    resultado = {
        "folds": parametros["cv_folds"],
        "baseline": _resumir_folds(folds_base),
        "modelo": _resumir_folds(folds_modelo),
        "folds_modelo_mejor_mae": sum(
            m["mae"] < b["mae"] for m, b in zip(folds_modelo, folds_base)
        ),
    }
    logger.info(
        "Validación cruzada (MAE promedio): baseline %s, modelo %s",
        resultado["baseline"]["promedio"]["mae"],
        resultado["modelo"]["promedio"]["mae"],
    )
    return resultado


def _feature_original(columna: str, categoricas: list[str]) -> str:
    """Traduce una columna transformada (p. ej. one-hot) a su feature original."""
    nombre = columna.split("__", 1)[1]
    for categorica in sorted(categoricas, key=len, reverse=True):
        if nombre.startswith(categorica + "_"):
            return categorica
    return nombre


def _grafica_shap(importancia: pd.DataFrame, efecto: pd.DataFrame) -> Figure:
    figura = Figure(figsize=(12, 5.5), facecolor=COLOR_FONDO, layout="constrained")
    ax_imp, ax_efecto = figura.subplots(1, 2, width_ratios=[1, 1])
    for ax in (ax_imp, ax_efecto):
        ax.set_facecolor(COLOR_FONDO)
        ax.tick_params(colors=COLOR_TEXTO_SECUNDARIO, length=0)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color("#d8d7d2")
        ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
        ax.set_axisbelow(True)

    orden = importancia.iloc[::-1]
    colores = np.where(orden["grupo"] == "Tienda", COLOR_TIENDA, COLOR_PRODUCTO)
    ax_imp.barh(orden["feature"], orden["pct_importancia"], color=colores, height=0.7)
    for y, pct in enumerate(orden["pct_importancia"]):
        ax_imp.text(
            pct + 0.8, y, f"{pct:.1f}%", va="center", fontsize=9, color=COLOR_TEXTO
        )
    ax_imp.set_xlabel("% de la importancia SHAP total", color=COLOR_TEXTO_SECUNDARIO)
    ax_imp.set_title(
        "¿Qué variables mueven las unidades?",
        loc="left",
        color=COLOR_TEXTO,
        fontweight="bold",
    )
    leyenda = [
        Patch(
            color=color,
            label=f"{grupo} "
            f"({importancia.loc[importancia['grupo'] == grupo, 'pct_importancia'].sum():.0f}%)",
        )
        for grupo, color in (("Tienda", COLOR_TIENDA), ("Producto", COLOR_PRODUCTO))
    ]
    ax_imp.legend(
        handles=leyenda, frameon=False, loc="lower right", labelcolor=COLOR_TEXTO
    )

    etiquetas = efecto["Outlet_Identifier"] + "  " + efecto["Outlet_Type"]
    ax_efecto.barh(etiquetas, efecto["efecto_unidades"], color=COLOR_TIENDA, height=0.7)
    ax_efecto.axvline(0, color=COLOR_TEXTO_SECUNDARIO, linewidth=0.8)
    for y, valor in enumerate(efecto["efecto_unidades"]):
        ax_efecto.text(
            valor + (0.4 if valor >= 0 else -0.4),
            y,
            f"{valor:+.1f}",
            va="center",
            ha="left" if valor >= 0 else "right",
            fontsize=9,
            color=COLOR_TEXTO,
        )
    limite = efecto["efecto_unidades"].abs().max() * 1.25
    ax_efecto.set_xlim(-limite, limite)
    ax_efecto.set_xlabel(
        "Unidades por producto frente al promedio", color=COLOR_TEXTO_SECUNDARIO
    )
    ax_efecto.set_title(
        "Efecto de cada tienda en las unidades",
        loc="left",
        color=COLOR_TEXTO,
        fontweight="bold",
    )
    return figura


def explicar_modelo(
    modelo: SkPipeline,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Figure]:
    """Explica el modelo con valores SHAP sobre el conjunto de prueba.

    Los valores SHAP de las columnas one-hot se suman para volver a la feature
    original. Las features que empiezan con ``Outlet_`` forman el grupo
    "Tienda" y las demás el grupo "Producto".

    Args:
        modelo: Pipeline entrenado (preprocesador + Random Forest).
        X_test: Features de prueba.

    Returns:
        - Importancia por feature: media del valor SHAP absoluto (en unidades)
          y su porcentaje del total, ordenada de mayor a menor.
        - Efecto de la tienda: suma de los SHAP del grupo "Tienda", promediada
          por tienda. Son las unidades por producto que la tienda suma o resta
          frente a la predicción promedio.
        - Figura con ambos resúmenes.
    """
    preprocesador = modelo.named_steps["preprocesador"]
    X_transformado = preprocesador.transform(X_test)
    if hasattr(X_transformado, "toarray"):
        X_transformado = X_transformado.toarray()
    categoricas = preprocesador.transformers_[0][2]
    columnas = [
        _feature_original(c, categoricas) for c in preprocesador.get_feature_names_out()
    ]

    valores = shap.TreeExplainer(modelo.named_steps["regresor"]).shap_values(
        X_transformado
    )
    por_feature = pd.DataFrame(valores, columns=columnas).T.groupby(level=0).sum().T

    importancia = por_feature.abs().mean().sort_values(ascending=False)
    importancia = pd.DataFrame(
        {
            "feature": importancia.index,
            "grupo": np.where(
                importancia.index.str.startswith("Outlet_"), "Tienda", "Producto"
            ),
            "shap_medio_abs": importancia.round(4).to_numpy(),
            "pct_importancia": (100 * importancia / importancia.sum())
            .round(2)
            .to_numpy(),
        }
    )

    columnas_tienda = [c for c in por_feature if c.startswith("Outlet_")]
    efecto = (
        por_feature[columnas_tienda]
        .sum(axis=1)
        .groupby(
            [X_test["Outlet_Identifier"].to_numpy(), X_test["Outlet_Type"].to_numpy()]
        )
        .mean()
        .rename_axis(["Outlet_Identifier", "Outlet_Type"])
        .rename("efecto_unidades")
        .round(2)
        .sort_values()
        .reset_index()
    )
    logger.info(
        "SHAP: el grupo Tienda explica el %.1f%% de la importancia",
        importancia.loc[importancia["grupo"] == "Tienda", "pct_importancia"].sum(),
    )
    return importancia, efecto, _grafica_shap(importancia, efecto)
