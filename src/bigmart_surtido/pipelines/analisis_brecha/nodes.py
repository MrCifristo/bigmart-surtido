"""Nodos del análisis de brecha entre ventas reales y potenciales por tienda.

El potencial de una tienda es lo que vendería, con su mismo surtido, una tienda
del formato de referencia (por defecto, Supermarket Type1). Se estima con un
modelo entrenado solo con productos de ese formato y atributos del producto.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bigmart_surtido.pipelines.data_science.nodes import construir_modelo

logger = logging.getLogger(__name__)

COLUMNAS_ID = [
    "Item_Identifier",
    "Item_Type",
    "Item_MRP",
    "Outlet_Identifier",
    "Outlet_Type",
    "Outlet_Size",
    "Outlet_Location_Type",
]


def estimar_potencial(
    df: pd.DataFrame,
    parametros: dict[str, Any],
    hiperparametros: dict[str, Any],
    random_state: int,
) -> pd.DataFrame:
    """Estima las unidades que cada producto vendería en una tienda de referencia.

    Para cada tienda se entrena el modelo con las tiendas del formato de
    referencia **excluyendo a la propia tienda**. Así la predicción siempre es
    fuera de muestra: una tienda Type1 se compara contra las otras Type1, y
    las demás contra todas las Type1.

    Args:
        df: Dataset con features (una fila por producto y tienda).
        parametros: Parámetros ``analisis_brecha``; usa ``target``,
            ``formato_referencia``, ``features_numericas`` y
            ``features_categoricas``.
        hiperparametros: Hiperparámetros del Random Forest.
        random_state: Semilla del modelo.

    Returns:
        Una fila por producto y tienda con las unidades reales, las esperadas
        y la brecha en unidades y en Q (brecha en unidades × precio).
    """
    target = parametros["target"]
    features = parametros["features_numericas"] + parametros["features_categoricas"]
    referencia = df[df["Outlet_Type"] == parametros["formato_referencia"]]
    if referencia.empty:
        raise ValueError(
            f"No hay tiendas del formato {parametros['formato_referencia']!r}"
        )

    esperado = pd.Series(np.nan, index=df.index)
    for tienda, filas in df.groupby("Outlet_Identifier"):
        entrenamiento = referencia[referencia["Outlet_Identifier"] != tienda]
        modelo = construir_modelo(
            parametros["features_categoricas"], hiperparametros, random_state
        ).fit(entrenamiento[features], entrenamiento[target])
        esperado.loc[filas.index] = modelo.predict(filas[features])

    resultado = df[COLUMNAS_ID].copy()
    resultado["unidades_reales"] = df[target]
    resultado["unidades_esperadas"] = esperado
    resultado["brecha_unidades"] = resultado["unidades_reales"] - esperado
    resultado["brecha_q"] = resultado["brecha_unidades"] * df["Item_MRP"]
    return resultado


def resumir_brecha(ventas: pd.DataFrame, niveles: list[str]) -> pd.DataFrame:
    """Agrega la brecha por los niveles indicados (tienda, formato, categoría...).

    Args:
        ventas: Salida de ``estimar_potencial``.
        niveles: Columnas por las que se agrupa.

    Returns:
        Por grupo: productos, unidades reales y esperadas, brecha en unidades,
        en porcentaje y en Q, ventas reales en Q, proporción de productos por
        debajo de lo esperado y el estadístico t de la brecha media por
        producto (|t| > 2 indica una brecha consistente, no ruido). Ordenado de
        la mayor brecha negativa en Q a la mayor positiva.
    """
    ventas = ventas.assign(
        ventas_q=ventas["unidades_reales"] * ventas["Item_MRP"],
        bajo_esperado=ventas["brecha_unidades"] < 0,
    )
    resumen = ventas.groupby(niveles).agg(
        productos=("brecha_unidades", "size"),
        unidades_reales=("unidades_reales", "sum"),
        unidades_esperadas=("unidades_esperadas", "sum"),
        brecha_unidades=("brecha_unidades", "sum"),
        ventas_q=("ventas_q", "sum"),
        brecha_q=("brecha_q", "sum"),
        pct_productos_bajo_esperado=("bajo_esperado", "mean"),
        _media=("brecha_unidades", "mean"),
        _desv=("brecha_unidades", "std"),
    )
    resumen["brecha_pct"] = 100 * (
        resumen["unidades_reales"] / resumen["unidades_esperadas"] - 1
    )
    resumen["t_estadistico"] = resumen["_media"] / (
        resumen["_desv"] / np.sqrt(resumen["productos"])
    )
    resumen["pct_productos_bajo_esperado"] *= 100
    columnas = [
        "productos",
        "unidades_reales",
        "unidades_esperadas",
        "brecha_unidades",
        "brecha_pct",
        "ventas_q",
        "brecha_q",
        "pct_productos_bajo_esperado",
        "t_estadistico",
    ]
    return resumen[columnas].round(2).sort_values("brecha_q").reset_index()
