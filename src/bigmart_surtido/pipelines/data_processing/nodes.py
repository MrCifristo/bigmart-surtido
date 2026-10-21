"""Nodos de limpieza y transformación del dataset BigMart."""

from __future__ import annotations

from typing import Any

import pandas as pd

COLUMNAS_TEXTO = [
    "Item_Identifier",
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]


def generar_reporte_calidad(raw: pd.DataFrame) -> dict[str, Any]:
    """Resume los problemas de calidad del dataset crudo.

    Args:
        raw: Dataset BigMart tal como viene de Kaggle.

    Returns:
        Diccionario con filas, faltantes por columna, duplicados, visibilidades
        en cero y etiquetas encontradas en ``Item_Fat_Content``.
    """
    return {
        "filas": int(len(raw)),
        "columnas": int(raw.shape[1]),
        "faltantes": {k: int(v) for k, v in raw.isna().sum().items() if v > 0},
        "duplicados_exactos": int(raw.duplicated().sum()),
        "duplicados_producto_tienda": int(
            raw.duplicated(["Item_Identifier", "Outlet_Identifier"]).sum()
        ),
        "visibilidad_cero": int((raw["Item_Visibility"] == 0).sum()),
        "etiquetas_fat_content": {
            k: int(v) for k, v in raw["Item_Fat_Content"].value_counts().items()
        },
    }


def normalizar_datos(
    raw: pd.DataFrame,
    mapa_fat_content: dict[str, str],
    mapa_categoria_producto: dict[str, str],
) -> pd.DataFrame:
    """Elimina duplicados y unifica etiquetas categóricas.

    Args:
        raw: Dataset BigMart crudo.
        mapa_fat_content: Equivalencias para unificar ``Item_Fat_Content``.
        mapa_categoria_producto: Prefijo de ``Item_Identifier`` a categoría.

    Returns:
        Dataset sin duplicados, con etiquetas homogéneas y la columna
        ``Item_Category``. Los productos no consumibles quedan con
        ``Item_Fat_Content = "No aplica"``.
    """
    df = raw.drop_duplicates().copy()
    for col in COLUMNAS_TEXTO:
        df[col] = df[col].str.strip()

    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(mapa_fat_content)
    df["Item_Category"] = df["Item_Identifier"].str[:2].map(mapa_categoria_producto)
    df.loc[df["Item_Category"] == "No consumible", "Item_Fat_Content"] = "No aplica"
    return df


def imputar_faltantes(
    df: pd.DataFrame, percentil_tope_visibilidad: float
) -> pd.DataFrame:
    """Imputa faltantes y trata valores imposibles o extremos.

    - ``Item_Weight``: el peso de un producto es el mismo en todas las tiendas,
      así que se usa el peso del mismo producto en otra tienda; si no existe,
      la mediana global.
    - ``Item_Visibility``: un valor de 0 es imposible para un producto que se
      vendió, se trata como faltante y se imputa con la media del producto.
      Después se recortan los extremos en el percentil indicado.
    - ``Outlet_Size``: se imputa con la moda de las tiendas del mismo
      ``Outlet_Type``.

    Args:
        df: Dataset normalizado.
        percentil_tope_visibilidad: Percentil (0-1) usado como tope superior
            de ``Item_Visibility``.

    Returns:
        Dataset sin valores faltantes.
    """
    df = df.copy()
    por_producto = df.groupby("Item_Identifier")

    df["Item_Weight"] = df["Item_Weight"].fillna(
        por_producto["Item_Weight"].transform("mean")
    )
    df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].median())

    visibilidad = df["Item_Visibility"].mask(df["Item_Visibility"] == 0)
    visibilidad = visibilidad.fillna(
        visibilidad.groupby(df["Item_Identifier"]).transform("mean")
    )
    visibilidad = visibilidad.fillna(visibilidad.median())
    df["Item_Visibility"] = visibilidad.clip(
        upper=visibilidad.quantile(percentil_tope_visibilidad)
    )

    moda_por_tipo = (
        df.dropna(subset=["Outlet_Size"])
        .groupby("Outlet_Type")["Outlet_Size"]
        .agg(lambda s: s.mode().iloc[0])
    )
    df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Type"].map(moda_por_tipo))
    df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Size"].mode().iloc[0])
    return df


def crear_features(df: pd.DataFrame, anio_referencia: int) -> pd.DataFrame:
    """Crea variables derivadas para el modelado.

    Args:
        df: Dataset limpio.
        anio_referencia: Año de los datos, para calcular la antigüedad de la tienda.

    Returns:
        Dataset con ``Outlet_Age`` y ``Item_Units`` (ventas / precio), que
        separa la demanda en unidades del efecto mecánico del precio.
    """
    df = df.copy()
    df["Outlet_Age"] = anio_referencia - df["Outlet_Establishment_Year"]
    df["Item_Units"] = df["Item_Outlet_Sales"] / df["Item_MRP"]
    return df
