import pandas as pd
import pytest

from bigmart_surtido.pipelines.data_processing.nodes import (
    crear_features,
    generar_reporte_calidad,
    imputar_faltantes,
    normalizar_datos,
)

MAPA_FAT = {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
MAPA_CAT = {"FD": "Comida", "DR": "Bebida", "NC": "No consumible"}


@pytest.fixture
def normalizado(raw_bigmart):
    return normalizar_datos(raw_bigmart, MAPA_FAT, MAPA_CAT)


def test_reporte_calidad_cuenta_problemas(raw_bigmart):
    reporte = generar_reporte_calidad(raw_bigmart)
    assert reporte["filas"] == 6
    assert reporte["faltantes"] == {"Item_Weight": 2, "Outlet_Size": 2}
    assert reporte["visibilidad_cero"] == 2
    assert reporte["duplicados_exactos"] == 0


def test_normalizar_unifica_fat_content(normalizado):
    assert set(normalizado["Item_Fat_Content"]) == {"Low Fat", "Regular", "No aplica"}


def test_normalizar_no_consumibles_sin_grasa(normalizado):
    no_consumibles = normalizado[normalizado["Item_Category"] == "No consumible"]
    assert (no_consumibles["Item_Fat_Content"] == "No aplica").all()


def test_normalizar_elimina_duplicados(raw_bigmart):
    duplicado = pd.concat([raw_bigmart, raw_bigmart.iloc[[0]]])
    assert len(normalizar_datos(duplicado, MAPA_FAT, MAPA_CAT)) == len(raw_bigmart)


def test_imputar_no_deja_faltantes(normalizado):
    limpio = imputar_faltantes(normalizado, 0.99)
    assert limpio.isna().sum().sum() == 0


def test_imputar_peso_con_mismo_producto(normalizado):
    limpio = imputar_faltantes(normalizado, 0.99)
    assert limpio.loc[1, "Item_Weight"] == pytest.approx(9.3)


def test_imputar_visibilidad_sin_ceros(normalizado):
    limpio = imputar_faltantes(normalizado, 0.99)
    assert (limpio["Item_Visibility"] > 0).all()


def test_imputar_tamano_por_tipo_de_tienda(normalizado):
    limpio = imputar_faltantes(normalizado, 0.99)
    # OUT017 es Supermarket Type1: toma la moda de las otras tiendas Type1.
    tipo1 = normalizado.loc[
        normalizado["Outlet_Type"] == "Supermarket Type1", "Outlet_Size"
    ]
    assert limpio.loc[4, "Outlet_Size"] in set(tipo1.dropna())


def test_crear_features_unidades_y_antiguedad(normalizado):
    features = crear_features(imputar_faltantes(normalizado, 0.99), 2013)
    assert features.loc[0, "Outlet_Age"] == 14
    assert features.loc[0, "Item_Units"] == pytest.approx(3735.1 / 249.8)
