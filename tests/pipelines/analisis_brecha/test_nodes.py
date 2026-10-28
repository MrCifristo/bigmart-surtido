import numpy as np
import pandas as pd
import pytest

from bigmart_surtido.pipelines.analisis_brecha.nodes import (
    estimar_potencial,
    resumir_brecha,
)

PARAMETROS = {
    "target": "Item_Units",
    "formato_referencia": "Supermarket Type1",
    "features_numericas": ["Item_MRP"],
    "features_categoricas": ["Item_Type"],
}
HIPERPARAMETROS = {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1}

# Unidades por producto de cada tienda: dos Type1 y una Grocery.
TIENDAS = {
    ("OUT_A", "Supermarket Type1"): 10.0,
    ("OUT_B", "Supermarket Type1"): 20.0,
    ("OUT_G", "Grocery Store"): 2.0,
}


@pytest.fixture
def features():
    filas = []
    for (tienda, formato), unidades in TIENDAS.items():
        for i in range(5):
            filas.append(
                {
                    "Item_Identifier": f"FD{i}",
                    "Item_Type": "Dairy",
                    "Item_MRP": 100.0,
                    "Outlet_Identifier": tienda,
                    "Outlet_Type": formato,
                    "Outlet_Size": "Medium",
                    "Outlet_Location_Type": "Tier 1",
                    "Item_Units": unidades,
                }
            )
    return pd.DataFrame(filas)


def test_potencial_excluye_a_la_propia_tienda(features):
    ventas = estimar_potencial(features, PARAMETROS, HIPERPARAMETROS, 0)
    esperado = ventas.groupby("Outlet_Identifier")["unidades_esperadas"].mean()
    # Cada Type1 se compara solo contra la otra Type1.
    assert esperado["OUT_A"] == pytest.approx(20.0)
    assert esperado["OUT_B"] == pytest.approx(10.0)
    # Las demás tiendas se comparan contra todas las Type1.
    assert esperado["OUT_G"] == pytest.approx(15.0, abs=2.0)


def test_brecha_en_q_es_unidades_por_precio(features):
    ventas = estimar_potencial(features, PARAMETROS, HIPERPARAMETROS, 0)
    np.testing.assert_allclose(
        ventas["brecha_q"], ventas["brecha_unidades"] * ventas["Item_MRP"]
    )


def test_formato_de_referencia_inexistente(features):
    with pytest.raises(ValueError, match="Type9"):
        estimar_potencial(
            features,
            {**PARAMETROS, "formato_referencia": "Supermarket Type9"},
            HIPERPARAMETROS,
            0,
        )


def test_resumir_brecha():
    ventas = pd.DataFrame(
        {
            "Outlet_Identifier": ["X", "X", "Y", "Y"],
            "Item_MRP": [10.0, 20.0, 10.0, 10.0],
            "unidades_reales": [4.0, 6.0, 12.0, 14.0],
            "unidades_esperadas": [5.0, 10.0, 10.0, 10.0],
        }
    )
    ventas["brecha_unidades"] = ventas["unidades_reales"] - ventas["unidades_esperadas"]
    ventas["brecha_q"] = ventas["brecha_unidades"] * ventas["Item_MRP"]

    resumen = resumir_brecha(ventas, ["Outlet_Identifier"]).set_index(
        "Outlet_Identifier"
    )
    # Ordenado de la brecha más negativa a la más positiva.
    assert list(resumen.index) == ["X", "Y"]
    assert resumen.loc["X", "brecha_q"] == pytest.approx(-90.0)
    assert resumen.loc["X", "brecha_pct"] == pytest.approx(100 * (10 / 15 - 1), 0.01)
    assert resumen.loc["X", "pct_productos_bajo_esperado"] == 100
    assert resumen.loc["Y", "ventas_q"] == pytest.approx(260.0)
    assert resumen.loc["Y", "t_estadistico"] == pytest.approx(3.0)
