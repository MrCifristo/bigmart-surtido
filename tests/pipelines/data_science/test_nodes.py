import numpy as np
import pandas as pd
import pytest

from bigmart_surtido.pipelines.data_science.nodes import (
    BaselinePorGrupo,
    dividir_datos,
    entrenar_baseline,
    entrenar_modelo,
    evaluar_modelos,
)

PARAMETROS = {
    "target": "Item_Units",
    "features_numericas": ["Item_MRP"],
    "features_categoricas": ["Outlet_Type"],
    "baseline_grupo": "Outlet_Type",
    "test_size": 0.25,
    "random_state": 0,
    "modelo": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1},
}


@pytest.fixture
def features():
    rng = np.random.default_rng(0)
    n = 80
    tipo = rng.choice(["Grocery Store", "Supermarket Type1"], n)
    return pd.DataFrame(
        {
            "Item_MRP": rng.uniform(30, 260, n),
            "Outlet_Type": tipo,
            "Item_Units": np.where(tipo == "Grocery Store", 2.0, 16.0)
            + rng.normal(0, 1, n),
        }
    )


def test_baseline_predice_promedio_del_grupo():
    X = pd.DataFrame({"g": ["a", "a", "b"]})
    y = pd.Series([1.0, 3.0, 10.0])
    pred = (
        BaselinePorGrupo("g")
        .fit(X, y)
        .predict(pd.DataFrame({"g": ["a", "b", "nuevo"]}))
    )
    np.testing.assert_allclose(pred, [2.0, 10.0, 14 / 3])


def test_dividir_datos_tamanos(features):
    X_train, X_test, y_train, y_test = dividir_datos(features, PARAMETROS)
    assert len(X_test) == 20
    assert len(X_train) == len(y_train) == 60
    assert list(y_test.columns) == ["Item_Units"]
    assert "Item_Units" not in X_train.columns


def test_flujo_completo_de_evaluacion(features):
    X_train, X_test, y_train, y_test = dividir_datos(features, PARAMETROS)
    baseline = entrenar_baseline(X_train, y_train, PARAMETROS["baseline_grupo"])
    modelo = entrenar_modelo(X_train, y_train, PARAMETROS)
    metricas = evaluar_modelos(baseline, modelo, X_test, y_test)
    assert metricas["objetivo"] == "Item_Units"
    assert set(metricas["modelo"]) == {"mae", "rmse", "r2"}
    assert metricas["baseline"]["r2"] > 0.5
