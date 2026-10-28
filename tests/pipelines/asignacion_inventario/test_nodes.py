import copy

import numpy as np
import pandas as pd
import pytest

from bigmart_surtido.pipelines.asignacion_inventario.nodes import (
    ReglaCuantilica,
    ajustar_modelo_cuantilico,
    analizar_sensibilidad,
    comparar_reglas_planificacion,
    construir_regla_asignacion,
    costo_asignacion,
    cuantil_critico,
    evaluar_escenarios_costo,
    graficar_asignacion,
    marcar_perecederos,
    validar_parametros,
)

PARAMETROS = {
    "target": "Item_Units",
    "grupo_validacion": "Item_Identifier",
    "cv_folds": 3,
    "random_state": 0,
    "categorias_perecederas": ["Dairy"],
    "features_numericas": ["Item_MRP"],
    "features_categoricas": ["Outlet_Type", "Tipo_Producto"],
    "cuantil_ajuste": 0.75,
    "grilla_hiperparametros": {"max_depth": [2], "max_iter": [20, 40]},
    "escenarios_costo": [{"nombre": "Iguales", "faltante": 1, "sobrante": 1}],
    "costos_por_tipo": {
        "Perecedero": {"faltante": 1, "sobrante": 3},
        "No perecedero": {"faltante": 3, "sobrante": 1},
    },
    "ratios_sensibilidad": [0.5, 1, 2],
}


@pytest.fixture
def datos():
    """Dos formatos y 40 productos; las unidades dependen solo del formato."""
    rng = np.random.default_rng(0)
    filas = []
    for i in range(40):
        for tienda, formato, media in (
            ("OUT_G", "Grocery Store", 3.0),
            ("OUT_A", "Supermarket Type1", 16.0),
            ("OUT_B", "Supermarket Type1", 16.0),
        ):
            filas.append(
                {
                    "Item_Identifier": f"P{i}",
                    "Item_Type": "Dairy" if i % 2 else "Household",
                    "Item_MRP": rng.uniform(30, 260),
                    "Outlet_Identifier": tienda,
                    "Outlet_Type": formato,
                    "Item_Units": max(0.5, rng.normal(media, media / 3)),
                }
            )
    return marcar_perecederos(pd.DataFrame(filas), ["Dairy"])


def test_cuantil_critico():
    assert cuantil_critico(1, 1) == 0.5
    assert cuantil_critico(3, 1) == 0.75
    assert cuantil_critico(1, 3) == 0.25


def test_costo_asignacion_pondera_faltante_y_sobrante():
    # Faltan 2 unidades (costo 3 c/u) y sobra 1 (costo 1).
    assert costo_asignacion([10, 5], [8, 6], 3, 1) == 7
    # Costos distintos por fila.
    assert costo_asignacion([10, 5], [8, 6], np.array([1, 1]), np.array([1, 4])) == 6


def test_marcar_perecederos(datos):
    tipos = datos.groupby("Item_Type")["Tipo_Producto"].first()
    assert tipos["Dairy"] == "Perecedero"
    assert tipos["Household"] == "No perecedero"


def test_regla_cuantilica_usa_el_cuantil_de_cada_tipo():
    df = pd.DataFrame(
        {
            "Outlet_Type": ["T1"] * 8,
            "Tipo_Producto": ["Perecedero"] * 4 + ["No perecedero"] * 4,
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    regla = ReglaCuantilica({"Perecedero": 0.0, "No perecedero": 1.0}).fit(df, y)
    nuevo = pd.DataFrame(
        {
            "Outlet_Type": ["T1", "T1", "Nuevo"],
            "Tipo_Producto": ["Perecedero", "No perecedero", "No perecedero"],
        }
    )
    # Formato desconocido: usa el cuantil del tipo sobre todos los formatos.
    np.testing.assert_allclose(regla.predict(nuevo), [1.0, 4.0, 4.0])


def test_comparar_reglas_formato_supera_a_repartir_igual(datos):
    resultado = comparar_reglas_planificacion(datos, PARAMETROS)
    reglas = resultado["reglas"]
    assert set(reglas) == {
        "Repartir igual",
        "Historial del producto",
        "Promedio por formato",
    }
    # Las unidades dependen solo del formato: el historial del producto no ayuda.
    assert (
        reglas["Promedio por formato"]["mae"]
        < reglas["Historial del producto"]["mae"]
        < reglas["Repartir igual"]["mae"]
    )


def test_ajuste_y_escenarios(datos):
    ajuste = ajustar_modelo_cuantilico(datos, PARAMETROS)
    assert len(ajuste["resultados"]) == 2
    assert set(ajuste["mejores_hiperparametros"]) == {"max_depth", "max_iter"}

    costos = evaluar_escenarios_costo(datos, PARAMETROS, ajuste)
    assert set(costos["escenario"]) == {"Iguales", "Según tipo de producto"}
    assert len(costos) == 6
    base = costos[costos["regla"] == "Promedio por formato"]
    assert (base["ahorro_vs_promedio_pct"] == 0).all()
    # Con costos asimétricos, enviar el cuantil correcto cuesta menos que el promedio.
    por_tipo = costos[
        (costos["escenario"] == "Según tipo de producto")
        & (costos["regla"] == "Cuantil por formato")
    ]
    assert por_tipo["ahorro_vs_promedio_pct"].iloc[0] > 0


def test_sensibilidad_y_grafica(datos):
    sensibilidad = analizar_sensibilidad(datos, PARAMETROS)
    assert list(sensibilidad["cuantil"]) == pytest.approx([1 / 3, 0.5, 2 / 3], abs=1e-3)
    regla, tabla = construir_regla_asignacion(datos, PARAMETROS)
    assert len(tabla) == 4
    # Perecederos se envían por debajo del promedio y no perecederos por encima.
    tabla = tabla.set_index(["Outlet_Type", "Tipo_Producto"])
    fila_p = tabla.loc[("Supermarket Type1", "Perecedero")]
    fila_np = tabla.loc[("Supermarket Type1", "No perecedero")]
    assert fila_p["unidades_a_enviar"] < fila_p["unidades_promedio"]
    assert fila_np["unidades_a_enviar"] > fila_np["unidades_promedio"]
    figura = graficar_asignacion(sensibilidad, tabla.reset_index())
    assert len(figura.axes) == 2


def test_validar_parametros_validos_devuelve_el_mismo_dict():
    assert validar_parametros(PARAMETROS) is PARAMETROS


def _con(ruta, valor):
    """Copia de PARAMETROS con ``valor`` en ``ruta`` (lista de claves)."""
    parametros = copy.deepcopy(PARAMETROS)
    destino = parametros
    for clave in ruta[:-1]:
        destino = destino[clave]
    destino[ruta[-1]] = valor
    return parametros


@pytest.mark.parametrize(
    ("ruta", "valor", "mensaje"),
    [
        (["target"], "", "target"),
        (["target"], None, "target"),
        (["cv_folds"], 1, "cv_folds"),
        (["cv_folds"], 2.5, "cv_folds"),
        (["categorias_perecederas"], [], "categorias_perecederas"),
        (["cuantil_ajuste"], 0, "cuantil_ajuste"),
        (["cuantil_ajuste"], 1, "cuantil_ajuste"),
        (["cuantil_ajuste"], "0.5", "cuantil_ajuste"),
        (["escenarios_costo"], [], "escenarios_costo"),
        (["escenarios_costo", 0, "nombre"], "", r"escenarios_costo\[0\]\.nombre"),
        (["escenarios_costo", 0, "faltante"], -1, r"escenarios_costo\[0\]\.faltante"),
        (["escenarios_costo", 0, "sobrante"], 0, r"escenarios_costo\[0\]\.sobrante"),
        (
            ["costos_por_tipo"],
            {"Perecedero": {"faltante": 1, "sobrante": 3}},
            "costos_por_tipo",
        ),
        (
            ["costos_por_tipo", "Perecedero", "faltante"],
            -1,
            r"costos_por_tipo\.Perecedero\.faltante",
        ),
        (
            ["costos_por_tipo", "No perecedero", "sobrante"],
            0,
            r"costos_por_tipo\.No perecedero\.sobrante",
        ),
        (["ratios_sensibilidad"], [], "ratios_sensibilidad"),
        (["ratios_sensibilidad"], [1, 0], "ratios_sensibilidad"),
        (["grilla_hiperparametros"], {}, "grilla_hiperparametros"),
        (["grilla_hiperparametros"], {"max_depth": []}, "grilla_hiperparametros"),
    ],
)
def test_validar_parametros_invalidos(ruta, valor, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        validar_parametros(_con(ruta, valor))
