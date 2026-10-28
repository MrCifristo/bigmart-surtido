"""Nodos de la asignación de inventario por producto y tienda.

Pregunta: cuántas unidades enviar de cada producto a cada tienda cuando faltar
(venta perdida) y sobrar (merma o bodega) no cuestan lo mismo. Con costos
asimétricos, la cantidad que minimiza el costo esperado es un cuantil de la
demanda (problema del vendedor de periódicos):

    cuantil = costo_faltante / (costo_faltante + costo_sobrante)

Todas las reglas se evalúan con validación cruzada agrupada por producto,
comparando lo enviado contra las unidades que realmente se vendieron.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GroupKFold, ParameterGrid

from bigmart_surtido.estilo import (
    COLOR_FONDO,
    COLOR_PRODUCTO,
    COLOR_TEXTO,
    COLOR_TEXTO_SECUNDARIO,
    COLOR_TIENDA,
    estilo_ejes,
    titulo,
)
from bigmart_surtido.pipelines.data_science.nodes import BaselinePorGrupo

logger = logging.getLogger(__name__)

TIPO = "Tipo_Producto"
PERECEDERO = "Perecedero"
NO_PERECEDERO = "No perecedero"
TIPOS = [PERECEDERO, NO_PERECEDERO]
MIN_CV_FOLDS = 2


def cuantil_critico(costo_faltante: float, costo_sobrante: float) -> float:
    """Cuantil de la demanda que minimiza el costo esperado."""
    return costo_faltante / (costo_faltante + costo_sobrante)


def costo_asignacion(
    real: np.ndarray,
    enviado: np.ndarray,
    costo_faltante: float | np.ndarray,
    costo_sobrante: float | np.ndarray,
) -> float:
    """Costo total de enviar ``enviado`` cuando se vendió ``real``.

    Cada unidad que faltó cuesta ``costo_faltante`` y cada unidad que sobró
    cuesta ``costo_sobrante``. Los costos pueden variar por fila.
    """
    error = np.asarray(real, dtype=float) - np.asarray(enviado, dtype=float)
    return float(
        np.sum(np.where(error > 0, costo_faltante * error, -costo_sobrante * error))
    )


class ReglaCuantilica:
    """Envía el cuantil de las unidades vendidas por formato y tipo de producto.

    Es la regla recomendada: con ``{"Perecedero": 0.25, "No perecedero": 0.75}``
    envía a cada tienda lo que su formato vende en el 25% o el 75% de los casos,
    según el tipo de producto.
    """

    def __init__(self, cuantiles: dict[str, float]):
        """Guarda el cuantil que corresponde a cada tipo de producto."""
        self.cuantiles = cuantiles
        self.tabla_: pd.Series | None = None
        self.respaldo_: dict[str, float] | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series) -> ReglaCuantilica:
        """Calcula el cuantil de unidades de cada formato y tipo de producto."""
        partes, self.respaldo_ = [], {}
        for tipo, cuantil in self.cuantiles.items():
            filas = df[TIPO] == tipo
            partes.append(
                y[filas]
                .groupby([df.loc[filas, "Outlet_Type"], df.loc[filas, TIPO]])
                .quantile(cuantil)
            )
            self.respaldo_[tipo] = float(y[filas].quantile(cuantil))
        self.tabla_ = pd.concat(partes)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Unidades a enviar; usa el cuantil del tipo si el formato es nuevo."""
        claves = pd.MultiIndex.from_frame(df[["Outlet_Type", TIPO]])
        prediccion = self.tabla_.reindex(claves).to_numpy()
        respaldo = df[TIPO].map(self.respaldo_).to_numpy(dtype=float)
        return np.where(np.isnan(prediccion), respaldo, prediccion)


def _folds(df: pd.DataFrame, parametros: dict[str, Any]) -> list:
    grupos = df[parametros["grupo_validacion"]]
    return list(GroupKFold(n_splits=parametros["cv_folds"]).split(df, groups=grupos))


def _matriz_modelo(df: pd.DataFrame, parametros: dict[str, Any]) -> pd.DataFrame:
    X = df[parametros["features_numericas"] + parametros["features_categoricas"]]
    return X.astype({c: "category" for c in parametros["features_categoricas"]})


def _modelo_cuantilico(
    cuantil: float, hiperparametros: dict[str, Any], random_state: int
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=cuantil,
        categorical_features="from_dtype",
        random_state=random_state,
        **hiperparametros,
    )


def _costos_escenario(escenario: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Costo de faltar y de sobrar por tipo de producto en un escenario."""
    if "por_tipo" in escenario:
        return {
            tipo: (costos["faltante"], costos["sobrante"])
            for tipo, costos in escenario["por_tipo"].items()
        }
    return {tipo: (escenario["faltante"], escenario["sobrante"]) for tipo in TIPOS}


def _es_numero(valor: Any) -> bool:
    return isinstance(valor, int | float) and not isinstance(valor, bool)


def _exigir(condicion: bool, nombre: str, valor: Any, regla: str) -> None:
    if not condicion:
        raise ValueError(
            f"Parámetro inválido '{nombre}': {regla}. Valor recibido: {valor!r}"
        )


def _validar_costos(costos: Any, nombre: str) -> None:
    _exigir(isinstance(costos, dict), nombre, costos, "debe ser un diccionario")
    for clave in ("faltante", "sobrante"):
        valor = costos.get(clave)
        _exigir(
            _es_numero(valor) and valor > 0,
            f"{nombre}.{clave}",
            valor,
            "debe ser un número mayor que 0",
        )


def validar_parametros(parametros: dict[str, Any]) -> dict[str, Any]:
    """Revisa los parámetros de ``asignacion_inventario`` antes de usarlos.

    Detiene el pipeline al inicio con un mensaje claro, en lugar de un error
    confuso de pandas o sklearn (o resultados sin sentido) más adelante.

    Args:
        parametros: Parámetros ``asignacion_inventario``.

    Returns:
        Los mismos parámetros, sin cambios.

    Raises:
        ValueError: Si algún parámetro no cumple su regla. El mensaje nombra el
            parámetro y el valor recibido.
    """
    p = parametros
    target = p.get("target")
    _exigir(
        isinstance(target, str) and target.strip() != "",
        "target",
        target,
        "debe ser un texto no vacío",
    )
    cv_folds = p.get("cv_folds")
    _exigir(
        isinstance(cv_folds, int)
        and not isinstance(cv_folds, bool)
        and cv_folds >= MIN_CV_FOLDS,
        "cv_folds",
        cv_folds,
        f"debe ser un entero mayor o igual a {MIN_CV_FOLDS}",
    )
    perecederas = p.get("categorias_perecederas")
    _exigir(
        isinstance(perecederas, list) and len(perecederas) > 0,
        "categorias_perecederas",
        perecederas,
        "debe ser una lista no vacía",
    )
    cuantil = p.get("cuantil_ajuste")
    _exigir(
        _es_numero(cuantil) and 0 < cuantil < 1,
        "cuantil_ajuste",
        cuantil,
        "debe ser un número estrictamente entre 0 y 1",
    )

    escenarios = p.get("escenarios_costo")
    _exigir(
        isinstance(escenarios, list) and len(escenarios) > 0,
        "escenarios_costo",
        escenarios,
        "debe ser una lista no vacía",
    )
    for i, escenario in enumerate(escenarios):
        nombre = f"escenarios_costo[{i}]"
        _validar_costos(escenario, nombre)
        etiqueta = escenario.get("nombre")
        _exigir(
            isinstance(etiqueta, str) and etiqueta.strip() != "",
            f"{nombre}.nombre",
            etiqueta,
            "debe ser un texto no vacío",
        )

    costos_por_tipo = p.get("costos_por_tipo")
    _exigir(
        isinstance(costos_por_tipo, dict) and set(costos_por_tipo) == set(TIPOS),
        "costos_por_tipo",
        costos_por_tipo,
        f"debe tener exactamente las claves {TIPOS}",
    )
    for tipo, costos in costos_por_tipo.items():
        _validar_costos(costos, f"costos_por_tipo.{tipo}")

    ratios = p.get("ratios_sensibilidad")
    _exigir(
        isinstance(ratios, list)
        and len(ratios) > 0
        and all(_es_numero(r) and r > 0 for r in ratios),
        "ratios_sensibilidad",
        ratios,
        "debe ser una lista no vacía de números mayores que 0",
    )
    grilla = p.get("grilla_hiperparametros")
    _exigir(
        isinstance(grilla, dict)
        and len(grilla) > 0
        and all(isinstance(v, list) and len(v) > 0 for v in grilla.values()),
        "grilla_hiperparametros",
        grilla,
        "debe ser un diccionario no vacío cuyos valores son listas no vacías",
    )
    return parametros


def marcar_perecederos(
    df: pd.DataFrame, categorias_perecederas: list[str]
) -> pd.DataFrame:
    """Agrega la columna ``Tipo_Producto`` (perecedero o no perecedero)."""
    return df.assign(
        **{
            TIPO: np.where(
                df["Item_Type"].isin(categorias_perecederas), PERECEDERO, NO_PERECEDERO
            )
        }
    )


def comparar_reglas_planificacion(
    df: pd.DataFrame, parametros: dict[str, Any]
) -> dict[str, Any]:
    """Compara tres formas de decidir cuánto enviar, con costos iguales.

    - **Repartir igual:** la misma cantidad a todas las tiendas.
    - **Historial del producto:** lo que el producto vendió en las otras
      tiendas (relativo a su formato), escalado al formato de destino. Es lo
      que haría un planificador que confía en el historial de cada producto.
    - **Promedio por formato:** lo que vende un producto promedio en ese
      formato, sin mirar el producto.

    Args:
        df: Datos con features y ``Tipo_Producto``.
        parametros: Parámetros ``asignacion_inventario``.

    Returns:
        Por regla: error medio absoluto (unidades por producto) y unidades mal
        asignadas (las que sobraron más las que faltaron), en total y en % de
        las unidades vendidas.
    """
    y = df[parametros["target"]]
    reglas = ["Repartir igual", "Historial del producto", "Promedio por formato"]
    pred = {regla: np.zeros(len(df)) for regla in reglas}
    for ent, val in _folds(df, parametros):
        y_ent, d_val = y.iloc[ent], df.iloc[val]
        promedio_formato = y_ent.groupby(df["Outlet_Type"].iloc[ent]).mean()
        formato = (
            d_val["Outlet_Type"]
            .map(promedio_formato)
            .fillna(y_ent.mean())
            .to_numpy(dtype=float)
        )
        # Venta relativa al formato, promediada en las OTRAS tiendas del producto.
        relativa = y.iloc[val].to_numpy() / formato
        producto = d_val["Item_Identifier"].to_numpy()
        suma = pd.Series(relativa).groupby(producto).transform("sum").to_numpy()
        n = pd.Series(relativa).groupby(producto).transform("size").to_numpy()
        historial = np.where(n > 1, (suma - relativa) / np.maximum(n - 1, 1), 1.0)

        pred["Repartir igual"][val] = y_ent.mean()
        pred["Historial del producto"][val] = historial * formato
        pred["Promedio por formato"][val] = formato

    vendidas = float(y.sum())
    resultado = {}
    for regla, enviado in pred.items():
        mal_asignadas = float(np.abs(y.to_numpy() - enviado).sum())
        resultado[regla] = {
            "mae": round(mal_asignadas / len(df), 4),
            "unidades_mal_asignadas": round(mal_asignadas),
            "pct_mal_asignadas": round(100 * mal_asignadas / vendidas, 2),
        }
    historial = resultado["Historial del producto"]["unidades_mal_asignadas"]
    formato = resultado["Promedio por formato"]["unidades_mal_asignadas"]
    logger.info("Comparación de reglas de planificación: %s", resultado)
    return {
        "unidades_vendidas": round(vendidas),
        "reglas": resultado,
        "mejora_formato_vs_historial_pct": round(100 * (1 - formato / historial), 2),
    }


def ajustar_modelo_cuantilico(
    df: pd.DataFrame, parametros: dict[str, Any]
) -> dict[str, Any]:
    """Elige los hiperparámetros del Gradient Boosting cuantílico.

    Busca en la grilla de ``grilla_hiperparametros`` la combinación con menor
    pérdida pinball (la métrica propia de un cuantil) en ``cuantil_ajuste``,
    con validación cruzada agrupada por producto. Reporta también la pérdida
    de la regla de cuantil por formato como referencia.

    Returns:
        Cuantil usado, mejores hiperparámetros, su pérdida, la de la regla de
        referencia y la pérdida de cada combinación probada.
    """
    cuantil = parametros["cuantil_ajuste"]
    y = df[parametros["target"]]
    X = _matriz_modelo(df, parametros)
    folds = _folds(df, parametros)

    resultados = []
    for hiperparametros in ParameterGrid(parametros["grilla_hiperparametros"]):
        perdidas = [
            mean_pinball_loss(
                y.iloc[val],
                _modelo_cuantilico(cuantil, hiperparametros, parametros["random_state"])
                .fit(X.iloc[ent], y.iloc[ent])
                .predict(X.iloc[val]),
                alpha=cuantil,
            )
            for ent, val in folds
        ]
        resultados.append(
            {**hiperparametros, "perdida_pinball": round(float(np.mean(perdidas)), 4)}
        )

    referencia = [
        mean_pinball_loss(
            y.iloc[val],
            ReglaCuantilica({tipo: cuantil for tipo in TIPOS})
            .fit(df.iloc[ent], y.iloc[ent])
            .predict(df.iloc[val]),
            alpha=cuantil,
        )
        for ent, val in folds
    ]
    mejor = min(resultados, key=lambda r: r["perdida_pinball"])
    resultado = {
        "cuantil": cuantil,
        "mejores_hiperparametros": {
            k: v for k, v in mejor.items() if k != "perdida_pinball"
        },
        "perdida_mejor": mejor["perdida_pinball"],
        "perdida_cuantil_por_formato": round(float(np.mean(referencia)), 4),
        "resultados": resultados,
    }
    logger.info(
        "Ajuste cuantílico: mejor %s (pinball %s, regla por formato %s)",
        resultado["mejores_hiperparametros"],
        resultado["perdida_mejor"],
        resultado["perdida_cuantil_por_formato"],
    )
    return resultado


def evaluar_escenarios_costo(
    df: pd.DataFrame,
    parametros: dict[str, Any],
    ajuste: dict[str, Any],
) -> pd.DataFrame:
    """Compara el costo de tres reglas de envío en cada escenario de costos.

    Reglas: promedio por formato (línea base), cuantil por formato y tipo de
    producto, y Gradient Boosting cuantílico con los hiperparámetros ajustados.
    Además de ``escenarios_costo`` se evalúa el escenario de la regla
    recomendada (``costos_por_tipo``).

    Returns:
        Una fila por escenario y regla con el costo total y el ahorro frente a
        la línea base.
    """
    y = df[parametros["target"]]
    X = _matriz_modelo(df, parametros)
    folds = _folds(df, parametros)
    escenarios = parametros["escenarios_costo"] + [
        {"nombre": "Según tipo de producto", "por_tipo": parametros["costos_por_tipo"]}
    ]

    filas = []
    for escenario in escenarios:
        costos = _costos_escenario(escenario)
        cuantiles = {tipo: cuantil_critico(*c) for tipo, c in costos.items()}
        faltante = df[TIPO].map({t: c[0] for t, c in costos.items()}).to_numpy()
        sobrante = df[TIPO].map({t: c[1] for t, c in costos.items()}).to_numpy()

        pred = {
            regla: np.zeros(len(df))
            for regla in [
                "Promedio por formato",
                "Cuantil por formato",
                "GBM cuantílico",
            ]
        }
        for ent, val in folds:
            d_ent, d_val, y_ent = df.iloc[ent], df.iloc[val], y.iloc[ent]
            pred["Promedio por formato"][val] = (
                BaselinePorGrupo("Outlet_Type").fit(d_ent, y_ent).predict(d_val)
            )
            pred["Cuantil por formato"][val] = (
                ReglaCuantilica(cuantiles).fit(d_ent, y_ent).predict(d_val)
            )
            for cuantil in set(cuantiles.values()):
                tipos = [t for t, q in cuantiles.items() if q == cuantil]
                filas_tipo = val[d_val[TIPO].isin(tipos).to_numpy()]
                modelo = _modelo_cuantilico(
                    cuantil,
                    ajuste["mejores_hiperparametros"],
                    parametros["random_state"],
                ).fit(X.iloc[ent], y_ent)
                pred["GBM cuantílico"][filas_tipo] = modelo.predict(X.iloc[filas_tipo])

        base = costo_asignacion(y, pred["Promedio por formato"], faltante, sobrante)
        etiqueta = " / ".join(f"{t}: {q:.2f}" for t, q in cuantiles.items())
        for regla, enviado in pred.items():
            costo = costo_asignacion(y, enviado, faltante, sobrante)
            filas.append(
                {
                    "escenario": escenario["nombre"],
                    "cuantiles": etiqueta,
                    "regla": regla,
                    "costo": round(costo, 1),
                    "ahorro_vs_promedio_pct": round(100 * (1 - costo / base), 2),
                }
            )
    resultado = pd.DataFrame(filas)
    logger.info("Costos por escenario:\n%s", resultado.to_string(index=False))
    return resultado


def analizar_sensibilidad(df: pd.DataFrame, parametros: dict[str, Any]) -> pd.DataFrame:
    """Ahorro de la regla de cuantil frente al promedio para cada proporción de costos.

    Como los costos reales de BigMart no están en los datos, en vez de fijar
    uno se recorre ``ratios_sensibilidad`` (costo de faltar / costo de sobrar).

    Returns:
        Por proporción: cuantil óptimo, costo de cada regla y ahorro en %.
    """
    y = df[parametros["target"]]
    folds = _folds(df, parametros)
    filas = []
    for ratio in parametros["ratios_sensibilidad"]:
        cuantil = cuantil_critico(ratio, 1)
        promedio, cuantilico = np.zeros(len(df)), np.zeros(len(df))
        for ent, val in folds:
            d_ent, d_val, y_ent = df.iloc[ent], df.iloc[val], y.iloc[ent]
            promedio[val] = (
                BaselinePorGrupo("Outlet_Type").fit(d_ent, y_ent).predict(d_val)
            )
            cuantilico[val] = (
                ReglaCuantilica({tipo: cuantil for tipo in TIPOS})
                .fit(d_ent, y_ent)
                .predict(d_val)
            )
        costo_promedio = costo_asignacion(y, promedio, ratio, 1)
        costo_cuantil = costo_asignacion(y, cuantilico, ratio, 1)
        filas.append(
            {
                "ratio_faltante_sobrante": ratio,
                "cuantil": round(cuantil, 3),
                "costo_promedio": round(costo_promedio, 1),
                "costo_cuantil": round(costo_cuantil, 1),
                "ahorro_pct": round(100 * (1 - costo_cuantil / costo_promedio), 2),
            }
        )
    return pd.DataFrame(filas)


def construir_regla_asignacion(
    df: pd.DataFrame, parametros: dict[str, Any]
) -> tuple[ReglaCuantilica, pd.DataFrame]:
    """Entrena la regla recomendada con todos los datos y arma la tabla de envíos.

    Returns:
        La regla entrenada (modelo final) y la tabla de unidades a enviar por
        producto según formato y tipo de producto, junto al promedio vendido.
    """
    y = df[parametros["target"]]
    costos = _costos_escenario({"por_tipo": parametros["costos_por_tipo"]})
    cuantiles = {tipo: cuantil_critico(*c) for tipo, c in costos.items()}
    regla = ReglaCuantilica(cuantiles).fit(df, y)

    grupos = [df["Outlet_Type"], df[TIPO]]
    tabla = pd.DataFrame(
        {
            "productos": y.groupby(grupos).size(),
            "unidades_promedio": y.groupby(grupos).mean().round(1),
            "unidades_a_enviar": regla.tabla_.round(1),
        }
    ).reset_index()
    tabla.insert(2, "cuantil", tabla[TIPO].map(cuantiles).round(2))
    return regla, tabla


def graficar_asignacion(sensibilidad: pd.DataFrame, tabla: pd.DataFrame) -> Figure:
    """Gráfica del ahorro según los costos y de las unidades a enviar por formato."""
    figura = Figure(figsize=(12, 5.5), facecolor=COLOR_FONDO, layout="constrained")
    ax_sens, ax_tabla = figura.subplots(1, 2)
    for ax in (ax_sens, ax_tabla):
        estilo_ejes(ax)

    ax_sens.plot(
        sensibilidad["ratio_faltante_sobrante"],
        sensibilidad["ahorro_pct"],
        color=COLOR_TIENDA,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=COLOR_FONDO,
        markeredgewidth=2,
    )
    ax_sens.set_xscale("log")
    ratios = sensibilidad["ratio_faltante_sobrante"]
    marcas = [r for r in ratios if r in (0.2, 0.33, 1, 3, 5)]
    ax_sens.set_xticks(marcas, [f"{r:g}" for r in marcas])
    ax_sens.minorticks_off()
    for _, fila in sensibilidad[ratios.isin([0.33, 1, 3])].iterrows():
        ax_sens.annotate(
            f"{fila['ahorro_pct']:.0f}%",
            (fila["ratio_faltante_sobrante"], fila["ahorro_pct"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=COLOR_TEXTO,
        )
    ax_sens.set_ylim(bottom=min(0, sensibilidad["ahorro_pct"].min()) - 2)
    ax_sens.set_xlabel(
        "Costo de que falte ÷ costo de que sobre (escala log)",
        color=COLOR_TEXTO_SECUNDARIO,
    )
    ax_sens.set_ylabel(
        "Ahorro frente a enviar el promedio (%)", color=COLOR_TEXTO_SECUNDARIO
    )
    titulo(ax_sens, "Cuanto más desiguales los costos, más ahorra")

    formatos = tabla["Outlet_Type"].unique()
    x = np.arange(len(formatos))
    ancho = 0.38
    for desplazamiento, tipo, color in (
        (-ancho / 2, PERECEDERO, COLOR_PRODUCTO),
        (ancho / 2, NO_PERECEDERO, COLOR_TIENDA),
    ):
        datos = tabla[tabla[TIPO] == tipo].set_index("Outlet_Type").loc[formatos]
        cuantil = datos["cuantil"].iloc[0]
        ax_tabla.bar(
            x + desplazamiento,
            datos["unidades_a_enviar"],
            width=ancho - 0.04,
            color=color,
            label=f"{tipo} (cuantil {cuantil:.2f})",
        )
        ax_tabla.scatter(
            x + desplazamiento,
            datos["unidades_promedio"],
            marker="_",
            s=300,
            linewidths=2,
            color=COLOR_TEXTO,
            zorder=3,
        )
        for xi, valor in zip(x + desplazamiento, datos["unidades_a_enviar"]):
            ax_tabla.text(
                xi,
                valor + 0.6,
                f"{valor:.0f}",
                ha="center",
                fontsize=9,
                color=COLOR_TEXTO,
            )
    ax_tabla.scatter(
        [],
        [],
        marker="_",
        s=300,
        linewidths=2,
        color=COLOR_TEXTO,
        label="Promedio vendido",
    )
    ax_tabla.set_xticks(x, [f.replace("Supermarket ", "Super ") for f in formatos])
    ax_tabla.set_ylabel("Unidades por producto", color=COLOR_TEXTO_SECUNDARIO)
    ax_tabla.legend(frameon=False, loc="upper left", labelcolor=COLOR_TEXTO)
    titulo(ax_tabla, "Unidades a enviar por producto")
    return figura
