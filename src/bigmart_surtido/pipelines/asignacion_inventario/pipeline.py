"""Pipeline de asignación de inventario con costos asimétricos."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    ajustar_modelo_cuantilico,
    analizar_sensibilidad,
    comparar_reglas_planificacion,
    construir_regla_asignacion,
    evaluar_escenarios_costo,
    graficar_asignacion,
    marcar_perecederos,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``asignacion_inventario``."""
    return Pipeline(
        [
            Node(
                func=marcar_perecederos,
                inputs=[
                    "bigmart_features",
                    "params:asignacion_inventario.categorias_perecederas",
                ],
                outputs="datos_asignacion",
                name="marcar_perecederos",
            ),
            Node(
                func=comparar_reglas_planificacion,
                inputs=["datos_asignacion", "params:asignacion_inventario"],
                outputs="comparacion_reglas",
                name="comparar_reglas_planificacion",
            ),
            Node(
                func=ajustar_modelo_cuantilico,
                inputs=["datos_asignacion", "params:asignacion_inventario"],
                outputs="ajuste_cuantilico",
                name="ajustar_modelo_cuantilico",
            ),
            Node(
                func=evaluar_escenarios_costo,
                inputs=[
                    "datos_asignacion",
                    "params:asignacion_inventario",
                    "ajuste_cuantilico",
                ],
                outputs="costos_por_escenario",
                name="evaluar_escenarios_costo",
            ),
            Node(
                func=analizar_sensibilidad,
                inputs=["datos_asignacion", "params:asignacion_inventario"],
                outputs="sensibilidad_costos",
                name="analizar_sensibilidad",
            ),
            Node(
                func=construir_regla_asignacion,
                inputs=["datos_asignacion", "params:asignacion_inventario"],
                outputs=["modelo_asignacion", "tabla_asignacion"],
                name="construir_regla_asignacion",
            ),
            Node(
                func=graficar_asignacion,
                inputs=["sensibilidad_costos", "tabla_asignacion"],
                outputs="grafica_asignacion",
                name="graficar_asignacion",
            ),
        ]
    )
