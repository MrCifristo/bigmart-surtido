"""Pipeline de carga, limpieza y transformación."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    crear_features,
    generar_reporte_calidad,
    imputar_faltantes,
    normalizar_datos,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``data_processing``."""
    return Pipeline(
        [
            Node(
                func=generar_reporte_calidad,
                inputs="bigmart_raw",
                outputs="reporte_calidad_datos",
                name="generar_reporte_calidad",
            ),
            Node(
                func=normalizar_datos,
                inputs=[
                    "bigmart_raw",
                    "params:data_processing.mapa_fat_content",
                    "params:data_processing.mapa_categoria_producto",
                ],
                outputs="bigmart_normalizado",
                name="normalizar_datos",
            ),
            Node(
                func=imputar_faltantes,
                inputs=[
                    "bigmart_normalizado",
                    "params:data_processing.percentil_tope_visibilidad",
                ],
                outputs="bigmart_limpio",
                name="imputar_faltantes",
            ),
            Node(
                func=crear_features,
                inputs=["bigmart_limpio", "params:data_processing.anio_referencia"],
                outputs="bigmart_features",
                name="crear_features",
            ),
        ]
    )
