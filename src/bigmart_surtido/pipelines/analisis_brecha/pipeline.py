"""Pipeline de brecha entre ventas reales y potenciales por tienda."""

from kedro.pipeline import Node, Pipeline

from .nodes import estimar_potencial, resumir_brecha


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``analisis_brecha``."""
    return Pipeline(
        [
            Node(
                func=estimar_potencial,
                inputs=[
                    "bigmart_features",
                    "params:analisis_brecha",
                    "params:data_science.modelo",
                    "params:data_science.random_state",
                ],
                outputs="ventas_esperadas",
                name="estimar_potencial",
            ),
            Node(
                func=resumir_brecha,
                inputs=["ventas_esperadas", "params:analisis_brecha.niveles_tienda"],
                outputs="brecha_por_tienda",
                name="brecha_por_tienda",
            ),
            Node(
                func=resumir_brecha,
                inputs=["ventas_esperadas", "params:analisis_brecha.niveles_formato"],
                outputs="brecha_por_formato",
                name="brecha_por_formato",
            ),
            Node(
                func=resumir_brecha,
                inputs=[
                    "ventas_esperadas",
                    "params:analisis_brecha.niveles_tienda_categoria",
                ],
                outputs="brecha_tienda_categoria",
                name="brecha_tienda_categoria",
            ),
        ]
    )
