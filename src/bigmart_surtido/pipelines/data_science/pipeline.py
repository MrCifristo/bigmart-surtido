"""Pipeline de entrenamiento y evaluación."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    dividir_datos,
    entrenar_baseline,
    entrenar_modelo,
    evaluar_modelos,
    explicar_modelo,
    validacion_cruzada,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``data_science``."""
    return Pipeline(
        [
            Node(
                func=dividir_datos,
                inputs=["bigmart_features", "params:data_science"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="dividir_datos",
            ),
            Node(
                func=entrenar_baseline,
                inputs=["X_train", "y_train", "params:data_science.baseline_grupo"],
                outputs="modelo_baseline",
                name="entrenar_baseline",
            ),
            Node(
                func=entrenar_modelo,
                inputs=["X_train", "y_train", "params:data_science"],
                outputs="modelo",
                name="entrenar_modelo",
            ),
            Node(
                func=evaluar_modelos,
                inputs=["modelo_baseline", "modelo", "X_test", "y_test"],
                outputs="metricas",
                name="evaluar_modelos",
            ),
            Node(
                func=validacion_cruzada,
                inputs=["X_train", "y_train", "params:data_science"],
                outputs="metricas_cv",
                name="validacion_cruzada",
            ),
            Node(
                func=explicar_modelo,
                inputs=["modelo", "X_test"],
                outputs=["importancia_shap", "efecto_tienda_shap", "grafica_shap"],
                name="explicar_modelo",
            ),
        ]
    )
