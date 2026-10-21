from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


def test_pipelines_registrados():
    bootstrap_project(Path(__file__).resolve().parents[1])
    assert {"data_processing", "data_science", "__default__"} <= set(pipelines)
    nodos = {n.name for n in pipelines["__default__"].nodes}
    assert {"imputar_faltantes", "entrenar_modelo", "evaluar_modelos"} <= nodos
