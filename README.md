# BigMart: asignación de inventario por producto y tienda

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Proyecto 1 de *Machine Learning for Business* (UFM, 2026), por George Albadr y Milton Beltrán.

Aplica Machine Learning, con la metodología CRISP-DM, al dataset
[BigMart Sales](https://www.kaggle.com/datasets/yasserh/bigmartsalesdataset) para
decidir cuántas unidades de cada producto enviar a cada tienda cuando faltar y
sobrar no cuestan lo mismo (regresión cuantílica con costos asimétricos).
La pregunta de negocio y las decisiones de diseño están en
[`docs/decisiones.md`](docs/decisiones.md).

## Estructura

```
conf/base/
  catalog.yml                      # datasets de entrada y salida
  parameters_data_processing.yml   # parámetros de limpieza
  parameters_data_science.yml      # objetivo, features, hiperparámetros
  parameters_asignacion_inventario.yml  # perecederos, costos, grilla y sensibilidad
data/01_raw/bigmart.csv            # dataset original (Kaggle)
src/bigmart_surtido/pipelines/
  data_processing/                 # carga, limpieza, features y gráfica exploratoria
  data_science/                    # división, línea base, modelo, evaluación, CV y SHAP
  asignacion_inventario/           # unidades a enviar por producto y tienda con costos asimétricos
tests/                             # pruebas con pytest
.github/workflows/ci.yml           # Ruff + pruebas + kedro run en cada push/PR
```

### Pipelines

| Pipeline | Nodos | Salidas principales |
|---|---|---|
| `data_processing` | `generar_reporte_calidad` → `normalizar_datos` → `imputar_faltantes` → `crear_features` | `data/08_reporting/reporte_calidad_datos.json`, `data/04_feature/bigmart_features.csv` |
| `data_science` | `dividir_datos` → `entrenar_baseline` / `entrenar_modelo` → `evaluar_modelos`, `validacion_cruzada`, `explicar_modelo` | `data/06_models/*.pkl`, `data/08_reporting/metricas*.json`, `data/08_reporting/*shap*` |
| `asignacion_inventario` | `marcar_perecederos` → `comparar_reglas_planificacion`, `ajustar_modelo_cuantilico` → `evaluar_escenarios_costo`, `analizar_sensibilidad`, `construir_regla_asignacion` → `graficar_asignacion` | `data/06_models/modelo_asignacion.pkl`, `data/07_model_output/tabla_asignacion.csv`, `data/08_reporting/{comparacion_reglas,ajuste_cuantilico}.json`, `costos_por_escenario.csv`, `sensibilidad_costos.csv`, `asignacion_inventario.png` |

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Ejecución

```bash
kedro run                              # pipeline completo
kedro run --pipeline data_processing   # solo limpieza
kedro run --pipeline data_science      # solo modelado (requiere data_processing)
kedro run --pipeline asignacion_inventario  # solo asignación de inventario (requiere data_processing)
```

Los parámetros (variable objetivo, features, hiperparámetros, tratamiento de
outliers) se cambian en `conf/base/parameters_*.yml`, sin tocar el código.

## Reporte

El reporte final está en [`docs/reporte/reporte.pdf`](docs/reporte/reporte.pdf). Su fuente,
`docs/reporte/reporte.html`, usa las gráficas que genera `kedro run` en
`data/08_reporting/`. Para regenerarlo (requiere [WeasyPrint](https://weasyprint.org)):

```bash
kedro run
weasyprint docs/reporte/reporte.html docs/reporte/reporte.pdf
```

## Calidad y pruebas

```bash
ruff check .
ruff format --check .
pytest
```

El workflow de GitHub Actions corre estos mismos pasos más `kedro run` en cada
push y pull request.

## Fuentes

- Dataset: Yasser H., *BigMart Sales Dataset*, Kaggle.
  https://www.kaggle.com/datasets/yasserh/bigmartsalesdataset
- Kedro: https://docs.kedro.org
- scikit-learn: https://scikit-learn.org
