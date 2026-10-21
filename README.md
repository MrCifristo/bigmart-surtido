# BigMart: surtido y desempeño de tiendas

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Proyecto 1 de *Machine Learning for Business* (UFM, 2026), por George Albadr y Milton Beltrán.

Aplica Machine Learning, con la metodología CRISP-DM, al dataset
[BigMart Sales](https://www.kaggle.com/datasets/yasserh/bigmartsalesdataset) para
apoyar las decisiones de surtido y de gestión de tiendas de una cadena de retail.
La pregunta de negocio y las decisiones de diseño están en
[`docs/decisiones.md`](docs/decisiones.md).

## Estructura

```
conf/base/
  catalog.yml                      # datasets de entrada y salida
  parameters_data_processing.yml   # parámetros de limpieza
  parameters_data_science.yml      # objetivo, features, hiperparámetros
data/01_raw/bigmart.csv            # dataset original (Kaggle)
src/bigmart_surtido/pipelines/
  data_processing/                 # carga, limpieza y features
  data_science/                    # división, línea base, modelo y evaluación
tests/                             # pruebas con pytest
.github/workflows/ci.yml           # Ruff + pruebas + kedro run en cada push/PR
```

### Pipelines

| Pipeline | Nodos | Salidas principales |
|---|---|---|
| `data_processing` | `generar_reporte_calidad` → `normalizar_datos` → `imputar_faltantes` → `crear_features` | `data/08_reporting/reporte_calidad_datos.json`, `data/04_feature/bigmart_features.csv` |
| `data_science` | `dividir_datos` → `entrenar_baseline` / `entrenar_modelo` → `evaluar_modelos` | `data/06_models/*.pkl`, `data/08_reporting/metricas.json` |

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
```

Los parámetros (variable objetivo, features, hiperparámetros, tratamiento de
outliers) se cambian en `conf/base/parameters_*.yml`, sin tocar el código.

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
