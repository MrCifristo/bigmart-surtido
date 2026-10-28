# Decisiones del proyecto

Registro de las decisiones importantes y de su justificación. Se actualiza a
medida que avanza el proyecto.

## 1. Problema de negocio

| Elemento | Definición |
|---|---|
| Contexto | BigMart es una cadena de 10 tiendas de cuatro formatos (Grocery Store y Supermarket Type1, Type2 y Type3) con un catálogo de 1,559 productos. Cada producto se surte desde un centro de distribución, y enviar de más o de menos tiene costo: lo que sobra se echa a perder o ocupa bodega, y lo que falta es venta perdida. Esos dos costos no son iguales: en un perecedero pesa más que sobre y en un no perecedero pesa más que falte. |
| Stakeholder | Gerente de Logística / centro de distribución, que decide cuántas unidades de cada producto enviar a cada tienda. |
| Pregunta de negocio | ¿Cuántas unidades de cada producto conviene enviar a cada tienda, considerando que en los perecederos cuesta más que sobre y en los no perecederos cuesta más que falte? |
| Objetivo analítico | Regresión de las unidades vendidas por producto y tienda. Primero, regresión a la media para saber qué información sirve para planificar (historial del producto vs. formato de tienda). Después, **regresión cuantílica**: con costos asimétricos, la cantidad óptima a enviar es un cuantil de la demanda, `costo_faltante / (costo_faltante + costo_sobrante)` (problema del vendedor de periódicos). |
| Métrica de éxito | Técnica: MAE y pérdida pinball con validación cruzada agrupada por producto (los productos evaluados nunca se vieron al entrenar). De negocio: unidades mal asignadas (sobrantes + faltantes) y costo total ponderado, medidos contra las **unidades realmente vendidas**, frente a la línea base "enviar el promedio del formato". |
| Impacto | Planificar por formato en vez de por historial del producto reduce 9.5% las unidades mal asignadas; este resultado no depende de supuestos. Con costos asimétricos, el ahorro adicional de la regla cuantílica depende de la proporción de costos, que el dataset no trae: 0.6% con costos iguales, 18.8% en el escenario ilustrativo 3:1. El análisis de sensibilidad da la regla para cualquier proporción que defina BigMart. |

**Por qué este enfoque.** Una pregunta anterior medía la "brecha" de cada
tienda contra un potencial definido por nosotros (lo que vendería un Type1).
Se descartó porque ese potencial no se observa en los datos: la conclusión
dependía de la referencia elegida y no se podía verificar. En la asignación de
inventario todo se evalúa contra lo que realmente se vendió. Los costos de
faltar y sobrar tampoco están en los datos, por eso no se fija uno: se
presentan escenarios y un análisis de sensibilidad.

**Preguntas sugeridas por el catedrático** (y dónde se responden):

- *¿Qué ventas esperar según visibilidad, tamaño, ubicación y tier?* La
  visibilidad no tiene efecto; el tamaño y el tier no se pueden separar de la
  tienda (secciones 4 y 5). Por eso la regla de envío usa el formato.
- *¿Se puede enfocar en los productos?* El producto no tiene señal propia en
  unidades: planificar por su historial empeora el envío (sección 6.1). La
  decisión por producto que sí se sostiene es su tipo (perecedero o no), que
  define el costo de equivocarse.

## 2. Variable objetivo: unidades en lugar de ventas en Q

`Item_Outlet_Sales` está en dinero (unidades × precio). Su correlación con el
precio (`Item_MRP`) es 0.57, pero la de las **unidades** (`Item_Outlet_Sales / Item_MRP`)
con el precio es 0.01: el precio infla el monto, pero no mueve la demanda.
Modelar unidades separa la demanda del efecto mecánico del precio. El objetivo
es configurable en `parameters_data_science.yml` (`target`).

## 3. Limpieza de datos

| Problema | Hallazgo | Tratamiento |
|---|---|---|
| `Item_Weight` faltante | 1,463 filas, todas de OUT027 y OUT019 | El peso de un producto es constante entre tiendas: se toma el del mismo producto en otra tienda (recupera 1,459). El resto se imputa con la mediana. |
| `Outlet_Size` faltante | 2,410 filas: tiendas OUT010, OUT017 y OUT045 completas | Moda del tamaño entre las tiendas del mismo `Outlet_Type`. |
| `Item_Fat_Content` inconsistente | `LF`, `low fat` y `reg` | Se unifican en `Low Fat` y `Regular`. Los no consumibles (prefijo NC) pasan a `No aplica`. |
| `Item_Visibility` = 0 | 526 filas, imposible para un producto que se vendió | Se trata como faltante, se imputa con la media del producto y se recorta en el percentil 99. |
| Outliers de ventas | 186 por regla IQR | Se conservan: son ventas reales de los formatos grandes (Type3). |
| Duplicados | 0 exactos y 0 de producto-tienda | Se verifica y se eliminan en `normalizar_datos` por si aparecen. |

## 4. Hallazgos del análisis exploratorio

- **El formato de la tienda determina la demanda.** Para el mismo producto
  (mediana entre productos, frente a su promedio en las tiendas Type1), un
  Type3 vende 1.59× las unidades de un Type1, un Type2 vende 0.81× y una
  Grocery 0.13×.
- **El producto casi no influye en las unidades.** La categoría de producto
  solo mueve el promedio entre 14 y 16 unidades.
- **La visibilidad no muestra efecto.** Su correlación negativa con las ventas
  (−0.12) se debe a que las Grocery tienen más visibilidad por producto.
  Comparando el mismo producto en distintas tiendas Type1, la correlación con
  unidades es 0.01.
- **Una regla simple ya es muy competitiva.** "Unidades promedio del formato ×
  precio" explica el 60.5% de la varianza de las ventas en Q, igual que los
  modelos de ML probados.

- **Un producto que vende bien en unas tiendas no vende mejor en otras.**
  Dividiendo las 6 tiendas Type1 en dos grupos, la correlación de unidades de
  cada producto entre grupos es 0.04, y de los productos en el top 20% de un
  grupo solo el 20% repite en el otro (igual que el azar).
- **Las categorías rinden igual en todos los formatos.** De 64 combinaciones
  categoría × formato, solo 2 se desvían con |t| > 2 (se esperarían ~3 por
  azar); el ANOVA dentro de cada formato da p entre 0.44 y 0.86.
- **En ventas en Q, el producto solo importa por su precio.** Las unidades
  son planas entre bandas de precio (16.0 a 16.7 en Type1), así que un
  producto caro factura más por aritmética. El precio de un mismo producto
  casi no varía entre tiendas (desviación de Q1.1), así que no se puede
  estimar elasticidad.
- **El tamaño y el tier no se pueden aislar.** Con 10 tiendas, casi cada
  combinación formato × tamaño × tier es una sola tienda; dentro de Type1 las
  unidades no cambian con el tamaño ni el tier (16.0 a 16.6). Además el
  tamaño de OUT010, OUT017 y OUT045 fue imputado.

Implicación: para planificar envíos, la información útil es el formato de la
tienda. El producto aporta su tipo (perecedero o no), que define el costo de
equivocarse, pero no su historial de ventas.

## 5. Modelo de demanda promedio

- **Línea base:** promedio de unidades por `Outlet_Type` (`BaselinePorGrupo`).
- **Modelo:** Random Forest con one-hot de las variables categóricas.
- **Validación cruzada** (5 folds sobre el conjunto de entrenamiento, nodo
  `validacion_cruzada`, salida `metricas_cv.json`):

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Línea base (promedio por formato) | 5.34 ± 0.11 | 7.00 ± 0.21 | 0.423 ± 0.022 |
| Random Forest | 5.36 ± 0.11 | 7.03 ± 0.22 | 0.417 ± 0.025 |

El Random Forest **no supera a la línea base en ninguno de los 5 folds**. La
mejora de 0.7% en el conjunto de prueba (`metricas.json`) se debe a esa
partición en particular. Esto confirma el hallazgo del EDA: fuera del formato
de tienda, las variables no aportan señal sobre las unidades. Para las
recomendaciones se elige el modelo más simple e interpretable.

### Explicabilidad con SHAP

Nodo `explicar_modelo`: valores SHAP del Random Forest sobre el conjunto de
prueba. Los SHAP de las columnas one-hot se suman para volver a la variable
original. Salidas: `importancia_shap.csv`, `efecto_tienda_shap.csv` y
`shap_importancia.png`.

- **Las características de la tienda explican el 89.6% de la importancia** y
  las del producto el 10.4%. `Outlet_Type` sola explica el 60.6%.
- `Outlet_Age` (20.6%) y `Outlet_Identifier` (8.2%) son en la práctica la
  misma información: hay 10 tiendas y 9 antigüedades distintas, así que la
  antigüedad funciona como identificador de tienda. No hay que leerla como
  "las tiendas más antiguas venden más".
- Entre las variables de producto, la que más pesa es el precio, con 3.9%.
  La visibilidad (2.3%), el tipo de producto (0.8%) y el contenido de grasa
  (0.5%) casi no influyen.
- **Efecto de la tienda** (unidades por producto frente a la predicción
  promedio): OUT027 +11.1; las Type1 entre +0.6 y +1.2; OUT018 −1.5; las
  Grocery −13.1.

El ajuste de hiperparámetros se hace sobre el modelo cuantílico (sección 6.3),
que es el que responde la pregunta.

## 6. Asignación de inventario

Pipeline `asignacion_inventario`. Toda la evaluación usa validación cruzada de
5 folds **agrupada por producto**: cada producto se evalúa con reglas y
modelos que nunca lo vieron, como pasaría con un producto nuevo. Se compara lo
que cada regla habría enviado contra las unidades realmente vendidas.

Perecederos (32% de las filas): Dairy, Fruits and Vegetables, Breads, Meat,
Seafood y Breakfast (configurable en `parameters_asignacion_inventario.yml`).

### 6.1 ¿Con qué información planificar? (`comparacion_reglas.json`)

| Regla (costos iguales) | MAE (u/producto) | Unidades mal asignadas |
|---|---|---|
| Repartir igual entre tiendas | 7.32 | 47.5% |
| Historial del producto en las otras tiendas, escalado al formato | 5.87 | 38.0% |
| **Promedio del formato** | **5.31** | **34.4%** |

Planificar con el historial del producto, que es lo intuitivo, **asigna mal
9.5% más unidades** que usar solo el formato de la tienda.

### 6.2 Costos asimétricos: enviar un cuantil, no el promedio

Si faltar y sobrar cuestan distinto, enviar el promedio no es óptimo. La
cantidad que minimiza el costo esperado es el cuantil
`costo_faltante / (costo_faltante + costo_sobrante)` de la demanda:

- Perecedero, sobrar cuesta 3 veces más → cuantil 0.25 (enviar poco).
- No perecedero, faltar cuesta 3 veces más → cuantil 0.75 (enviar con colchón).

Las proporciones 1:3 y 3:1 son supuestos para ilustrar; no son costos reales
de BigMart. La sección 6.4 muestra la regla para cualquier proporción.

### 6.3 Modelos y ajuste de hiperparámetros

| Regla | Descripción |
|---|---|
| Promedio por formato (línea base) | Envía las unidades promedio del formato. |
| Cuantil por formato | Envía el cuantil de las unidades del formato y tipo de producto. |
| GBM cuantílico | `HistGradientBoostingRegressor` con pérdida cuantílica y todas las variables de tienda y producto. |

Ajuste del GBM (`ajuste_cuantilico.json`): grilla de 16 combinaciones de
`max_depth`, `learning_rate`, `max_iter` y `min_samples_leaf`, con pérdida
pinball en el cuantil 0.75. Mejor combinación: profundidad 3, tasa 0.03, 100
iteraciones, 20 por hoja (pinball 2.228). La regla de cuantil por formato
obtiene 2.217: **el modelo ajustado no supera a la regla simple**.

Costo total por escenario (`costos_por_escenario.csv`):

| Escenario | Cuantil por formato | GBM cuantílico |
|---|---|---|
| Costos iguales | −0.6% | +0.05% |
| Faltar cuesta 3× | −16.5% | −16.1% |
| Sobrar cuesta 3× | −25.2% | −24.8% |
| **Según tipo de producto (recomendado)** | **−18.8%** | −18.5% |

(Cambio de costo frente a enviar el promedio del formato.)

**Modelo elegido: cuantil por formato y tipo de producto.** Empata o supera
al GBM, se explica en una tabla de 8 filas, no necesita reentrenar un modelo
complejo y es fácil de implementar en el sistema de reposición. Se guarda en
`data/06_models/modelo_asignacion.pkl`.

### 6.4 Sensibilidad a los costos (`sensibilidad_costos.csv`)

| Faltar ÷ sobrar | 0.2 | 0.33 | 0.5 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|---|
| Cuantil óptimo | 0.17 | 0.25 | 0.33 | 0.50 | 0.67 | 0.75 | 0.83 |
| Ahorro vs. promedio | 42.9% | 25.6% | 12.5% | 0.6% | 6.2% | 16.5% | 33.0% |

El ahorro crece cuanto más desiguales son los costos. Si BigMart estima sus
costos reales, la tabla da el cuantil a usar sin rehacer el análisis.

### 6.5 Regla recomendada (`tabla_asignacion.csv`)

Unidades a enviar por producto:

| Formato | Perecedero (cuantil 0.25) | No perecedero (cuantil 0.75) | Promedio vendido |
|---|---|---|---|
| Grocery Store | 1.0 | 3.0 | 2.4 |
| Supermarket Type1 | 11.0 | 21.0 | 16.2–16.4 |
| Supermarket Type2 | 9.3 | 17.8 | 13.8–14.1 |
| Supermarket Type3 | 20.9 | 32.1 | 26.1–27.7 |

### 6.6 Limitaciones

- **Ventas ≠ demanda.** Sin datos de inventario ni quiebres de stock, se
  asume que lo vendido refleja la demanda; si hubo quiebres, la demanda real
  es mayor.
- **Costos supuestos.** Las proporciones de costo son ilustrativas; por eso
  se incluye la sensibilidad.
- **Un solo período.** El dataset no tiene fechas: no se modela
  estacionalidad ni tendencia.
- **Formatos con una sola tienda.** Type2 (OUT018) y Type3 (OUT027) son una
  tienda cada uno; su regla refleja a esa tienda.
- **Clasificación de perecederos** por `Item_Type`; es una aproximación
  (p. ej. `Breakfast` mezcla cereales y lácteos).
