# Decisiones del proyecto

Registro de las decisiones importantes y de su justificación. Se actualiza a
medida que avanza el proyecto.

## 1. Problema de negocio (pendiente de aprobación)

| Elemento | Propuesta |
|---|---|
| Contexto | BigMart es una cadena de 10 tiendas de cuatro formatos (Grocery Store y Supermarket Type1, Type2 y Type3). Con el mismo catálogo, las tiendas venden montos muy distintos, y la dirección necesita saber dónde intervenir y qué formato replicar. |
| Stakeholder | Gerente de Operaciones de tiendas y Dirección Comercial (decisiones de intervención, inversión y expansión), con apoyo de Category Management (surtido). |
| Pregunta de negocio | ¿Qué tiendas y formatos venden por debajo de su potencial, y dónde debería BigMart concentrar su surtido e inversión para aumentar las ventas? |
| Objetivo analítico | Regresión de unidades por producto y tienda para estimar cuánto *debería* vender cada tienda según su surtido. La brecha entre lo real y lo esperado (con predicciones fuera de muestra) mide el desempeño. Con SHAP se cuantifica el efecto del formato, la ubicación y el producto. |
| Métrica de éxito | Técnica: MAE, RMSE y R² con validación cruzada frente a la línea base "promedio por formato". De negocio: brecha en Q identificada por tienda y formato, y estabilidad de esa brecha entre productos. |
| Impacto | Priorizar la intervención en las tiendas con brecha, decidir qué formato replicar en la expansión y asignar el surtido de mayor precio a los formatos de mayor demanda. |

**Cómo se responden las preguntas derivadas:**

- **Agregar productos (A):** el surtido se decide por formato y rango de precio,
  no por producto individual, porque el producto no muestra señal propia (ver la sección 4).
- **Revisar tiendas (B):** es la pregunta principal (brecha real vs. esperado).
- **Retirar productos (C):** que un producto rinda poco en una tienda no
  predice que rinda poco en otra, así que la recomendación es no
  descatalogar por una mala venta puntual.
- **Formato de expansión (D):** el efecto del formato es grande y
  estadísticamente claro, pero Type2 y Type3 son una sola tienda cada uno.
  Va como recomendación, con esa limitación declarada.

**Cifras preliminares** (comparadas con el promedio de unidades de los Supermarket Type1, 16.3 por producto):

| Tienda | Formato | Brecha | Brecha en Q |
|---|---|---|---|
| OUT027 | Type3 | +61.7% | +1,318,103 |
| OUT035 | Type1 | +4.3% | +93,260 |
| OUT045 | Type1 | −4.8% | −102,825 |
| OUT018 | Type2 | −13.8% | −296,469 |
| OUT010 / OUT019 | Grocery | −85% | −1.09 M / −1.03 M |

- OUT018 (Type2) vende 2.6 unidades menos que el promedio Type1 **en los
  mismos 924 productos** (t ≈ −10.6).
- OUT027 (Type3) es 1 de 10 tiendas y genera el 18.6% de las ventas.
- La probabilidad de que un producto en el 25% inferior de una tienda Type1
  también lo esté en otra es de 0.245, prácticamente igual al azar (0.25).

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

- **El formato de la tienda determina la demanda.** Para el mismo producto, un
  Type3 vende 1.66× las unidades de un Type1, un Type2 vende 0.81× y una
  Grocery 0.13×.
- **El producto casi no influye en las unidades.** Las unidades del mismo
  producto en dos tiendas Type1 tienen una correlación de 0.01, y la
  categoría de producto solo mueve el promedio entre 14 y 16 unidades.
- **La visibilidad no muestra efecto.** Su correlación negativa con las ventas
  (−0.13) se debe a que las Grocery tienen más visibilidad por producto. Dentro
  de los Supermarket Type1 la correlación con unidades es −0.01.
- **Una regla simple ya es muy competitiva.** "Unidades promedio del formato ×
  precio" explica el 60.5% de la varianza de las ventas en Q, igual que los
  modelos de ML probados.

Implicación: las decisiones por formato de tienda tienen respaldo en los
datos. Las recomendaciones por producto individual hay que plantearlas con
cautela y declararlas como limitación.

## 5. Modelado

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

- **Pendiente:** ajuste de hiperparámetros y SHAP.

## 6. Brecha entre ventas reales y potenciales

**Definición.** El potencial de una tienda es lo que vendería un Supermarket
Type1 (el formato estándar, 6 de las 10 tiendas) con su mismo surtido. Se
estima con un modelo entrenado solo con atributos del producto (precio, peso,
tipo, contenido de grasa y categoría) y solo con tiendas Type1. Cada tienda
Type1 se compara contra las **otras cinco**, nunca contra sí misma, para que
la predicción sea fuera de muestra. La visibilidad se excluye porque la decide
cada tienda.

`brecha = unidades reales − unidades esperadas`; en Q se multiplica por el
precio. El estadístico t mide si la brecha es consistente entre productos
(|t| > 2) o si es ruido.

Pipeline `analisis_brecha`: `estimar_potencial` →
`brecha_por_tienda` / `brecha_por_formato` / `brecha_tienda_categoria`.

| Tienda | Formato | Brecha | Brecha en Q | % productos bajo lo esperado | t |
|---|---|---|---|---|---|
| OUT027 | Type3 | +62.9% | +1,309,624 | 13% | 33.1 |
| OUT035 | Type1 | +4.9% | +108,253 | 48% | 3.3 |
| OUT017 | Type1 | +2.4% | +59,675 | 52% | 1.6 |
| OUT049 | Type1 | +2.1% | +51,811 | 52% | 1.5 |
| OUT013 | Type1 | −2.6% | −24,933 | 55% | −1.8 |
| OUT046 | Type1 | −1.2% | −61,213 | 55% | −0.8 |
| OUT045 | Type1 | −5.5% | −134,149 | 59% | −4.0 |
| OUT018 | Type2 | −15.1% | −302,579 | 69% | −11.6 |
| OUT019 | Grocery | −85.1% | −1,033,673 | 100% | −206 |
| OUT010 | Grocery | −85.4% | −1,092,165 | 100% | −216 |

**Lectura:**

- **OUT018 (Type2)** es la tienda con brecha más clara frente a un formato
  comparable: vende 15% menos que un Type1 con el mismo surtido, y la brecha
  aparece en 13 de sus 16 categorías con |t| > 2. Es un problema de la
  tienda, no de una categoría.
- **Dentro de Type1**, OUT045 (−5.5%) y OUT035 (+4.9%) son las únicas con
  brecha consistente (|t| > 3). Las demás están dentro del ruido.
- **Las Grocery** no son comparables con un supermercado: su −85% refleja el
  formato y no un mal desempeño. Sirve para dimensionar cuánto valdría
  convertirlas.
- **Tienda × categoría:** de las 96 celdas Type1, 15 tienen |t| > 2, frente
  a unas 5 que se esperarían por azar. Hay alguna señal, pero con tantas
  comparaciones las celdas individuales hay que tomarlas como hipótesis a
  revisar, no como conclusiones.
