# Decisiones del proyecto

Registro de las decisiones importantes y de su justificación. Se actualiza a
medida que avanza el proyecto.

## 1. Problema de negocio (pendiente de aprobación)

| Elemento | Propuesta |
|---|---|
| Contexto | BigMart es una cadena de 10 tiendas de cuatro formatos (Grocery Store y Supermarket Type1, Type2 y Type3). Asignar el surtido y el inventario sin saber cuánto venderá cada producto en cada tienda genera sobreinventario o ventas perdidas. |
| Stakeholder | Gerente de Category Management, con apoyo de Operaciones de tiendas. |
| Pregunta de negocio | ¿Cómo debería BigMart ajustar el surtido de cada tienda para vender más: qué productos agregar, dónde el desempeño está por debajo de lo esperado y qué productos retirar? |
| Objetivo analítico | Regresión de unidades vendidas por producto y tienda. Las tres salidas (agregar, revisar y retirar) se leen de las mismas predicciones. |
| Métrica de éxito | Técnica: MAE, RMSE y R² frente a la línea base. De negocio: monto en Q de las mejores oportunidades y brecha en Q de las tiendas o categorías por debajo de lo esperado. |
| Impacto | Asignación de surtido según la demanda esperada y foco de gestión en las tiendas con brecha. |

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

## 5. Modelado (estado actual)

- **Línea base:** promedio de unidades por `Outlet_Type` (`BaselinePorGrupo`).
- **Modelo inicial:** Random Forest con one-hot de las variables categóricas.
- **Pendiente:** validación cruzada, ajuste de hiperparámetros, SHAP y el
  pipeline de salidas de negocio.
