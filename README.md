# Challenge-3-Data-Science

Proyecto de Machine Learning para predicción de cancelación de clientes (churn).
<img width="865" height="396" alt="imagen decorativa con porcentajes de la tabla lift" src="https://github.com/user-attachments/assets/eada8a74-3690-4af4-ae10-62d43868aa73" />

## Introducción
Este proyecto desarrolla modelos predictivos para estimar el riesgo de cancelación de clientes (churn) en Telecom X. El objetivo es construir herramientas que permitan identificar clientes de alto riesgo y facilitar estrategias de retención.

## Objetivo del Proyecto
Predecir la cancelación de clientes e identificar que factores influyen en la misma.

## Resultado Principal
- El modelo permite priorizar a los clientes según su riesgo de cancelación. El top 20% de mayor riesgo contiene a la mitad de las cancelaciones y presenta una probabilidad de cancelación del 66%, más del doble que el 27% observado en la población general.

## Resumen
- La tasa de cancelación observada es del 27%.
- Se ajustaron 8 modelos predictivos con optimización de hiperparámetros.
- Bajo todas las métricas evaluadas, el mejor modelo fue la regresión logística regularizada.
- El modelo permite identificar el 80% de los clientes que cancelarán, con una precisión del 50%.
- Esto permite construir un grupo de alto riesgo donde la probabilidad de cancelación es aproximadamente el doble que en la población general.

Factores más importantes asociados al riesgo de cancelación:
- Antigüedad del cliente:
  -   Clientes más nuevos presentan mayor riesgo.
- Gasto mensual:
  -   Mayor gasto implica mayor riesgo.
- Tipo de internet:
  -   Fibra óptica tiene mayor riesgo que  DSL y  Sin internet
- Tipo de contrato:
  -   Mensual tiene mayor riesgo que  Anual y este que Bianual

## Tabla Lift
| Grupo |   Cantidad de clientes | Riesgo de Cancelación | Porcentaje Acumulado de Cancelaciones |
|------|----------------------|-----------------------|----------------------------------------|
| A |  282 | 66% | 50% |
| B |  281 | 39% | 80% |
| C |  281 | 17% | 93% |
| D |  281 | 8% | 99% |
| E |  282 | 0% | 100% |

*Nota*: Resultados del modelo aplicado a la base de prueba, simulando su aplicación real.


## Contenidos del análisis

### Introducción
- Breve introducción al problema
- Advertencia: Se asume que los datos pertenecen al mismo periodo de tiempo (mismo mes). Si los datos fueron extraidos durante periodos extendidos en el tiempo, entonces el tipo de análisis debería modificarse (a un análisis de supervivencia).
- Diccionario de variables (se anexa copia en el README)

### Preparación 
Se importan las librerías a utilizar, las bases de datos,crean algunas funciones e eliminan variables con información irrelevante o redundante (basandonos en los resultados del trabajo anterior). En particular se utilizaron las librerías:

- pandas: 2.2.2
- numpy: 2.0.2
- matplotlib: 3.10.0
- seaborn: 0.13.2
- joblib: 1.5.3
- statsmodels: 0.14.6
- scikit-learn: 1.6.1

  
###  Análisis Exploratorio
<img width="1421" height="644" alt="mapa de calor e importancia" src="https://github.com/user-attachments/assets/bc11307c-6a32-42f3-a336-f31fbbe55c36" />

- Se realiza un análisis de correlación para todas la variables.
- Se aplica un bosque aleatorio para obtener una primera estimación de la importancia de las variables.
- Para las variables numéricas se estima la media, desvío estandar y el estadístico d de cohen (que mide la diferencia entre las medias). Se añaden boxplots y un suavizado loess estimando el riesgo de cancelación.
- Para las variables categóricas se estima la proporción de cancelación para cada uno de sus niveles y la diferencia de dichas proporciones. Se provee de un gráfico de barras sobre estas diferencias.

### Preapración de la base
- Se separa la base en entrenamiento y prueba
- Se crea un objeto de validación cruzada para ser reutilizado durante el análisis (k=10 , estratificado por la respuesta)
- Se estandarizan las variables numéricas (se resta la media y se las divide por el desvío estandar)
- Se estandarizan las variables catgóricas (se dividen por el desvío estandar).
- De forma paralela, se crea una base con los datos escalados entre 0 y 1 (se les resta el mínimo y divide por el máximo).

### Valores de referencia
- Se crea un modelo base: un árbol de decisión de un único nivel. Este modelo representa el piso de los modelos, la mínima capacidad predictiva. Si un modelo tiene un rendimiento similar a este entonces probablemente esté sub-ajustando.
- Se crea un modelo tope: un árbol de decisión que se lo deja crecer casi en su totalidad. Este modelo no se evalúa ni en el conjunto de prueba ni por validación cruzada sino en el mismo conjunto usado para su entrenamiento. Debido a su sobre-ajuste nungún modelo será capaz de alcanzar su rendimiento. Los modelos que se acerquen mucho a este modelo estarán aprovechando casi toda la información disponible.

### Modelos
Se estiman 8 modelos diferentes, todos utilizan la corrección de pesos para balancear la base, validación cruzada para estimar su rendimiento de una forma insesgada, datos estandarizados y una busqueda de hiperparametros en grilla. Estos son:
- Árbol de decisión (podado con la regla 1SE)
- Bosque Aleatorio
- Potenciador de Gradiente (basado en Histograma)
- Regresión Logística (con regularización y se añaden Splines)
- Máquina de Vector Soporte ( con kernel gaussiano)
- Redes Neuronales (con una, dos y tres capas)
- Bayes Ingenuo (gaussiano)

### Comparación de modelos
<img width="1417" height="698" alt="matrices de confusion" src="https://github.com/user-attachments/assets/bd3064aa-2932-486e-a9c6-3cd33d4d9dc0" />

Se utiliza la base de prueba, la cual nunca fue vista por los modelos. Se fija un punto de corte tal que la sensibilidad de todos los modelos al 80% (capturan correctamente al 80% de los clientes que cancelan). Lo cual permite comparar el resto de las medidas:
- Precisión: la probabilidad de que un cliente catalogado de alto riesgo  cancele.
- F1: Un promedio entre Sensibilidad y Precisión. Si este número es alto entonces el modelo puede capturar a un gran porcentaje de clientes que cancelarán y además cuando un cliente es clasificado como "alto riesgo" tiene una chance elevada de cancelar.
- Exactitud: El porcentaje de clientes bien clasificado por el modelo. Una medida simple pero que favorece a la clase mayoritaria.
- AUC: A diferencia de las demás medidas esta no se ve afectada por el punte de corte, por lo que se puede pensar como una medida de la capacidad predictiva del modelo más pura.
- Matriz de confusión: Contiene los verdaderos positivos, verdaderos negativos , falsos positivos y falsos negativos de cada modelo.
- Decisión final: tras comparar todos los estadísticos se selecciona el mejor modelo.

### Interpretación del modelo
<img width="1062" height="524" alt="Importancia de las variables" src="https://github.com/user-attachments/assets/f5b62286-6b76-4f48-adb8-5578c6d5aace" />

- Se interpretan las variables más importantes. Incluyendo el alcance de sus efectos (ej: los clientes que contratan fibra óptica tienen un 26% más chance de rechazo que los clientes sin internet).
- Tabla Lift: la misma tabla presentada en este readme.

### Mejoras Futuras
Se mencionan las debilidades y posibles mejoras de este análisis.
En particular se advierte sobre el riesgo de que los datos estén mal interpretados (se asume que todos los datos pertenecen al mismo mes y no a un periodo prolongado en el tiempo.)


## **DICCIONARIO**

- customer_id: número de identificación único de cada cliente
- churn: cliente dejó la empresa (1:Sí, 0:No)
- gender_male: género (1:masculino, 0:femenino)
- senior_citizen_65: cliente mayor a 64 años (1:Sí, 0:No)
- partner: cliente tiene una pareja (1:Sí, 0:No)
- dependents: cliente tiene dependientes (1:Sí, 0:No)
- tenure: meses de contrato del cliente
- phone_service: cliente suscrito al servicio telefónico (1:Sí, 0:No)
- multiple_lines: cliente suscrito a más de una línea telefónica (1:Sí, 0:No)
- internet_service: cliente suscrito a un proveedor de internet (1:Sí, 0:No)
- online_security: cliente suscrito al seguridad en línea (1:Sí, 0:No)
- online_backup: cliente suscrito a respaldo en línea (1:Sí, 0:No)
- device_protection: cliente suscrito a protección del dispositivo (1:Sí, 0:No)
- tech_support: cliente suscrito a soporte técnico (1:Sí, 0:No)
- streaming_tv: cliente suscrito a televisión por cable (1:Sí, 0:No)
- streaming_movies: cliente suscrito a streaming de películas (1:Sí, 0:No)
- contract: tipo de contrato (month_to_month, one_year, two_year)
- paperless_billing: cliente prefiere recibir la factura en línea (1:Sí, 0:No)
- payment_method: forma de pago
- charges_monthly: total de todos los servicios del cliente por mes
- charges_total: total gastado por el cliente
- family: El cliente tiene pareja o dependientes. (1:sí, 0:no)
- tenure_log: logaritmo de la antigüedad del cliente.
- dsl: toma el valor 1 si el servicio de internet es por DSL, 0 en otro caso.
- fiber_optic: toma el valor 1 si el servicio de internet es por fibra optica, 0 en otro caso.
- no_internet: toma el valor 1 si no hay servicio de internet.
- subcription_total: Cuenta la cantidad de servicios a lo que el usuario esta subscrito.
- subcription_no_streaming: Cuenta la cantidad de servicios a lo que el usuario esta subscrito, ignorando los de streaming
- streaming: cliente suscrito a streaming_tv o streaming_movies (1:sí, 0:no)
- month_to_month: toma el valor 1 si el contrato es de ese tipo, 0 en otro caso.
- one_year: toma el valor 1 si el contrato es de ese tipo, 0 en otro caso.
- two_year: toma el valor 1 si el contrato es de ese tipo, 0 en otro caso.
- charges_average : Se suman los gaston mensuales y totales (los gastos acumulados más los gastos de este mes) y se dividen por tenure + 1 (contando el mes).
- charges_average_log : logaritmo de los gastos promedios.
- charges_total_log : logaritmo de los gastos totales.
- electronic_check: toma el valor 1 si el método de pago es de ese tipo, 0 en otro caso.
- mailed_check: toma el valor 1 si el método de pago es de ese tipo, 0 en otro caso.
- bank_transfer_automatic: toma el valor 1 si el método de pago es de ese tipo, 0 en otro caso.
- credit_card_automatic: toma el valor 1 si el método de pago es de ese tipo, 0 en otro caso.
- automatic_payment: : toma el valor 1 si el método de pago es credit_card_automatic o bank_transfer_automatic, 0 en otro caso.


