# Sistema de Recomendación para Publicidad Digital con Multi-Armed Bandits

## Abstract

Este proyecto desarrolla un sistema de recomendación para publicidad digital orientado a seleccionar el mejor canal de difusión para cada combinación de `Target_Audience` y `Campaign_Goal`, usando como criterio principal el `ROI` y como señales complementarias la `Conversion_Rate` y el `Acquisition_Cost`. A partir del dataset `Social Media Advertising Dataset`, se construyo un flujo completo de analisis, preparación de datos, modelado y evaluacion. El hallazgo central del EDA fue que no existe un canal universalmente ganador: el rendimiento cambia segun la audiencia y el objetivo de campaña. Por ello, en lugar de usar una politica global unica, se implemento un enfoque contextual inspirado en los notebooks `sesion5_02_ucb_thompson_publicidad.ipynb` y `sesion5_03_bandits_personalizados.ipynb`, donde cada contexto tiene su propio agente `Multi-Armed Bandit` con `Thompson Sampling`. Los resultados muestran que el enfoque contextual aprende políticas competitivas, se acerca al ROI histórico óptimo y supera en muchos casos al agente global, además de ofrecer una estrategia razonable para escenarios de `cold start`.

**Keywords:** recommender systems, multi-armed bandit, Thompson Sampling, digital advertising, contextual recommendation, ROI optimization

## Problema de negocio

La pregunta que queremos responder es esta:

> Dada una audiencia objetivo y un objetivo de campaña, que canal publicitario conviene recomendar?

En este proyecto el canal recomendado corresponde a la variable `Channel_Used`, y la decision se toma buscando maximizar principalmente el `ROI`. A diferencia de una recomendacion estandar basada solo en promedios globales, aca el sistema intenta aprender una politica distinta para cada contexto de negocio.

## Dataset utilizado

Archivo fuente:

- `social_media_ads_filtered.csv`

Fuente original del dataset:

- Kaggle: [Social Media Advertising Dataset](https://www.kaggle.com/datasets/jsonk11/social-media-advertising-dataset?resource=download)

En este repositorio se trabaja sobre una version filtrada y preparada localmente a partir de esa fuente.

Variables clave del problema:

- `Target_Audience`
- `Campaign_Goal`
- `Channel_Used`
- `Conversion_Rate`
- `Acquisition_Cost`
- `ROI`

### Muestra de 5 filas

La siguiente tabla muestra una pequena muestra del dataset utilizado:

| Campaign_ID | Target_Audience | Campaign_Goal | Duration | Channel_Used | Conversion_Rate | Acquisition_Cost | ROI | Location | Language | Clicks | Impressions | Engagement_Score | Customer_Segment | Date | Company |
|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|---|---|
| 793457 | Women 45-60 | Market Expansion | 15 Days | Facebook | 0.14 | $500.00 | 5.80 | Los Angeles | English | 506 | 3017 | 1 | Technology | 2022-10-24 | Tech Titans |
| 470283 | Men 35-44 | Increase Sales | 15 Days | Instagram | 0.08 | $500.00 | 5.39 | Los Angeles | English | 506 | 3019 | 9 | Technology | 2022-01-11 | Giga Geeks |
| 854343 | Men 45-60 | Increase Sales | 15 Days | Facebook | 0.13 | $500.00 | 5.37 | Los Angeles | English | 513 | 3040 | 6 | Technology | 2022-03-11 | Silicon Saga |
| 616108 | All Ages | Increase Sales | 15 Days | Twitter | 0.02 | $500.00 | 1.48 | Los Angeles | English | 518 | 3053 | 10 | Technology | 2022-02-26 | Code Crafters |
| 683515 | Men 35-44 | Brand Awareness | 15 Days | Facebook | 0.04 | $500.00 | 6.28 | Los Angeles | English | 521 | 3062 | 9 | Technology | 2022-01-05 | Cyber Circuit |

Del análisis exploratorio se observo que:

- el dataset contiene multiples audiencias y objetivos de campaña con comportamientos distintos
- los canales disponibles incluyen `Facebook`, `Instagram`, `Pinterest` y `Twitter`
- el mejor canal cambia segun la combinacion `audiencia + objetivo`
- por eso no tenia mucho sentido usar una sola regla global para todos los casos

## Enfoque de solución

La solución se organizó en dos niveles:

1. Un baseline simple, basado en medias historicas por canal y por contexto
2. Un modelo `Multi-Armed Bandit` contextual, donde cada combinacion `Target_Audience + Campaign_Goal` tiene su propio agente

La lógica sigue la misma idea pedagogica vista en los notebooks de soporte:

- en `sesion5_02_ucb_thompson_publicidad.ipynb` se presenta el problema clasico de elegir entre varios anuncios con recompensa incierta
- en `sesion5_03_bandits_personalizados.ipynb` se muestra por que conviene tener un agente por usuario o por segmento, en vez de un unico agente para todos

En este proyecto trasladamos esa misma estructura al caso de publicidad digital:

- cada **brazo** del bandit representa un `Channel_Used`
- cada **contexto** representa una combinacion de `Target_Audience` y `Campaign_Goal`
- cada agente aprende, a partir del feedback historico, que canal parece mas prometedor para su contexto

## Técnicas utilizadas

- análisis exploratorio de datos (`EDA`) orientado a distribuciones de ROI, costos, conversiones y canal ganador por contexto
- limpieza y tipado de variables
- construccion de contexto `audiencia + objetivo`
- agregacion de metricas por `contexto x canal`
- baseline de recomendacion global y contextual
- `Multi-Armed Bandit`
- `Thompson Sampling` como estrategia principal de aprendizaje
- comparacion entre agente global y agentes contextuales
- politica de `cold start` con fallback por audiencia, por objetivo y global

## Breve Marco Teórico

### Multi-Armed Bandit

El problema de `Multi-Armed Bandit` modela una situacion donde existen varias opciones posibles y no se conoce de antemano cual dara mejor recompensa. El objetivo no es solo escoger la mejor opcion, sino aprender mientras se toma la decision.

La tensión central del problema es:

- **explotación**: elegir la opcion que hasta ahora parece mejor
- **exploración**: seguir probando otras opciones porque todavia existe incertidumbre

En publicidad digital esto aplica de forma natural:

- si siempre mostramos el canal que historicamente parece mejor, dejamos de aprender
- si exploramos demasiado, sacrificamos rendimiento
- el bandit busca balancear ambas cosas

### UCB y Thompson Sampling

En los notebooks de soporte se revisan dos estrategias importantes:

- `UCB (Upper Confidence Bound)`: combina recompensa estimada mas un bonus de incertidumbre. Explora mas los brazos poco probados y se vuelve mas conservador a medida que acumula informacion.
- `Thompson Sampling`: adopta un enfoque bayesiano. Para cada brazo mantiene una distribucion de creencia y muestrea de ella antes de decidir. Esto hace que la exploracion sea natural: los brazos prometedores tienen mas probabilidad de ser elegidos, pero los inciertos siguen recibiendo oportunidades.

### Por qué Thompson Sampling en este proyecto

Se eligio `Thompson Sampling` porque, alineado con los notebooks del curso:

- es una tecnica muy adecuada cuando la recompensa es incierta
- maneja bien el trade-off entre exploracion y explotacion
- suele converger rapido en la practica
- se adapta bien a una implementacion simple con agentes por contexto

En nuestro codigo cada agente mantiene parametros tipo `alpha` y `beta` para cada canal. Con cada nueva observacion:

- aumenta la evidencia a favor de un canal si la recompensa observada es buena
- o aumenta la evidencia en contra si la recompensa es baja

Luego el agente muestrea de esas distribuciones y recomienda el canal con mayor valor esperado muestral en ese momento.

## Estructura del proyecto

```text
recommendation_system/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- .gitignore
|-- social_media_ads_filtered.csv
|-- recommendation_dataprep.ipynb
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- notebooks/
|   |-- 01_eda_publicidad.ipynb
|   |-- 02_preparacion_datos.ipynb
|   |-- 03_baseline_recomendacion.ipynb
|   |-- 04_bandit_thompson_sampling.ipynb
|   |-- 05_evaluacion_resultados.ipynb
|   `-- 06_presentacion_sistema.ipynb
`-- src/
    |-- config.py
    |-- data/
    |-- modelos/
    |-- evaluacion/
    `-- visualizacion/
```

## Flujo del proyecto

1. cargar y limpiar datos
2. construir el contexto `Target_Audience + Campaign_Goal`
3. analizar distribuciones de ROI, costo y conversion por canal y por contexto
4. crear tablas agregadas por `contexto x canal`
5. construir un baseline de recomendacion
6. entrenar un agente `Thompson Sampling` por contexto
7. entrenar tambien un agente global para comparar
8. evaluar la politica aprendida frente al mejor canal historico
9. generar recomendaciones para contextos conocidos y no conocidos

## Hallazgos principales del análisis

El `EDA` fue clave porque permitio justificar el modelado. Las conclusiones mas importantes fueron estas:

- no existe un canal ganador unico para todas las audiencias y campañas
- el `ROI` cambia de forma visible entre segmentos
- el costo de adquisicion tambien varia por canal y contexto
- algunas combinaciones tienen un ganador claro, pero otras son mas competidas
- esto vuelve razonable pasar de una recomendacion global a una recomendacion contextual

En otras palabras, el analisis inicial ya sugeria que el problema tenia heterogeneidad real, y eso fue exactamente lo que motivo usar un agente por contexto.

## Resultados generales

Tomando como referencia los notebooks de evaluacion y presentacion, los resultados obtenidos muestran que:

- se entrenaron `36` contextos distintos
- la `accuracy_mejor_canal` fue de `0.75`
- el `roi_promedio_bandit` quedo cercano al `roi_promedio_optimo`
- el `regret_aproximado_promedio` se mantuvo bajo
- en muchos contextos el agente contextual supero al agente global

Esto sugiere varias cosas:

- el sistema aprende una politica util, no solo una regla estatica
- aunque no siempre acierta exactamente el canal historicamente mejor, suele quedar cerca en rendimiento
- el enfoque contextual representa mejor la variabilidad del problema que una politica unica global

## Resultados clave

| Metrica | Valor observado | Lectura |
|---|---:|---|
| `n_contextos` | 36 | El sistema aprende una politica separada para 36 combinaciones de audiencia y objetivo |
| `accuracy_mejor_canal` | 0.75 | En 3 de cada 4 contextos coincide con el mejor canal historico |
| `roi_promedio_bandit` | 4.2467 | El rendimiento medio de la politica aprendida es competitivo |
| `roi_promedio_optimo` | 4.3232 | Sirve como referencia historica ideal por contexto |
| `regret_aproximado_promedio` | 0.1083 | La perdida promedio frente al optimo historico es baja |

En terminos practicos, estos numeros sugieren que el bandit contextual no solo aprende una recomendacion razonable, sino que ademas logra acercarse bastante a la mejor decision historica observada.

## Diagrama del pipeline

```text
Dataset Kaggle
   |
   v
Carga y limpieza de datos
   |
   v
Construccion del contexto
(Target_Audience + Campaign_Goal)
   |
   v
EDA y analisis por contexto x canal
   |
   +--> Baseline global y contextual
   |
   v
Entrenamiento de agentes Thompson Sampling
(un agente por contexto)
   |
   +--> Entrenamiento de agente global
   |
   v
Evaluacion de politica aprendida
   |
   +--> Comparacion contextual vs global
   +--> Analisis por brazo
   +--> Regret aproximado
   |
   v
Recomendacion final de canal
   |
   +--> Contexto conocido
   +--> Cold start por audiencia
   +--> Cold start por objetivo
   +--> Cold start global
```

## Estrategia de cold start

Cuando una combinacion nueva no existe en entrenamiento, el sistema usa una estrategia escalonada simple:

1. mejor canal para la misma audiencia
2. mejor canal para el mismo objetivo
3. mejor canal global

Esta parte se mantuvo deliberadamente simple para no sobrecomplicar `src/`, pero aun asi deja una recomendacion razonable para contextos no vistos.

## Instalación y ejecución

### Requisitos previos

- Git
- Python `3.12.x`
- de preferencia un entorno `conda` o `venv`

> Nota: el archivo `pyproject.toml` se dejo configurado para `Python >=3.12,<3.13`. Tome esta version desde la metadata local del kernel de los notebooks del entorno del proyecto. Desde este shell no fue posible consultar `conda` directamente.

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd recommendation_system
```

Si el repositorio padre contiene mas carpetas y este proyecto vive dentro de una carpeta mayor, entra especificamente a:

```bash
cd recommendation_system
```

### 2. Crear y activar un entorno

Con `conda`:

```bash
conda create -n dmc-aiengineer python=3.12
conda activate dmc-aiengineer
```

Con `venv`:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

Opcion simple con `requirements.txt`:

```bash
pip install -r requirements.txt
```

Opcion moderna usando el proyecto:

```bash
pip install -e .
```

### 4. Registrar el kernel para Jupyter

```bash
python -m ipykernel install --user --name dmc-aiengineer --display-name "dmc-aiengineer"
```

### 5. Ejecutar notebooks

```bash
jupyter lab
```

o

```bash
jupyter notebook
```

Orden sugerido para recorrer el proyecto:

1. `notebooks/01_eda_publicidad.ipynb`
2. `notebooks/02_preparacion_datos.ipynb`
3. `notebooks/03_baseline_recomendacion.ipynb`
4. `notebooks/04_bandit_thompson_sampling.ipynb`
5. `notebooks/05_evaluacion_resultados.ipynb`
6. `notebooks/06_presentacion_sistema.ipynb`

### 6. Ejecutar el flujo principal desde código

Desde la carpeta `recommendation_system`, se puede correr el flujo completo asi:

```python
from src.evaluacion.experimentos import ejecutar_flujo_completo

resultado = ejecutar_flujo_completo("social_media_ads_filtered.csv")

print(resultado["metricas_finales"])
print(resultado["recomendacion_ejemplo"])
```

### 7. Ejemplo de recomendación puntual

Si quieres pedir una recomendacion especifica para una audiencia y objetivo, se puede usar directamente el modulo del modelo:

```python
from src.evaluacion.experimentos import ejecutar_flujo_completo
from src.modelos.bandit_thompson import recomendar_canal

resultado = ejecutar_flujo_completo("social_media_ads_filtered.csv")

recomendacion = recomendar_canal(
    agentes=resultado["agentes"],
    tabla_agregada=resultado["tabla_agregada"],
    target_audience="Women 45-60",
    campaign_goal="Market Expansion",
)

print(recomendacion)
```

### 8. Qué devuelve `ejecutar_flujo_completo`?

La funcion central del proyecto devuelve un diccionario con objetos ya listos para analisis y presentacion. Entre las claves mas importantes estan:

- `metricas_finales`
- `politica_aprendida`
- `evaluacion`
- `tabla_politica_brazos`
- `comparacion_contextual_vs_global`
- `recomendacion_ejemplo`
- `cold_start_audiencia_nueva`
- `cold_start_objetivo_nuevo`
- `cold_start_total`

## Cómo reproducir el proyecto

Si quieres reproducir el trabajo de punta a punta, esta es la ruta recomendada:

1. clonar el repositorio y entrar a la carpeta `recommendation_system`
2. crear un entorno con Python `3.12.x`
3. instalar dependencias con `pip install -r requirements.txt` o `pip install -e .`
4. registrar el kernel y abrir `jupyter lab`
5. ejecutar los notebooks en orden
6. comparar la politica aprendida contra el baseline y el agente global
7. revisar la presentacion final y los ejemplos de `cold start`

Si prefieres correr solo el pipeline principal sin pasar por todos los notebooks, entonces basta con ejecutar `ejecutar_flujo_completo("social_media_ads_filtered.csv")` desde la raiz de `recommendation_system`.

## Notebooks principales

- [01_eda_publicidad.ipynb](notebooks/01_eda_publicidad.ipynb): analisis exploratorio del dataset y comparaciones de ROI, costos y mejor canal por contexto
- [02_preparacion_datos.ipynb](notebooks/02_preparacion_datos.ipynb): limpieza, construccion de variables y contexto
- [03_baseline_recomendacion.ipynb](notebooks/03_baseline_recomendacion.ipynb): baseline simple para tener un punto de comparacion
- [04_bandit_thompson_sampling.ipynb](notebooks/04_bandit_thompson_sampling.ipynb): entrenamiento del modelo principal y comparacion contextual vs global
- [05_evaluacion_resultados.ipynb](notebooks/05_evaluacion_resultados.ipynb): metricas, regret, politica por brazos y lectura de resultados
- [06_presentacion_sistema.ipynb](notebooks/06_presentacion_sistema.ipynb): resumen ejecutivo del sistema y ejemplos de recomendacion, incluyendo `cold start`

## Conclusiones generales

Este proyecto deja varias conclusiones claras.

Primero, el problema de recomendacion de canal en publicidad digital no se comporta igual para todos los segmentos, asi que una politica unica global pierde informacion importante. Segundo, el enfoque de `Multi-Armed Bandit` resulta especialmente util cuando se quiere aprender una politica de decision bajo incertidumbre sin necesidad de entrenar desde el inicio un modelo supervisado mucho mas pesado. Tercero, `Thompson Sampling` funciono bien como tecnica principal porque permite una exploracion bayesiana natural y mantiene una implementacion relativamente simple.

En conjunto, la evidencia del proyecto apunta a que la personalizacion por contexto si agrega valor. La recomendacion final no es solo "que canal tuvo mejor promedio", sino "que canal conviene mas para esta audiencia y este objetivo, dado lo que el sistema ha aprendido".

## Trabajo futuro

Para acercar este sistema a un nivel mas cercano al estado del arte, las mejoras mas importantes serian estas:

- migrar de bandits simples a **contextual bandits** con features explicitas de audiencia, campaña, costo y rendimiento historico
- incorporar **embeddings** de audiencias y campañas para capturar similaridad de forma mas rica en escenarios de `cold start`
- usar una arquitectura de **retrieval + ranking**, donde el bandit opere como capa de decision sobre candidatos previamente rankeados
- experimentar con recompensas compuestas y multiobjetivo, por ejemplo combinando `ROI`, `conversion_rate`, costo y estabilidad
- modelar **non-stationarity**, porque en publicidad real el rendimiento de los canales cambia en el tiempo
- agregar evaluacion temporal y simulaciones online mas realistas
- comparar contra enfoques modernos de recomendacion como `factorization machines`, `gradient boosting`, `two-tower models` o `deep ranking`
- incorporar exploracion segura y restricciones de presupuesto por canal
- añadir trazabilidad, tests, dashboards y monitoreo continuo del comportamiento del sistema

## Estado actual

El proyecto actualmente incluye:

- `EDA` y preparación de datos
- baseline simple
- bandit contextual con `Thompson Sampling`
- comparacion contra agente global
- evaluacion de politica aprendida
- recomendacion para contextos vistos y no vistos

Pendientes que quedan como backlog:

- tests automatizados
- reportes automatizados
- dashboards interactivos
- mejora de similaridad para `cold start`
- version contextual mas avanzada con variables adicionales
