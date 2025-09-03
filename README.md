# GLOBAL METRIX — Pronóstico de Mortalidad Infantil (t+1)

![Logo](app_streamlit/logo_big.png)


Predicción del **próximo año** de la **mortalidad infantil (por 1.000 nacidos vivos)** a nivel país, con visualización interactiva y diagnóstico de *factores determinantes*.

---

##  Objetivo

- Construir un modelo que **pronostique**  para el año siguiente (t+ 1) en cada país.
- Entregar una **app Streamlit** con:
  - **Mapa global** (coropleta)
  - **Demo país** (KPIs + factores determinantes del modelo)
  - **Datos** (tablas de datos desagradables)

## 🗂️ Estructura del repositorio

```
nombre_proyecto_final_ML
├─ data
│  ├─ raw/
│  ├─ dataset.csv
│  └─ processed/
│     ├─ df_wdi_clean_base.csv            # base EDA/FE consolidada
│     ├─ preds_next_year.csv              # predicciones t+1 por país
│     └─ feature_cols_training.txt        # lista de features usadas en entrenamiento
├─ notebooks/
│  ├─ 01_Fuentes.ipynb
│  ├─ 02_LimpiezaEDA.ipynb
│  └─ 03_Entrenamiento_Evaluacion.ipynb
├─ src/
│  ├─ data_processing.py
│  ├─ training.py
│  └─ evaluation.py
├─ models/
│  └─ trained_model.pkl                   # pipeline scikit-learn serializado (joblib)
├─ app_streamlit/
│  ├─ app.py                              # aplicación (3 páginas)
│  ├─ logo_big.png                        # logo mostrado en sidebar
│  └─ country_names.csv                   # (opcional) mapeo ISO3 → país
├─ docs/
│  ├─ metrics_backtest.csv                # métricas año a año (backtest)
│  ├─ negocio.ppt
│  ├─ ds.ppt
│  └─ memoria.md
└─ README.md
```

---

##  Datos

- **Fuente principal**: Indicadores socioeconómicos y de salud (Word Developpement Index/ World Bank).
- **Target**: `Mortality rate, infant (per 1,000 live births)`.
- **Identificadores**: `country_code` (ISO3), `year` (año).
- **Ejemplos de features**: gasto sanitario (%PIB y per capita), electricidad (% población), camas hospitalarias, alfabetización, inmunización, etc.
- **Procesado**:
  - Tipificación numérica, interpolaciones por país, *feature engineering* (lags, YoY, rolling).
  - Columnas derivadas (p. ej. `health_exp_share`, `log1p_health_exp_pc`, `GDP_pc_imputed`).



##  Pipeline (resumen)

1. **EDA/Limpieza** 
   - Auditoría, outliers básicos, NAs, conversión de tipos.
2. **Feature Engineering**
   - Lags, variaciones interanuales, medias móviles, transformaciones log.
3. **Backtest temporal**
   - Entrenamiento hasta t−1, validación en t (por año), baseline *naive lag-1*.
   - Export de métricas por año .
4. **Entrenamiento final**:
   - *Fit* del pipeline 
   - Export de lista de features .
5. **Inferencia t+1**
   - Predicciones sobre el último año observado .

**Nota**: La app streamlit **no entrena**; solo carga artefactos y muestra resultados.



##  Resultados, Métricas y Visualización (Negocio + Técnica)

### Métricas principales (regresión)
- **RMSE** (raíz del error cuadrático medio): error promedio en la misma unidad que el target (mortalidad por 1.000).
- **MAE** (error absoluto medio): desviación media.
- **R²**: proporción de varianza explicada .
- **Δ vs baseline lag-1**: ganancia/reducción de error respecto a un modelo ingenuo que usa el valor del año anterior.

Las métricas se calculan por año en backtest y se guardan en `docs/metrics_backtest.csv`.
La app muestra además la **mediana de RMSE** y un **intervalo 95% aproximado** para cada país.

### Variables más relevantes (factores demerminantes)  (drivers) e interpretación
- **MI (t−1)**: inercia temporal de la mortalidad infantil.
- **ln(1+Gasto salud pc)**: gasto sanitario per cápita estabilizado → más gasto suele asociarse a menor mortalidad.
- **Electricidad (%)**: acceso a servicios básicos correlaciona con mejores resultados sanitarios.
- **Gasto salud/PIB (%)** y **PIB pc (imput.)**: capacidad del sistema y nivel de desarrollo.
- **Alfabetización / Inmunización / Camas**: capacidades preventivas y de infraestructura sanitaria.

En la app (página *Demo*):
- Gráfico de barras con **importancias** (top-*k* variables, etiquetas ).
- **Tabla comparativa** del país: valor actual por variable, **percentil global** y contribución.

### Visualizaciones recomendadas
- **Coropleta mundial** de `y_pred` o `Δ` vs último año.
- **Barras** de importancias (variables) → lectura rápida para negocio.
- **Tablas** limpias y descargables (CSV).

### Conexión con objetivos de negocio
- Priorización de **países en riesgo** (clasificación por `y_pred`/Δ).
- Asignación **eficiente** de recursos en base a drivers principales.
- **Monitoreo** anual (backtest) para evidenciar mejora continua.



## App Streamlit (3 páginas)

### 1) **Mapa**
- Coropleta mundial con `y_pred` o `Δ vs último año`.
- KPIs: nº países, media mundial (no ponderada).
- *Tooltip*: código y nombre del país.

### 2) **Demo — Pronóstico t+1**
- Selector `ISO3 — País`.
- KPIs: `Predicción t+1`, `Observado t`, `Δ vs año anterior`.
- Intervalo aproximado con la **mediana de RMSE** del backtest.
- **Variables del modelo**:
  - Importancias (árboles) con **etiquetas cortas y en español**.
  - Tabla del país: valor en cada *variable* + percentil global.

### 3) **Datos**
- **Predicciones (t+1)**: filtro por país/código, descarga CSV.
- **Base (último año)**: muestra `country`, `country_code`, `year` primero, descarga CSV.



## Modelo (guía breve)

- **Familia**: árbol de decisión ensemble (p. ej. RandomForest/XGBoost) con *preprocessing* (imputación, escalado si aplica).
- **Métrica**: RMSE en backtest anual vs baseline `lag1`.
- **Objetivo de negocio**: priorizar **tendencia y ranking relativo** sobre países, más que error absoluto puntual.

---

## Limitaciones & próximos pasos

- desfase de publicación de indicadores => pronóstico condicionado a disponibilidad.
- Posibles **sesgos** por cobertura desigual entre países/años.
- Mejoras:
  - *Features* exógenas (conflictos, shocks, clima).
  - Modelos específicos por región/ingresos.
  - What-if locales en la app (sliders por indicador).
  - Intervalos de predicción calibrados (cuantiles, conformal).

