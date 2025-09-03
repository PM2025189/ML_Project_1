# app_streamlit/app.py  3 pages, sibebar + top banner
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Optional: needed to load sklearn pipelines
try:
    import joblib
except Exception:
    joblib = None

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Demo — Mortalidad Infantil (t+1)",
    page_icon="",
    layout="wide"
)

DEBUG = False

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent.parent
PATH_BASE_CSV = ROOT / "data" / "processed" / "df_wdi_clean_base.csv"
PATH_METRICS  = ROOT / "docs" / "metrics_backtest.csv"
PATH_PREDS    = ROOT / "data" / "processed" / "preds_next_year.csv"
PATH_MODEL    = ROOT / "models" / "trained_model.pkl"
PATH_FEATS    = ROOT / "data" / "processed" / "feature_cols_training.txt"

# Your logo — keep it portable (repo-relative by default)
LOGO_PATH = ROOT / "app_streamlit" / "logo_big.png"

# ---------------- Brand colors (match the logo gradient) ----------------
# Couleurs du logo Global Metrix
COLOR_BLUE  = "#1C8BE9"
COLOR_GREEN = "#2FB24A"
COLOR_ORANGE= "#FF7A00"
COLOR_DARK_BLUE = "#0162B4"
COLOR_DARK_ORANGE = "#F37300"
COLOR_LIGHT_GREEN = "#52C151"
COLOR_LIGHT_BLUE = "#1E88E5"

# Palette de couleurs pour les échelles (plus fidèle au dégradé du logo)
LOGO_SEQ_PALETTE = [COLOR_DARK_BLUE, COLOR_GREEN, COLOR_DARK_ORANGE]
LOGO_DIV_PALETTE = [COLOR_DARK_BLUE, COLOR_GREEN, COLOR_DARK_ORANGE]

# Échelles continues pour Plotly
LOGO_SEQ_SCALE = px.colors.make_colorscale(LOGO_SEQ_PALETTE)
LOGO_DIV_SCALE = px.colors.make_colorscale(LOGO_DIV_PALETTE)


# ---------------- Top banner CSS ----------------
st.markdown(f"""
<style>
.top-banner {{
  background: linear-gradient(90deg, {COLOR_ORANGE} 0%, {COLOR_GREEN} 50%, {COLOR_BLUE} 100%);
  padding: 14px 18px; border-radius: 12px; color: #FFFFFF; margin-bottom: 12px;
  display: flex; align-items: center; gap: 12px;
}}
.top-title {{ font-weight: 800; letter-spacing: .08em; font-size: 20px; margin: 0; }}
.top-tag   {{ font-size: 12px; opacity: .95; margin: 0; }}
section[data-testid="stSidebar"] .stRadio > label {{ color: #0F172A; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ---------------- Query Params helpers (new + experimental API) ----------------
def _qp_get(name: str, default=None):
    # New stable API
    try:
        qp = st.query_params
        if name in qp:
            v = qp[name]
            return v[0] if isinstance(v, list) else v
    except Exception:
        pass
    # Old experimental API
    try:
        v = st.experimental_get_query_params().get(name, default)
        return v[0] if isinstance(v, list) else v
    except Exception:
        return default

def _qp_set(**kwargs):
    # New stable API: assign like a dict
    try:
        for k, v in kwargs.items():
            st.query_params[k] = v if v is not None else ""
        return
    except Exception:
        pass
    # Old experimental API
    try:
        st.experimental_set_query_params(**kwargs)
    except Exception:
        pass

# ---------------- Cache-aware loaders (keyed by mtime) ----------------
@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(
        path_str,
        low_memory=False,
        na_values=["..", "NA", "N/A", "", "null", "None"]
        # dtype_backend="pyarrow",  # optional if enabled in env
    )

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Archivo no encontrado: {path}")
        return pd.DataFrame()
    try:
        return _read_csv_cached(path.as_posix(), path.stat().st_mtime)
    except Exception as e:
        st.error(f"Error leyendo {path}: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def _load_model_cached(path_str: str, mtime: float):
    return joblib.load(path_str)

def load_model(path: Path):
    if (joblib is None) or (not path.exists()):
        return None
    try:
        return _load_model_cached(path.as_posix(), path.stat().st_mtime)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo {path}: {e}")
        return None

# ---------------- Load artifacts ----------------
df_base    = load_csv(PATH_BASE_CSV)
metrics_df = load_csv(PATH_METRICS)
pred_df    = load_csv(PATH_PREDS)
model      = load_model(PATH_MODEL)

# feature list (txt one per line)
feature_list: list[str] = []
if PATH_FEATS.exists():
    try:
        feature_list = [ln.strip() for ln in PATH_FEATS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        feature_list = []

# --- Build a country lookup (code -> name) and merge into pred_df & df_base ---
country_lookup = pd.DataFrame()

# 1) Best source: df_base if it has 'country'
if not df_base.empty and {"country_code", "country"}.issubset(df_base.columns):
    country_lookup = (df_base[["country_code", "country"]]
                      .dropna()
                      .drop_duplicates(subset=["country_code"]))

# 2) Fallback: a small CSV you can keep at app_streamlit/country_names.csv
PATH_COUNTRIES = ROOT / "app_streamlit" / "country_names.csv"
if country_lookup.empty and PATH_COUNTRIES.exists():
    try:
        tmp = pd.read_csv(PATH_COUNTRIES)
        tmp = tmp[["country_code", "country"]].dropna().drop_duplicates(subset=["country_code"])
        country_lookup = tmp
    except Exception:
        pass

# 3) Optional fallback via pycountry (only if installed)
if country_lookup.empty:
    try:
        import pycountry
        codes = pred_df["country_code"].dropna().unique().tolist() if "country_code" in pred_df else []
        rows = []
        for c in codes:
            try:
                m = pycountry.countries.get(alpha_3=c)
                if m: rows.append({"country_code": c, "country": m.name})
            except Exception:
                pass
        if rows:
            country_lookup = pd.DataFrame(rows)
    except Exception:
        pass

# Merge into predictions
if not pred_df.empty and not country_lookup.empty and "country_code" in pred_df:
    pred_df = pred_df.merge(country_lookup, on="country_code", how="left")

# Also merge into df_base if missing 'country'
if not df_base.empty and "country" not in df_base.columns and not country_lookup.empty:
    if "country_code" in df_base.columns:
        df_base = df_base.merge(country_lookup, on="country_code", how="left", validate="m:1")

if DEBUG:
    st.caption(f"ROOT={ROOT}")
    st.caption(f"pred_df shape={getattr(pred_df,'shape',None)} | df_base shape={getattr(df_base,'shape',None)}")

# ---------------- Helpers ----------------
def median_backtest_rmse(metrics: pd.DataFrame) -> float | None:
    if metrics.empty: return None
    for col in ("rmse_ml", "rmse_naive"):
        if col in metrics and metrics[col].notna().any():
            return float(np.median(metrics[col].dropna()))
    return None

def pct_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value): return np.nan
    return float((s <= value).mean() * 100.0)

def get_feature_names_from_model_or_file(model_obj, saved_list: list[str]) -> list[str]:
    if model_obj is not None:
        try:
            return list(model_obj.feature_names_in_)
        except Exception:
            pass
    return saved_list or []

# recursively find any estimator that exposes feature_importances_
from sklearn.pipeline import Pipeline
def find_tree_estimator(obj):
    """Recursively find any estimator that exposes feature_importances_."""
    try:
        if hasattr(obj, "feature_importances_"):
            return obj
        if isinstance(obj, Pipeline):
            for _, step in obj.steps:
                found = find_tree_estimator(step)
                if found is not None:
                    return found
        for attr in ("estimator", "regressor", "final_estimator", "model"):
            if hasattr(obj, attr):
                found = find_tree_estimator(getattr(obj, attr))
                if found is not None:
                    return found
        return None
    except Exception:
        return None

def get_last_year(df: pd.DataFrame, col="year") -> int | None:
    if col not in df.columns: return None
    years = pd.to_numeric(df[col], errors="coerce")
    if not years.notna().any(): return None
    return int(years.max())

def require_cols(df: pd.DataFrame, req: set[str], name: str) -> bool:
    missing = [c for c in req if c not in df.columns]
    if missing:
        st.error(f"{name}: faltan columnas requeridas: {missing}")
        return False
    return True

# ---------------- Human-readable feature labels ----------------
def human_label(feat: str) -> str:
    """
    Convert technical feature name into a clear Spanish label for end users.
    - _lag1 -> 'año anterior' (multiple lag1 => 'rezago N años')
    - _roll3 -> 'media móvil (3 años)'
    - _yoy -> 'variación interanual (%)'
    - log1p_ prefix -> '(log)'
    - Map common WDI names to Spanish.
    """
    import re

    # 1) Peel suffixes and count
    base = feat
    lag_count = 0
    has_roll3 = False
    has_yoy = False
    changed = True
    while changed:
        changed = False
        if base.endswith("_lag1"):
            base = base[:-5]
            lag_count += 1
            changed = True
        elif base.endswith("_roll3"):
            base = base[:-6]
            has_roll3 = True
            changed = True
        elif base.endswith("_yoy"):
            base = base[:-4]
            has_yoy = True
            changed = True

    # 2) Human base map
    BASE_HUMAN = {
        "y_t1": "Mortalidad infantil (t+1)",
        "Mortality rate, infant (per 1,000 live births)": "Mortalidad infantil (por 1.000)",
        "Access to electricity (% of population)": "Acceso a la electricidad (%)",
        "Current health expenditure per capita (current US$)": "Gasto sanitario per cápita (US$ corrientes)",
        "Current health expenditure (% of GDP)": "Gasto sanitario / PIB (%)",
        "GDP_pc_imputed": "PIB per cápita (imputado)",
        "health_exp_share": "Gasto sanitario (proporción)",
        "health_exp_pc": "Gasto sanitario per cápita",
        "Literacy rate, adult total (% of people ages 15 and above)": "Tasa de alfabetización adulta (%)",
    }

    # 3) log1p_ handling
    if base.startswith("log1p_"):
        core = base[len("log1p_"):]
        base_label = BASE_HUMAN.get(core, core.replace("_", " "))
        base_label = f"{base_label} (log)"
    else:
        base_label = BASE_HUMAN.get(base, base.replace("_", " "))

    # 4) Special cases
    if base == "y_t1" and lag_count == 1:
        return "Mortalidad infantil (año anterior)"
    if base == "Mortality rate, infant (per 1,000 live births)" and lag_count == 1:
        return "Mortalidad infantil (por 1.000) — año anterior"

    # 5) Compose suffix text
    parts = []
    if has_roll3:
        parts.append("media móvil (3 años)")
    if has_yoy:
        parts.append("variación interanual (%)")
    if lag_count > 0:
        parts.append("año anterior" if lag_count == 1 else f"rezago {lag_count} años")

    if parts:
        return f"{base_label} — {', '.join(parts)}"
    return base_label

# ---------------- Sidebar (logo only) + Navigation ----------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=190)
    page = st.radio("Navegación", ["Mapa", "Pronóstico t+1", "Datos"])

# ---------------- Top banner (appears above all pages) ----------------
st.markdown("""
<div class="top-banner">
  <div>
    <div class="top-title">GLOBAL METRIX</div>
    <p class="top-tag">DATA FOR SOCIAL IMPACT · Forecast t+1</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# PAGE 1 — Mapa
# ======================================================
if page == "Mapa":
    st.title("🌍 Mapa mundial — Mortalidad infantil (t+1)")

    if pred_df.empty:
        st.error("Falta el archivo de predicciones (preds_next_year.csv).")
        st.stop()
    if not require_cols(pred_df, {"country_code", "y_pred"}, "predicciones"):
        st.stop()

    # ✨ Labels simplificados
    color_choice = st.radio("Color par", ["Predicción", "Δ vs último año"], horizontal=True)

    map_df = pred_df.copy()
    color_col, color_label, color_scale = "y_pred", "Predicción por 1.000", LOGO_SEQ_SCALE
    choropleth_kwargs = {
        "locations": "country_code",
        "color": color_col,
        "scope": "world",
        "projection": "natural earth",
        "labels": {color_col: color_label},
        "color_continuous_scale": color_scale,
    }

    # choose delta mode if requested and available
    if color_choice.startswith("Δ") and "target_last" in map_df.columns:
        map_df["delta"] = map_df["y_pred"] - map_df["target_last"]
        color_col, color_label = "delta", "Δ por 1.000"
        choropleth_kwargs["color"] = color_col
        choropleth_kwargs["labels"] = {color_col: color_label}
        choropleth_kwargs["color_continuous_scale"] = LOGO_DIV_SCALE
        choropleth_kwargs["color_continuous_midpoint"] = 0.0  # center at zero

    hover_data = {"country_code": True, "y_pred": ":.2f"}
    if "target_last" in map_df.columns:
        hover_data["target_last"] = ":.2f"
    if color_col not in hover_data:
        hover_data[color_col] = ":.2f"

    if "country" in map_df.columns:
        choropleth_kwargs["hover_name"] = "country"

    fig = px.choropleth(map_df, hover_data=hover_data, **choropleth_kwargs)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)

    k1, k2 = st.columns(2)
    k1.metric("Países", int(map_df["country_code"].nunique()))
    k2.metric("Media mundial (no ponderada)", f"{pd.to_numeric(map_df['y_pred'], errors='coerce').mean():.2f}")

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PAGE 2 — Demo (country drilldown)
# ======================================================
elif page == "Pronóstico t+1":
    st.title("Pronóstico de Mortalidad Infantil (t+1)")

    if pred_df.empty:
        st.error("Faltan predicciones.")
        st.stop()
    if not require_cols(pred_df, {"country_code", "y_pred", "year_pred", "year_input"}, "predicciones"):
        st.stop()

    # Deep-link preset via URL query params (?country=MAR)
    preset = _qp_get("country")
    preset = str(preset).upper() if preset else None

    # Country selector (robust)
    ccode = None
    if "country" in pred_df.columns and pred_df["country"].notna().any():
        opts = (pred_df.loc[pred_df["country_code"].notna(), ["country_code", "country"]]
                .drop_duplicates("country_code"))
        opts["label"] = opts["country_code"] + " — " + opts["country"].fillna("")
        labels = sorted(opts["label"].tolist())
        default_index = 0
        if preset is not None and preset in opts["country_code"].values:
            try:
                preset_label = (opts.assign(_ix=np.arange(len(opts)))
                                   .set_index("country_code")
                                   .loc[preset, "label"])
                default_index = labels.index(preset_label)
            except Exception:
                default_index = 0
        choice = st.selectbox("País", labels, index=(default_index if labels else None))
        ccode = opts.set_index("label").loc[choice, "country_code"] if labels else None
    else:
        ccodes = sorted(pred_df["country_code"].dropna().unique())
        default_index = 0
        if preset is not None and preset in ccodes:
            try:
                default_index = ccodes.index(preset)
            except ValueError:
                default_index = 0
        ccode = st.selectbox("País (ISO-3)", ccodes, index=(default_index if len(ccodes) else None))

    if ccode is None:
        st.warning("No hay países disponibles en las predicciones.")
        st.stop()

    # Write back selection to URL (avoids rerun loops)
    if _qp_get("country") != ccode:
        _qp_set(country=ccode)

    row = pred_df[pred_df["country_code"] == ccode].head(1)
    if row.empty:
        st.warning("No hay predicción para el país seleccionado.")
        st.stop()

    y_pred      = float(row["y_pred"].iloc[0])
    year_pred   = int(row["year_pred"].iloc[0])
    year_input  = int(row["year_input"].iloc[0])
    target_last = float(row["target_last"].iloc[0]) if "target_last" in row.columns and pd.notna(row["target_last"].iloc[0]) else None

    rmse_med = median_backtest_rmse(metrics_df)
    if rmse_med is not None:
        pi_low  = max(0.0, y_pred - 1.96 * rmse_med)
        pi_high = y_pred + 1.96 * rmse_med
    else:
        pi_low = pi_high = None

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Predicción {year_pred} (por 1.000)", f"{y_pred:.2f}")
    if target_last is not None:
        k2.metric(f"Observado {year_input}", f"{target_last:.2f}")
        k3.metric("Δ vs año anterior", f"{(y_pred - target_last):+,.2f}")
    else:
        k2.metric("Observado (último año)", "n/a")
        k3.metric("Δ vs año anterior", "n/a")

    if pi_low is not None:
        st.caption(f"Intervalo 95% aprox.: [{pi_low:.2f}, {pi_high:.2f}]  (mediana RMSE del backtest).")

    st.markdown("---")
    st.subheader("¿Por qué esta predicción? (indicadores globales + valor del país)")

    feat_names = get_feature_names_from_model_or_file(model, feature_list)
    rf_like = find_tree_estimator(model)

    if (rf_like is None) or (not feat_names) or (not hasattr(rf_like, "feature_importances_")):
        st.info("No hay importancias del modelo. Guarde el modelo con 'feature_importances_' y la lista de features.")
    else:
        importances = pd.Series(rf_like.feature_importances_, index=feat_names).sort_values(ascending=False)
        topk = st.slider("Número de indicadores a mostrar", 5, 15, 8)
        top_feats = importances.head(topk).sort_values(ascending=True)  # for horizontal bar

        disp = pd.DataFrame({
            "Indicador": [human_label(f) for f in top_feats.index],
            "Importancia": top_feats.values
        })
        fig_imp = px.bar(disp, x="Importancia", y="Indicador", orientation="h",
                         color_discrete_sequence=[COLOR_GREEN])
        fig_imp.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_imp, use_container_width=True)

        # Country’s values on these indicators (last observed year)
        last_year = get_last_year(df_base, "year")
        if last_year is None:
            st.warning("El campo 'year' no tiene valores válidos en el dataset base.")
        else:
            base_last = (df_base[df_base["year"].astype(str).eq(str(last_year))]
                         .loc[:, ~df_base.columns.duplicated()]
                         .copy())
            X_future = base_last.reindex(columns=feat_names)

            idx = base_last.index[base_last["country_code"] == ccode]
            if len(idx) == 0:
                st.warning("El país seleccionado no está en el último año del dataset base.")
            else:
                x_row = X_future.loc[idx[0]]
                rows = []
                for feat in importances.index[:topk]:
                    val = x_row.get(feat, np.nan)
                    pct = pct_rank(X_future[feat], val) if feat in X_future.columns else np.nan
                    rows.append({
                        "indicador": human_label(feat),
                        "valor_pais": np.round(val, 3),
                        "percentil_en_mundo": np.round(pct, 1),
                        "importancia_global": np.round(importances[feat], 5)
                    })
                comp_df = pd.DataFrame(rows).sort_values("importancia_global", ascending=False)
                st.dataframe(comp_df, use_container_width=True)

# ======================================================
# PAGE 3 — Datos (prioritario + tabs)
# ======================================================
elif page == "Datos":
    st.title(" Datos")

    # ---- Vista prioritaria: Mortalidad infantil 2004 y 2005 (primero) ----
    # This section will only appear if the data can be generated.
    # No warning/info message will be shown if it fails.
    if not df_base.empty and "year" in df_base.columns:
        mi_candidates = [c for c in df_base.columns if "Mortality rate, infant" in c]
        if mi_candidates:
            mi_col = mi_candidates[0]
            df_mi = df_base.loc[:, [c for c in ["country_code", "country", "year", mi_col] if c in df_base.columns]].copy()
            df_mi["year"] = pd.to_numeric(df_mi["year"], errors="coerce").astype("Int64")
            df_mi = df_mi[df_mi["year"].isin([2004, 2005])]
            
            if not df_mi.empty:
                mi_pivot = (df_mi
                            .pivot_table(index=[c for c in ["country_code", "country"] if c in df_mi.columns],
                                         columns="year", values=mi_col, aggfunc="first")
                            .rename(columns={2004: "Mortalidad infantil 2004", 2005: "Mortalidad infantil 2005"})
                            .reset_index())
                
                # Check if pivot is successful and has data before displaying
                if not mi_pivot.empty:
                    st.subheader("Indicadores clave (vista prioritaria)")
                    search0 = st.text_input("Filtrer par pays ou code (contient)", "", key="mi_prior_search")
                    if search0:
                        mask0 = pd.Series(False, index=mi_pivot.index)
                        if "country" in mi_pivot.columns:
                            mask0 |= mi_pivot["country"].astype(str).str.contains(search0, case=False, na=False)
                        if "country_code" in mi_pivot.columns:
                            mask0 |= mi_pivot["country_code"].astype(str).str.contains(search0, case=False, na=False)
                        mi_pivot = mi_pivot[mask0]
                    st.dataframe(mi_pivot, use_container_width=True)
                    st.download_button(
                        "Télécharger CSV (Mortalité infantile 2004–2005)",
                        data=mi_pivot.to_csv(index=False).encode("utf-8"),
                        file_name="mortalidad_infantil_2004_2005.csv",
                        mime="text/csv"
                    )
                    st.markdown("---")

    # ---- Tabs standard ----
    tab1, tab2 = st.tabs(["Prédictions (t+1)", "Base (dernière année)"])

    # ---- Prédictions (t+1) ----
    with tab1:
        DISPLAY_RENAME = {
            "country": "Pays",
            "country_code": "Code ISO-3",
            "year_input": "Année de base",
            "year_pred": "Année de prédiction",
            "y_pred": "Mortalité (par 1.000)",
            "target_last": "Dernière valeur",
            "delta_abs": "Δ absolu",
            "delta_pct": "Δ %",
        }
        DISPLAY_ORDER = ["Pays", "Code ISO-3", "Année de base", "Année de prédiction",
                         "Mortalité (par 1.000)", "Dernière valeur", "Δ absolu", "Δ %"]

        if pred_df.empty:
            st.warning("Il n'y a pas de prédictions.")
        else:
            colA, colB = st.columns(2)
            search = colA.text_input("Filtrer par pays ou code (contient)", "", key="pred_search")
            year_opts = sorted(pred_df["year_pred"].dropna().unique().tolist()) if "year_pred" in pred_df else []
            year_sel = colB.selectbox("Année prédite", year_opts, index=0 if year_opts else None, key="pred_year")

            df_show = pred_df.copy()
            if year_opts:
                df_show = df_show[df_show["year_pred"].eq(year_sel)]

            # compute deltas if available (safe division)
            if "target_last" in df_show.columns:
                df_show["delta_abs"] = df_show["y_pred"] - df_show["target_last"]
                denom = df_show["target_last"].replace(0, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    df_show["delta_pct"] = 100.0 * df_show["delta_abs"] / denom
                df_show["delta_pct"] = df_show["delta_pct"].replace([np.inf, -np.inf], np.nan)

            # filter by search
            if search:
                mask = pd.Series(False, index=df_show.index)
                if "country" in df_show.columns:
                    mask |= df_show["country"].astype(str).str.contains(search, case=False, na=False)
                if "country_code" in df_show.columns:
                    mask |= df_show["country_code"].astype(str).str.contains(search, case=False, na=False)
                df_show = df_show[mask]

            df_show_disp = df_show.rename(columns=DISPLAY_RENAME)
            order = [c for c in DISPLAY_ORDER if c in df_show_disp.columns]
            df_show_disp = df_show_disp[order + [c for c in df_show_disp.columns if c not in order]]

            st.dataframe(df_show_disp, use_container_width=True)

            st.download_button(
                "Télécharger CSV (prédictions)",
                data=df_show_disp.to_csv(index=False).encode("utf-8"),
                file_name=f"predicciones_{year_sel if year_opts else 'all'}.csv",
                mime="text/csv"
            )

    # ---- Base (dernière année) ----
    with tab2:
        if df_base.empty or "year" not in df_base.columns:
            st.warning("Pas de jeu de données de base ou la colonne 'year' est manquante.")
        else:
            last_year = get_last_year(df_base, "year")
            if last_year is None:
                st.warning("Le champ 'year' n'a pas de valeurs valides.")
                st.stop()

            base_last = df_base[df_base["year"].astype(str).eq(str(last_year))].copy()
            st.write(f"Affichage de l'année: **{last_year}**")

            search2 = st.text_input("Filtrer par pays ou code (contient)", "", key="base_search")
            if search2:
                mask2 = pd.Series(False, index=base_last.index)
                if "country" in base_last.columns:
                    mask2 |= base_last["country"].astype(str).str.contains(search2, case=False, na=False)
                if "country_code" in base_last.columns:
                    mask2 |= base_last["country_code"].astype(str).str.contains(search2, case=False, na=False)
                base_last = base_last[mask2]

            # Ensure 'country' exists (merge if needed)
            if "country" not in base_last.columns and not country_lookup.empty and "country_code" in base_last.columns:
                base_last = base_last.merge(country_lookup, on="country_code", how="left", validate="m:1")

            first_cols = [c for c in ["country","country_code","year"] if c in base_last.columns]
            base_disp = base_last[first_cols + [c for c in base_last.columns if c not in first_cols]]

            st.dataframe(base_disp.head(1000), use_container_width=True)
            st.download_button(
                "Télécharger CSV (base dernière année)",
                data=base_disp.to_csv(index=False).encode("utf-8"),
                file_name=f"base_{last_year}.csv",
                mime="text/csv"
            )