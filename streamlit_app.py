import os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# Utilidades de carga/cache
# =========================
@st.cache_resource(show_spinner=True)
def _latest_art_dir(root="artefactos") -> Path:
    subs = [Path(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    if not subs:
        raise FileNotFoundError(
            f"No hay versiones en '{root}'. Exporta artefactos con el Paso 11 (regresión)."
        )
    subs.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return subs[0]

@st.cache_resource(show_spinner=True)
def load_artifacts(artefacts_root="artefactos"):
    art_dir = _latest_art_dir(artefacts_root)

    with open(art_dir / "input_schema.json", "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    with open(art_dir / "decision_policy.json", "r", encoding="utf-8") as f:
        policy = json.load(f)

    # Cargar pipeline
    winner = policy["winner"]
    pipe = joblib.load(art_dir / f"pipeline_{winner}.joblib")

    # Compatibilidad de schema
    if "columns" in input_schema and "dtypes" in input_schema:
        features = input_schema["columns"]
        dtypes   = input_schema["dtypes"]
    else:  # legado
        features = list(input_schema.keys())
        dtypes   = input_schema

    # Samples (opcional)
    samples_in, samples_out = None, None
    sp_in  = art_dir / "sample_inputs.json"
    sp_out = art_dir / "sample_outputs.json"
    if sp_in.exists():
        samples_in = json.load(open(sp_in, "r", encoding="utf-8"))
    if sp_out.exists():
        samples_out = json.load(open(sp_out, "r", encoding="utf-8"))

    return {
        "art_dir": art_dir,
        "pipe": pipe,
        "features": features,
        "dtypes": dtypes,
        "policy": policy,
        "winner": winner,
        "samples_in": samples_in,
        "samples_out": samples_out,
    }

def coerce_and_align(df: pd.DataFrame, *, features, dtypes) -> pd.DataFrame:
    df = df.copy()
    # crear faltantes
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    # coaccionar tipos
    for c in features:
        t = str(dtypes.get(c, "object")).lower()
        if t.startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t in ("bool", "boolean"):
            df[c] = (df[c].astype("string").str.strip().str.lower()
                     .map({"true": True, "false": False}))
            df[c] = df[c].fillna(False).astype(bool)
        else:
            df[c] = df[c].astype("string").str.strip()
    return df[features]

def predict_reg(pipe, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict(X)

# ==========
# Interfaz UI
# ==========
st.set_page_config(page_title="Regresión — Inferencia (Línea de crédito)", page_icon="📈", layout="wide")
st.title("📈 Predicción de Línea de Crédito — Inferencia con artefactos")
st.caption("Usa los artefactos exportados en el Paso 11 (Regresión).")

# Cargar artefactos
try:
    A = load_artifacts("artefactos")
except Exception as e:
    st.error(f"No se pudieron cargar artefactos: {e}")
    st.stop()

st.success(f"Versión en uso: **{A['art_dir'].name}** — Modelo ganador: **{A['winner']}**")

# Métricas de TEST (policy)
with st.expander("Métricas de TEST del policy", expanded=True):
    m = A["policy"]["test_metrics"]
    cols = st.columns(len(m))
    for i, (k, v) in enumerate(m.items()):
        cols[i].metric(k, f"{v:.4f}")

with st.expander("Esquema de entrada", expanded=False):
    dd = pd.DataFrame({"column": A["features"], "dtype": [A["dtypes"][c] for c in A["features"]]})
    st.dataframe(dd, hide_index=True, use_container_width=True)

# Sidebar: plantilla CSV
template_df = pd.DataFrame({c: [""] for c in A["features"]})
st.sidebar.download_button(
    "Descargar plantilla CSV",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="template_reg_inferencia.csv",
    mime="text/csv"
)

tab1, tab2, tab3 = st.tabs(["🔹 Predicción individual", "📄 Predicción por lote (CSV)", "🧪 Muestras exportadas"])

# -------------------------
# Predicción individual
# -------------------------
with tab1:
    st.subheader("🔹 Predicción individual")
    col_left, col_right = st.columns(2)

    sample_hint = (A["samples_in"][0] if A["samples_in"] else None)
    user_input = {}

    for i, col in enumerate(A["features"]):
        t = str(A["dtypes"].get(col, "object")).lower()
        container = col_left if i % 2 == 0 else col_right

        with container:
            if t.startswith(("int", "float")):
                default = None
                if sample_hint and col in sample_hint:
                    try:
                        default = float(sample_hint[col])
                    except Exception:
                        default = None
                user_input[col] = st.number_input(col, value=default, step=1.0 if t.startswith("int") else 0.01, format="%.4f")
            elif t in ("bool", "boolean"):
                default = False
                if sample_hint and col in sample_hint:
                    default = bool(sample_hint[col])
                user_input[col] = st.checkbox(col, value=default)
            else:
                default = ""
                if sample_hint and col in sample_hint:
                    default = str(sample_hint[col])
                user_input[col] = st.text_input(col, value=default)

    if st.button("Predecir", use_container_width=True):
        X1 = coerce_and_align(pd.DataFrame([user_input]), features=A["features"], dtypes=A["dtypes"])
        yhat = predict_reg(A["pipe"], X1)[0]
        st.metric("Predicción — Línea de crédito", f"{yhat:,.2f}")

# -------------------------
# Predicción por lote
# -------------------------
with tab2:
    st.subheader("📄 Predicción por lote (CSV)")
    up = st.file_uploader("Sube un CSV con las columnas del schema", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception:
            df_in = pd.read_csv(up, encoding="latin-1")

        st.write("Vista previa:")
        st.dataframe(df_in.head(), use_container_width=True)

        Xb = coerce_and_align(df_in, features=A["features"], dtypes=A["dtypes"])
        preds = predict_reg(A["pipe"], Xb)
        out = pd.DataFrame({"y_pred": preds})
        result = pd.concat([df_in.reset_index(drop=True), out], axis=1)

        st.success(f"Predicciones listas (n={len(result)})")
        st.dataframe(result.head(50), use_container_width=True)
        st.download_button(
            "Descargar predicciones CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_regresion.csv",
            mime="text/csv",
            use_container_width=True
        )

# -------------------------
# Muestras exportadas (smoke)
# -------------------------
with tab3:
    st.subheader("🧪 Muestras del Paso 11")
    if A["samples_in"]:
        st.write("Inputs de ejemplo:")
        st.json(A["samples_in"][:3], expanded=False)
        Xs = coerce_and_align(pd.DataFrame(A["samples_in"]), features=A["features"], dtypes=A["dtypes"])
        ys = predict_reg(A["pipe"], Xs)
        st.write("Predicciones (primeros 3):")
        st.json([{"y_pred": float(p)} for p in ys[:3]], expanded=False)
    else:
        st.info("No hay sample_inputs.json en la carpeta de artefactos.")
