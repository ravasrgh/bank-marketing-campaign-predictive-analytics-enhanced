"""
Fungsi utilitas dan konstanta untuk Streamlit App Bank Maju Sejahtera (BMS).
"""
import sys
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

# Palet Warna
COLORS = {
    "navy": "#0A1628",
    "dark_blue": "#1B2A4A",
    "blue": "#2563EB",
    "light_blue": "#3B82F6",
    "accent": "#06B6D4",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
    "card_bg": "rgba(30, 41, 59, 0.7)",
    "border": "rgba(148, 163, 184, 0.15)",
}

# Definisi Fitur (19 fitur final, sesuai categorical_cols/numeric_cols di model_metadata.json)
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome",
]
NUMERIC_FEATURES = [
    "age", "campaign", "previous", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "was_contacted_before",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # 19 total

CATEGORY_OPTIONS = {
    "job": ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
            'retired', 'self-employed', 'services', 'student', 'technician',
            'unemployed', 'unknown'],
    "marital": ['divorced', 'married', 'single', 'unknown'],
    "education": ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                  'illiterate', 'professional.course', 'university.degree', 'unknown'],
    "default": ['no', 'unknown', 'yes'],
    "housing": ['no', 'unknown', 'yes'],
    "loan": ['no', 'unknown', 'yes'],
    "contact": ['cellular', 'telephone'],
    # jan dan feb tidak ada observasinya di data, sehingga dihapus dari pilihan
    "month": ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    "day_of_week": ['mon', 'tue', 'wed', 'thu', 'fri'],
    "poutcome": ['failure', 'nonexistent', 'success'],
}

FEATURE_LABELS = {
    "age": "Usia", "job": "Jenis Pekerjaan", "marital": "Status Pernikahan",
    "education": "Pendidikan", "default": "Punya Kredit Macet?",
    "housing": "Punya KPR?", "loan": "Punya Pinjaman Pribadi?",
    "contact": "Jenis Kontak", "month": "Bulan Kontak Terakhir",
    "day_of_week": "Hari dalam Minggu", "campaign": "Jumlah Kontak di Kampanye Ini",
    "previous": "Jumlah Kontak Sebelumnya", "poutcome": "Hasil Kampanye Sebelumnya",
    "emp.var.rate": "Employment Variation Rate", "cons.price.idx": "Consumer Price Index",
    "cons.conf.idx": "Consumer Confidence Index", "euribor3m": "Euribor 3 Bulan",
    "nr.employed": "Jumlah Pekerja (kuartalan)",
    "was_contacted_before": "Pernah Dihubungi Sebelumnya?",
}

# Fallback default makroekonomi, hasil hitung median dari dataset/bms_dataset_clean.csv
MACRO_DEFAULTS_FALLBACK = {
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.749,
    "cons.conf.idx": -41.8,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
}


class CampaignCapper(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Custom transformer, harus sama persis dengan definisi di notebook (Section E.3)
    supaya model pipeline yang di-pickle bisa dimuat kembali."""

    def __init__(self, quantile=0.95):
        self.quantile = quantile

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.cap_value_ = np.percentile(X_arr, self.quantile * 100)
        self.n_features_in_ = X_arr.shape[1] if X_arr.ndim > 1 else 1
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return np.clip(X_arr, a_min=None, a_max=self.cap_value_)


@st.cache_resource
def load_model():
    """Muat model pipeline dan metadata dengan caching."""
    try:
        # Model disimpan lewat joblib.dump() dari notebook, dan classnya tercatat
        # dengan module __main__. Supaya bisa di-unpickle di luar notebook,
        # CampaignCapper perlu tersedia di sys.modules['__main__'] sebelum load.
        sys.modules["__main__"].CampaignCapper = CampaignCapper

        model = joblib.load("models/final_model.pkl")
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
        return model, metadata
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        return None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None


@st.cache_data
def load_data():
    """Muat dataset bersih hasil notebook remedial."""
    try:
        df = pd.read_csv("dataset/bms_dataset_clean.csv")
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None


def get_macro_defaults(df: pd.DataFrame) -> dict:
    """Hitung nilai default (median) fitur makroekonomi dari dataset yang dimuat,
    dengan fallback ke nilai hasil analisis notebook kalau data tidak tersedia."""
    macro_cols = list(MACRO_DEFAULTS_FALLBACK.keys())
    if df is None or not all(c in df.columns for c in macro_cols):
        return dict(MACRO_DEFAULTS_FALLBACK)
    return {c: float(df[c].median()) for c in macro_cols}


def prepare_input(data: pd.DataFrame) -> pd.DataFrame:
    """Siapkan DataFrame supaya persis punya 19 fitur yang dibutuhkan model."""
    df = data.copy()

    legacy_cols = [c for c in ("pdays", "contact_intensity") if c in df.columns]
    if legacy_cols:
        st.warning(
            f"Kolom {', '.join(legacy_cols)} ditemukan di data yang diupload, "
            "tapi tidak dipakai oleh model final sehingga diabaikan."
        )

    if "was_contacted_before" not in df.columns:
        df["was_contacted_before"] = 0

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib yang hilang: {missing}")

    return df[ALL_FEATURES]


def predict_with_threshold(model, df: pd.DataFrame, threshold: float):
    """Jalankan prediksi dan terapkan threshold kustom."""
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs, preds


def get_custom_css():
    """Kembalikan CSS kustom untuk app."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
.stApp {
    background: linear-gradient(135deg, #0A1628 0%, #1B2A4A 50%, #0F172A 100%);
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid rgba(148,163,184,0.1);
}
[data-testid="stSidebar"] .stRadio > label { color: #E2E8F0; font-weight: 600; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #CBD5E1; padding: 0.5rem 0.75rem; border-radius: 8px; transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(37,99,235,0.15); color: #fff;
}

/* Metric Cards */
.metric-card {
    background: rgba(30,41,59,0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.25s, box-shadow 0.25s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(37,99,235,0.15);
}
.metric-value {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.85rem; color: #CBD5E1; margin-top: 0.3rem; }

/* Prediction Result */
.pred-card {
    border-radius: 20px; padding: 2rem; text-align: center;
    backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.08);
}
.pred-priority {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(6,182,212,0.1));
    border-color: rgba(16,185,129,0.3);
}
.pred-nonpriority {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(245,158,11,0.08));
    border-color: rgba(239,68,68,0.25);
}
.pred-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.pred-label { font-size: 1.6rem; font-weight: 700; }
.pred-prob { font-size: 1.1rem; color: #CBD5E1; margin-top: 0.3rem; }

/* Section Headers */
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #E2E8F0;
    border-left: 4px solid #3B82F6; padding-left: 1rem; margin: 2rem 0 1rem;
}

/* Hero */
.hero-container {
    background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(6,182,212,0.08));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 20px; padding: 2.5rem; margin-bottom: 2rem; text-align: center;
}
.hero-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: #CBD5E1; font-size: 1.05rem; margin-top: 0.5rem; }

/* Streamlit Overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem; background: rgba(15,23,42,0.6); border-radius: 12px; padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; color: #CBD5E1; font-weight: 500; padding: 0.6rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(37,99,235,0.25) !important; color: #fff !important;
}
div[data-testid="stForm"] {
    background: rgba(30,41,59,0.5); border: 1px solid rgba(148,163,184,0.1);
    border-radius: 16px; padding: 1.5rem;
}
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: #fff; border: none; border-radius: 12px;
    padding: 0.7rem 2rem; font-weight: 600; transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3B82F6, #2563EB);
    box-shadow: 0 8px 25px rgba(37,99,235,0.35);
    transform: translateY(-2px);
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #10B981, #059669);
    color: #fff; border: none; border-radius: 12px; font-weight: 600;
}

/* Global text overrides - keep all text light on dark background */
.stApp p, .stApp li, .stApp span,
.stApp label, .stApp div {
    color: #E2E8F0;
}
/* Form labels and help text */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #CBD5E1 !important;
}
/* Markdown body text */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
    color: #E2E8F0 !important;
}
/* Radio / selectbox options */
.stRadio label, .stSelectbox label {
    color: #CBD5E1 !important;
}
/* st.metric label and value */
[data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
}
[data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
}
/* Expander header */
.stExpander summary p {
    color: #CBD5E1 !important;
}
/* Info / success / error box text */
.stAlert p {
    color: #1E293B !important;
}
</style>
"""
