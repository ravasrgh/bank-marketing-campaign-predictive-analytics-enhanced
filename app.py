"""
Bank Maju Sejahtera (BMS) - Dashboard Prediksi Kampanye Deposito Berjangka
============================================================================
Aplikasi Streamlit untuk Tim Telemarketing dan CMO BMS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_model, load_data, prepare_input, predict_with_threshold,
    get_custom_css, get_macro_defaults, CATEGORY_OPTIONS, FEATURE_LABELS,
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, COLORS,
)

# ══════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BMS Bank - Prediksi Kampanye",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Muat resource
model, metadata = load_model()
df_raw = load_data()

# Angka dampak bisnis final, harus identik dengan Section F notebook remedial.
# profit_with_model, calls_with_model, dan recall dibaca dari metadata (ada di
# deployment_threshold_metrics); baseline_profit dan baseline_calls tidak ada di
# metadata sehingga jadi konstanta di sini, bersumber dari Section F.1 notebook.
BASELINE_PROFIT_RP = 52_200_000
BASELINE_CALLS = 8_236

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## BMS Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["Konteks Proyek", "Dashboard Interaktif", "Prediksi Nasabah"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if metadata:
        model_label = f"{metadata.get('model_name', 'Model')} + {metadata.get('resampler_name', '')}"
    else:
        model_label = "Model"
    st.markdown(
        f"<div style='text-align:center;color:#94A3B8;font-size:0.78rem;'>"
        f"{model_label}<br>Bank Maju Sejahtera (BMS)</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  HALAMAN 1 - KONTEKS PROYEK
# ══════════════════════════════════════════════════════════════
def page_context():
    st.markdown(
        '<div class="hero-container">'
        '<div class="hero-title">Prediksi Kampanye Deposito Berjangka</div>'
        '<div class="hero-subtitle">Alat Bantu Keputusan untuk Tim Telemarketing Bank Maju Sejahtera (BMS)</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if metadata:
        deploy = metadata["deployment_threshold_metrics"]
        profit_with_model = deploy["profit_rp"]
        calls_with_model = deploy["calls"]
        recall_pct = deploy["recall"] * 100
        profit_increase_pct = (profit_with_model / BASELINE_PROFIT_RP - 1) * 100
        calls_reduction_pct = (1 - calls_with_model / BASELINE_CALLS) * 100

        cols = st.columns(4)
        kpis = [
            (f"Rp {profit_with_model:,.0f}".replace(",", "."), "Net Profit dengan Model", "1.4rem"),
            (f"+{profit_increase_pct:.1f}%", "Kenaikan Profit vs Baseline", "2rem"),
            (f"-{calls_reduction_pct:.1f}%", "Reduksi Volume Panggilan", "2rem"),
            (f"{recall_pct:.1f}%", "Nasabah Potensial Tertangkap", "2rem"),
        ]
        for col, (val, lbl, font_size) in zip(cols, kpis):
            col.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:{font_size};">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Tujuan Bisnis</div>', unsafe_allow_html=True)
        st.markdown(
            """
            Bank Maju Sejahtera (BMS) menjalankan kampanye telemarketing untuk
            menjual produk **deposito berjangka**. Pendekatan lama, menghubungi
            nasabah secara acak, menghabiskan waktu agen dan sering melewatkan
            nasabah dengan potensi konversi tinggi.

            Aplikasi ini memakai model **machine learning (LightGBM)** untuk
            menghitung probabilitas setiap nasabah akan berlangganan, sehingga
            tim bisa **memprioritaskan panggilan** dan memaksimalkan konversi
            sambil menekan biaya.
            """
        )
        st.markdown('<div class="section-header">Siapa yang Memakai</div>', unsafe_allow_html=True)
        st.markdown(
            """
            | Peran | Cara Memakai App Ini |
            |---|---|
            | **Tim Telemarketing** | Pengguna utama; menghubungi nasabah dengan skor di atas threshold lebih dulu |
            | **CMO (Chief Marketing Officer)** | Memantau hasil agregat dan dampak bisnis dari dashboard |
            | **Analis Marketing** | Menelusuri pendorong konversi lewat dashboard interaktif |
            """
        )

    with c2:
        st.markdown('<div class="section-header">Kapan dan Bagaimana Dipakai</div>', unsafe_allow_html=True)
        st.markdown(
            """
            Dipakai di **awal siklus kampanye**, untuk menyusun daftar prioritas
            kontak harian atau mingguan dari database nasabah.

            1. **Upload** daftar nasabah (CSV) atau isi profil satu nasabah secara manual.
            2. Model menghitung **probabilitas berlangganan** (0-100%).
            3. Nasabah dengan skor di atas threshold ditandai **Prioritas**
               (hubungi lebih dulu); yang di bawahnya **Bukan Prioritas**
               (lewati atau turunkan prioritas).
            """
        )
        if metadata:
            st.markdown(
                f'<div style="background:rgba(37,99,235,0.15);border:1px solid rgba(59,130,246,0.35);'
                f'border-radius:12px;padding:1rem 1.25rem;margin-top:0.75rem;">'
                f'<p style="color:#E2E8F0;margin:0;"><strong>Threshold deployment:</strong> '
                f'{metadata["threshold_deployment"]:.2f}<br>'
                f'<strong>Threshold profit-optimal:</strong> {metadata["threshold_profit_optimal"]:.2f}<br>'
                f'<strong>Threshold F2-optimal:</strong> {metadata["threshold_f2_optimal"]:.2f}<br>'
                f'Threshold bisa digeser dalam koridor sehat 0,40-0,49 sesuai kapasitas '
                f'call center saat itu, tanpa perlu melatih ulang model.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Kerangka Biaya Kesalahan</div>', unsafe_allow_html=True)
        if metadata:
            cf = metadata["cost_framework"]
            st.markdown(
                f"""
                | Kesalahan | Biaya | Risiko |
                |---|---|---|
                | **False Negative** - lewatkan nasabah yang mau berlangganan | Rp {abs(cf['fn_rp']):,} margin hilang | Kritis |
                | **False Positive** - hubungi nasabah yang tidak tertarik | Rp {abs(cf['fp_rp']):,} biaya panggilan sia-sia | Sedang |

                > Rasio biaya FN dibanding FP adalah **10 banding 1**, sehingga meminimalkan False Negative jadi prioritas utama.
                """
            )


# ══════════════════════════════════════════════════════════════
#  HALAMAN 2 - DASHBOARD INTERAKTIF
# ══════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#CBD5E1"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def page_dashboard():
    st.markdown(
        '<div class="hero-container">'
        '<div class="hero-title">Dashboard Interaktif</div>'
        '<div class="hero-subtitle">Telusuri pendorong konversi di dataset nasabah BMS</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if df_raw is None:
        st.error("Dataset tidak tersedia.")
        return

    df = df_raw.copy()
    df["target"] = (df["y"] == "yes").astype(int)
    df["Berlangganan"] = df["target"].map({1: "Ya", 0: "Tidak"})

    # Baris 1: KPI Ringkasan
    total = len(df)
    subs = df["target"].sum()
    rate = subs / total * 100
    k1, k2, k3, k4 = st.columns(4)
    for col, (v, l) in zip(
        [k1, k2, k3, k4],
        [
            (f"{total:,}", "Total Nasabah"),
            (f"{subs:,}", "Berlangganan"),
            (f"{total - subs:,}", "Tidak Berlangganan"),
            (f"{rate:.1f}%", "Tingkat Konversi"),
        ],
    ):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{v}</div>'
            f'<div class="metric-label">{l}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Chart 1: Tingkat Konversi per Pekerjaan
    st.markdown('<div class="section-header">Tingkat Konversi per Jenis Pekerjaan</div>', unsafe_allow_html=True)
    job_stats = (
        df.groupby("job")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "conversion_rate", "count": "n_customers"})
        .sort_values("conversion_rate", ascending=True)
    )
    job_stats["conversion_pct"] = job_stats["conversion_rate"] * 100

    fig_job = px.bar(
        job_stats, y="job", x="conversion_pct",
        orientation="h",
        color="conversion_pct",
        color_continuous_scale=["#1E3A5F", "#2563EB", "#06B6D4", "#10B981"],
        labels={"conversion_pct": "Tingkat Konversi (%)", "job": "Jenis Pekerjaan"},
        title="",
        hover_data={"n_customers": True},
    )
    fig_job.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=420)
    fig_job.update_traces(
        hovertemplate="<b>%{y}</b><br>Rate: %{x:.1f}%<br>Nasabah: %{customdata[0]:,}<extra></extra>"
    )
    st.plotly_chart(fig_job, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Distribusi Usia vs Konversi</div>', unsafe_allow_html=True)
        fig_age = px.histogram(
            df, x="age", color="Berlangganan", barmode="overlay",
            nbins=40, opacity=0.75,
            color_discrete_map={"Ya": "#10B981", "Tidak": "#3B82F6"},
            labels={"age": "Usia Nasabah", "count": "Jumlah"},
        )
        age_layout = {**PLOT_LAYOUT, "legend": {**PLOT_LAYOUT.get("legend", {}), "x": 0.75, "y": 0.95}}
        fig_age.update_layout(**age_layout, height=400, bargap=0.05)
        st.plotly_chart(fig_age, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Jenis Kontak dan Konversi</div>', unsafe_allow_html=True)
        contact_stats = (
            df.groupby("contact")["target"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "rate", "count": "n"})
        )
        contact_stats["pct"] = contact_stats["rate"] * 100
        fig_contact = px.bar(
            contact_stats, x="contact", y="pct",
            color="contact",
            color_discrete_map={"cellular": "#10B981", "telephone": "#3B82F6"},
            text=contact_stats["pct"].apply(lambda x: f"{x:.1f}%"),
            labels={"pct": "Tingkat Konversi (%)", "contact": "Jenis Kontak"},
        )
        fig_contact.update_traces(textposition="outside")
        fig_contact.update_layout(**PLOT_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig_contact, use_container_width=True)

    st.markdown('<div class="section-header">Hasil Kampanye Sebelumnya vs Konversi</div>', unsafe_allow_html=True)
    pout_stats = (
        df.groupby("poutcome")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "n"})
    )
    pout_stats["pct"] = pout_stats["rate"] * 100
    fig_pout = px.bar(
        pout_stats, x="poutcome", y="pct",
        color="pct",
        color_continuous_scale=["#1E3A5F", "#2563EB", "#10B981"],
        text=pout_stats["pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"pct": "Tingkat Konversi (%)", "poutcome": "Hasil Kampanye Sebelumnya"},
    )
    fig_pout.update_traces(textposition="outside")
    fig_pout.update_layout(**PLOT_LAYOUT, height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_pout, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Euribor 3 Bulan vs Berlangganan</div>', unsafe_allow_html=True)
        fig_eur = px.box(
            df, x="Berlangganan", y="euribor3m",
            color="Berlangganan",
            color_discrete_map={"Ya": "#10B981", "Tidak": "#3B82F6"},
            labels={"euribor3m": "Euribor 3 Bulan"},
        )
        fig_eur.update_layout(**PLOT_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig_eur, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Tingkat Konversi per Bulan</div>', unsafe_allow_html=True)
        month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month_stats = (
            df.groupby("month")["target"].agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "rate", "count": "n"})
        )
        month_stats["month"] = pd.Categorical(month_stats["month"], categories=month_order, ordered=True)
        month_stats = month_stats.sort_values("month")
        month_stats["pct"] = month_stats["rate"] * 100

        fig_month = px.bar(
            month_stats, x="month", y="pct",
            color="pct",
            color_continuous_scale=["#1E3A5F", "#2563EB", "#06B6D4", "#10B981"],
            text=month_stats["pct"].apply(lambda x: f"{x:.0f}%"),
            labels={"pct": "Tingkat Konversi (%)", "month": "Bulan"},
        )
        fig_month.update_traces(textposition="outside")
        fig_month.update_layout(**PLOT_LAYOUT, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  HALAMAN 3 - PREDIKSI NASABAH
# ══════════════════════════════════════════════════════════════
def page_predictor():
    st.markdown(
        '<div class="hero-container">'
        '<div class="hero-title">Prediksi Nasabah</div>'
        '<div class="hero-subtitle">Klasifikasikan nasabah sebagai Prioritas atau Bukan Prioritas untuk kampanye</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if model is None or metadata is None:
        st.error("Model tidak berhasil dimuat. Periksa file .pkl dan .json di folder models/.")
        return

    threshold = metadata["threshold_deployment"]
    macro_defaults = get_macro_defaults(df_raw)

    tab_single, tab_batch = st.tabs(["Prediksi Satu Nasabah", "Prediksi Batch"])

    # ────────────────────────────────────────────
    # TAB 1 - Prediksi Satu Nasabah
    # ────────────────────────────────────────────
    with tab_single:
        st.markdown('<div class="section-header">Profil Nasabah</div>', unsafe_allow_html=True)

        with st.form("single_pred_form"):
            st.markdown("##### Demografi")
            d1, d2, d3, d4 = st.columns(4)
            age = d1.number_input("Usia", 17, 100, 35, key="age_input")
            job = d2.selectbox("Jenis Pekerjaan", CATEGORY_OPTIONS["job"], key="job_input")
            marital = d3.selectbox("Status Pernikahan", CATEGORY_OPTIONS["marital"], key="marital_input")
            education = d4.selectbox("Pendidikan", CATEGORY_OPTIONS["education"], index=6, key="edu_input")

            st.markdown("##### Status Finansial")
            f1, f2, f3 = st.columns(3)
            default = f1.selectbox("Punya Kredit Macet?", CATEGORY_OPTIONS["default"], key="def_input")
            housing = f2.selectbox("Punya KPR?", CATEGORY_OPTIONS["housing"], key="house_input")
            loan = f3.selectbox("Punya Pinjaman Pribadi?", CATEGORY_OPTIONS["loan"], key="loan_input")

            st.markdown("##### Kampanye Saat Ini")
            c1, c2, c3, c4 = st.columns(4)
            contact = c1.selectbox("Jenis Kontak", CATEGORY_OPTIONS["contact"], key="contact_input")
            month = c2.selectbox("Bulan Kontak", CATEGORY_OPTIONS["month"], index=2, key="month_input")
            day_of_week = c3.selectbox("Hari dalam Minggu", CATEGORY_OPTIONS["day_of_week"], key="dow_input")
            campaign = c4.number_input("Jumlah Kontak Kampanye Ini", 1, 60, 2, key="campaign_input")

            st.markdown("##### Riwayat dan Indikator Makroekonomi")
            m1, m2, m3, m4 = st.columns(4)
            poutcome = m1.selectbox("Hasil Kampanye Sebelumnya", CATEGORY_OPTIONS["poutcome"], index=1, key="pout_input")
            previous = m2.number_input("Jumlah Kontak Sebelumnya", 0, 50, 0, key="prev_input")
            was_contacted = m3.selectbox(
                "Pernah Dihubungi Sebelumnya?", [0, 1],
                format_func=lambda x: "Ya" if x else "Tidak", key="wc_input",
            )
            emp_var = m4.number_input(
                "Employment Variation Rate", -4.0, 2.0, macro_defaults["emp.var.rate"],
                step=0.1, key="emp_input",
            )

            m5, m6, m7 = st.columns(3)
            cons_price = m5.number_input(
                "Consumer Price Index", 92.0, 95.0, macro_defaults["cons.price.idx"],
                step=0.01, format="%.3f", key="cpi_input",
            )
            cons_conf = m6.number_input(
                "Consumer Confidence Index", -51.0, -26.0, macro_defaults["cons.conf.idx"],
                step=0.1, key="cci_input",
            )
            euribor = m7.number_input(
                "Euribor 3 Bulan", 0.5, 5.1, macro_defaults["euribor3m"],
                step=0.01, key="eur_input",
            )
            nr_emp = st.number_input(
                "Jumlah Pekerja (kuartalan)", 4960.0, 5230.0, macro_defaults["nr.employed"],
                step=0.1, format="%.1f", key="nre_input",
            )

            st.caption(
                "Nilai indikator makroekonomi sudah diisi dengan median data historis. "
                "Sesuaikan hanya kalau tahu kondisi terkini yang berbeda."
            )

            submitted = st.form_submit_button("Prediksi", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([{
                "job": job, "marital": marital, "education": education,
                "default": default, "housing": housing, "loan": loan,
                "contact": contact, "month": month, "day_of_week": day_of_week,
                "poutcome": poutcome, "age": age, "campaign": campaign,
                "previous": previous, "emp.var.rate": emp_var,
                "cons.price.idx": cons_price, "cons.conf.idx": cons_conf,
                "euribor3m": euribor, "nr.employed": nr_emp,
                "was_contacted_before": was_contacted,
            }])

            try:
                probs, preds = predict_with_threshold(model, input_df[ALL_FEATURES], threshold)
                prob = probs[0]
                pred = preds[0]
                is_priority = pred == 1

                st.markdown("")
                r1, r2 = st.columns([1, 1])
                with r1:
                    cls = "pred-priority" if is_priority else "pred-nonpriority"
                    label = "PRIORITAS: Hubungi Nasabah Ini" if is_priority else "BUKAN PRIORITAS: Lewati atau Turunkan Prioritas"
                    color = "#10B981" if is_priority else "#EF4444"
                    st.markdown(
                        f'<div class="pred-card {cls}">'
                        f'<div class="pred-label" style="color:{color}">{label}</div>'
                        f'<div class="pred-prob">Probabilitas Berlangganan: <b>{prob:.1%}</b></div>'
                        f'<div class="pred-prob">Threshold yang dipakai: {threshold:.2f}</div>'
                        "</div>",
                        unsafe_allow_html=True,
                    )

                with r2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%", "font": {"size": 42, "color": "#E2E8F0"}},
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor="#64748B"),
                            bar=dict(color="#3B82F6"),
                            bgcolor="rgba(30,41,59,0.5)",
                            steps=[
                                dict(range=[0, threshold * 100], color="rgba(239,68,68,0.15)"),
                                dict(range=[threshold * 100, 100], color="rgba(16,185,129,0.15)"),
                            ],
                            threshold=dict(
                                line=dict(color="#F59E0B", width=3),
                                thickness=0.8,
                                value=threshold * 100,
                            ),
                        ),
                        title=dict(text="Probabilitas Berlangganan", font=dict(size=16, color="#94A3B8")),
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=280,
                        margin=dict(l=30, r=30, t=60, b=10),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan prediksi: {e}")

    # ────────────────────────────────────────────
    # TAB 2 - Prediksi Batch
    # ────────────────────────────────────────────
    with tab_batch:
        st.markdown('<div class="section-header">Upload CSV Nasabah</div>', unsafe_allow_html=True)
        st.markdown(
            "Upload file CSV berisi data nasabah. Kolom yang dibutuhkan: "
            f"`{'`, `'.join(CATEGORICAL_FEATURES + NUMERIC_FEATURES)}`."
        )

        uploaded = st.file_uploader("Pilih file CSV", type=["csv"], key="batch_upload")

        th_col1, th_col2 = st.columns(2)
        with th_col1:
            th_choice = st.radio(
                "Strategi Threshold",
                ["Deployment (0,40)", "Profit-Optimal (0,33)", "F2-Optimal (0,49)"],
                horizontal=True,
                key="th_choice",
            )
        threshold_map = {
            "Deployment (0,40)": metadata["threshold_deployment"],
            "Profit-Optimal (0,33)": metadata["threshold_profit_optimal"],
            "F2-Optimal (0,49)": metadata["threshold_f2_optimal"],
        }
        batch_threshold = threshold_map[th_choice]
        with th_col2:
            st.metric("Threshold Aktif", f"{batch_threshold:.2f}")

        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                st.success(f"Berhasil memuat **{len(df_upload):,}** baris x **{len(df_upload.columns)}** kolom")

                with st.expander("Pratinjau data yang diupload", expanded=False):
                    st.dataframe(df_upload.head(10), use_container_width=True)

                df_pred = prepare_input(df_upload)
                probs, preds = predict_with_threshold(model, df_pred, batch_threshold)

                result = df_upload.copy()
                result["probability"] = probs
                result["prediction"] = preds
                result["recommendation"] = result["prediction"].map(
                    {1: "Prioritas", 0: "Bukan Prioritas"}
                )
                result = result.sort_values("probability", ascending=False)

                n_priority = (preds == 1).sum()
                n_total = len(preds)
                st.markdown('<div class="section-header">Ringkasan Hasil</div>', unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4)
                for col, (v, l) in zip(
                    [s1, s2, s3, s4],
                    [
                        (f"{n_total:,}", "Total Nasabah"),
                        (f"{n_priority:,}", "Prioritas (Hubungi)"),
                        (f"{n_total - n_priority:,}", "Bukan Prioritas (Lewati)"),
                        (f"{n_priority / n_total:.1%}", "Tingkat Panggilan"),
                    ],
                ):
                    col.markdown(
                        f'<div class="metric-card"><div class="metric-value">{v}</div>'
                        f'<div class="metric-label">{l}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("")

                fig_dist = px.histogram(
                    result, x="probability", color="recommendation",
                    nbins=40, barmode="overlay", opacity=0.7,
                    color_discrete_map={"Prioritas": "#10B981", "Bukan Prioritas": "#3B82F6"},
                    labels={"probability": "Probabilitas Berlangganan", "recommendation": ""},
                )
                fig_dist.add_vline(
                    x=batch_threshold, line_dash="dash", line_color="#F59E0B",
                    annotation_text=f"Threshold = {batch_threshold:.2f}",
                    annotation_font_color="#F59E0B",
                )
                fig_dist.update_layout(**PLOT_LAYOUT, height=350)
                st.plotly_chart(fig_dist, use_container_width=True)

                st.markdown('<div class="section-header">Hasil Detail</div>', unsafe_allow_html=True)
                display_cols = [c for c in result.columns if c not in ("target", "y")]
                st.dataframe(
                    result[display_cols].style.format({"probability": "{:.2%}"}),
                    use_container_width=True,
                    height=400,
                )

                csv_out = result[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Hasil Prediksi",
                    data=csv_out,
                    file_name="bms_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except ValueError as ve:
                st.error(f"Data tidak valid: {ve}")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses: {e}")


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
if page == "Konteks Proyek":
    page_context()
elif page == "Dashboard Interaktif":
    page_dashboard()
else:
    page_predictor()
