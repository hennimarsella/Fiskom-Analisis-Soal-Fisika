import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Analisis Soal Fisika", layout="wide")
st.title("📊 Dashboard Analisis Hasil Ujian Fisika")
st.markdown("Analisis berbasis data untuk evaluasi 20 butir soal dan 50 siswa")

# Upload Data
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Baca data
    df = pd.read_excel(uploaded_file)
    
    # Ambil kolom soal (Soal_1 sampai Soal_20)
    soal_cols = [col for col in df.columns if 'Soal' in col]
    indikator = df[soal_cols].apply(pd.to_numeric, errors="coerce")
    
    # Hitung total skor per siswa
    df['Total_Skor'] = indikator.sum(axis=1)
    max_score = len(soal_cols) * 4
    avg_total = df['Total_Skor'].mean()
    avg_percentage = (avg_total / max_score) * 100
    
    # KPI
    def kategori_ikm(x):
        if x >= 80: return "Sangat Baik"
        elif x >= 65: return "Baik"
        elif x >= 50: return "Cukup"
        else: return "Perlu Perbaikan"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Rata-rata Skor Ujian", f"{avg_percentage:.2f}%")
    col2.metric("🏷️ Kategori", kategori_ikm(avg_percentage))
    col3.metric("👥 Total Responden", len(df))
    st.divider()
    
    # 1️⃣ Analisis GAP
    st.header("1️⃣ Analisis GAP")
    mean_scores = indikator.mean()
    gap_scores = 4 - mean_scores
    prioritas_gap = gap_scores.idxmax()
    
    fig_gap, ax_gap = plt.subplots(figsize=(14, 5))
    ax_gap.bar(gap_scores.index, gap_scores.values, color=plt.cm.Set2(range(len(gap_scores))))
    ax_gap.axhline(1.0, linestyle='--', color='blue', linewidth=2, label='Threshold GAP = 1.0')
    ax_gap.axhline(1.5, linestyle='--', color='red', linewidth=1, label='Zona Prioritas')
    ax_gap.set_ylabel("Nilai GAP (4 - Rata-rata)")
    ax_gap.set_title("GAP Kinerja per Butir Soal")
    ax_gap.set_ylim(0, 3)
    ax_gap.legend()
    ax_gap.tick_params(axis='x', rotation=45)
    st.pyplot(fig_gap)
    
    # 2️⃣ Korelasi
    st.header("2️⃣ Korelasi Antar Soal")
    corr = indikator.corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_corr)
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=90)
    ax_corr.set_yticklabels(corr.columns)
    st.pyplot(fig_corr)
    
    # 3️⃣ Regresi
    st.header("3️⃣ Analisis Regresi")
    X = sm.add_constant(indikator.iloc[:, :-1])
    y = indikator.iloc[:, -1]
    model = sm.OLS(y, X, missing="drop").fit()
    r2 = model.rsquared
    st.info(f"Nilai R²: {r2:.3f}")
    
    # 4️⃣ Segmentasi
    st.header("4️⃣ Segmentasi Siswa")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_label = kmeans.fit_predict(X_scaled)
    df['Cluster'] = cluster_label
    
    avg_per_cluster = df.groupby('Cluster')['Total_Skor'].mean()
    cluster_urut = avg_per_cluster.sort_values(ascending=False).index.tolist()
    
    segment_map = {}
    for i, cluster_id in enumerate(cluster_urut):
        if i == 0:
            segment_map[cluster_id] = 'Kelompok Mahir'
        elif i == 1:
            segment_map[cluster_id] = 'Kelompok Menengah'
        else:
            segment_map[cluster_id] = 'Kelompok Perlu Bimbingan'
    
    df['Segment'] = df['Cluster'].map(segment_map)
    st.success("📌 Segmentasi berhasil – siap untuk strategi pembelajaran & remedial")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard Analisis Butir Soal Fisika")
else:
    st.info("👈 Silakan upload file Excel di sidebar")
