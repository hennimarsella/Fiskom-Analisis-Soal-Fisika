import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Soal Fisika", layout="wide")
st.title("📊 Dashboard Analisis Hasil Ujian Fisika")
st.markdown("**Oleh: Henni Marsella**")
st.markdown("Analisis berbasis data untuk evaluasi 20 butir soal dan 50 siswa")

# ==========================================================
# UPLOAD DATA
# ==========================================================
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
    
    # ==========================================================
    # KPI HASIL UJIAN
    # ==========================================================
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
    
    # ==========================================================
    # 1️⃣ ANALISIS KINERJA BUTIR SOAL
    # ==========================================================
    st.header("1️⃣ Analisis Kinerja Butir Soal")
    st.markdown("Mengukur rata-rata jawaban siswa per soal untuk identifikasi soal yang perlu perhatian")
    
    mean_scores = indikator.mean()
    
    fig_perf, ax_perf = plt.subplots(figsize=(14, 5))
    colors = ['green' if x >= 3.0 else ('orange' if x >= 2.0 else 'red') for x in mean_scores]
    ax_perf.bar(mean_scores.index, mean_scores.values, color=colors, alpha=0.8)
    ax_perf.axhline(3.0, linestyle='--', color='green', linewidth=1, label='Baik (≥3.0)')
    ax_perf.axhline(2.0, linestyle='--', color='orange', linewidth=1, label='Cukup (≥2.0)')
    ax_perf.set_ylabel("Rata-rata Skor per Soal")
    ax_perf.set_title("Kinerja Butir Soal (Skala 1-4)")
    ax_perf.set_ylim(0, 4.5)
    ax_perf.legend()
    ax_perf.tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(mean_scores.values):
        ax_perf.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    
    st.pyplot(fig_perf)
    
    soal_terendah = mean_scores.idxmin()
    st.info(f"📌 **Soal dengan kinerja terendah**: {soal_terendah} (rata-rata: {mean_scores.min():.2f})")
    st.divider()
    
    # ==========================================================
    # 2️⃣ KORELASI ANTAR SOAL (LABEL ANGKA 1-20)
    # ==========================================================
    st.header("2️⃣ Korelasi Antar Butir Soal")
    st.markdown("Menunjukkan hubungan linear antar butir soal. Nilai mendekati 1 = hubungan kuat positif.")
    
    corr = indikator.corr()
    labels_angka = [str(i) for i in range(1, 21)]
    
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_corr, label='Koefisien Korelasi')
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(labels_angka, rotation=90, ha="center", fontsize=8)
    ax_corr.set_yticklabels(labels_angka, fontsize=8)
    
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    
    ax_corr.set_title("Heatmap Korelasi Pearson (Soal 1 - 20)")
    st.pyplot(fig_corr)
    
    corr_total = indikator.corrwith(df['Total_Skor']).sort_values(ascending=False)
    st.subheader("📊 Ranking Soal Berpengaruh terhadap Total Skor")
    st.dataframe(corr_total.head(10).to_frame("Koefisien Korelasi"), use_container_width=True, hide_index=True)
    st.divider()
    
    # ==========================================================
    # 3️⃣ ANALISIS REGRESI (DENGAN PENJELASAN)
    # ==========================================================
    st.header("3️⃣ Analisis Regresi Linear")
    st.markdown("""
    **Apa itu Regresi?**  
    Regresi linear berganda digunakan untuk melihat seberapa besar pengaruh Soal_1 sampai Soal_19 terhadap hasil Soal_20.
    
    **Interpretasi:**
    - **Nilai R²**: Menunjukkan persentase variasi Soal_20 yang dapat dijelaskan oleh Soal_1–19
    - **Koefisien positif**: Semakin tinggi skor soal tersebut, semakin tinggi prediksi skor Soal_20
    - **Koefisien negatif**: Semakin tinggi skor soal tersebut, semakin rendah prediksi skor Soal_20
    """)
    
    X = sm.add_constant(indikator.iloc[:, :-1])
    y = indikator.iloc[:, -1]
    model = sm.OLS(y, X, missing="drop").fit()
    coef = model.params[1:]
    r2 = model.rsquared
    
    st.info(f"📈 **Nilai R² = {r2:.3f}** → {r2*100:.1f}% variasi skor Soal_20 dapat dijelaskan oleh kombinasi Soal_1–19")
    
    fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
    coef_sorted = coef.abs().sort_values(ascending=False).head(5)
    colors_reg = ['green' if coef.loc[idx] > 0 else 'red' for idx in coef_sorted.index]
    ax_reg.bar(coef_sorted.index, [coef.loc[idx] for idx in coef_sorted.index], color=colors_reg, alpha=0.8)
    ax_reg.axhline(0, linestyle="--", color="black")
    ax_reg.set_title("TOP 5 Soal Paling Berpengaruh terhadap Soal_20")
    ax_reg.set_ylabel("Koefisien Regresi")
    ax_reg.tick_params(axis='x', rotation=45)
    
    for bar, idx in zip(ax_reg.patches, coef_sorted.index):
        val = coef.loc[idx]
        ax_reg.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', ha='center', fontsize=8)
    
    st.pyplot(fig_reg)
    st.success(f"🔑 **Faktor paling dominan**: {coef.abs().idxmax()} (koefisien: {coef.abs().max():.4f})")
    st.divider()
    
    # ==========================================================
    # 4️⃣ SEGMENTASI SISWA (DENGAN HASIL TABEL)
    # ==========================================================
    st.header("4️⃣ Segmentasi Kemampuan Siswa")
    st.markdown("""
    **Apa itu Segmentasi?**  
    Pengelompokan siswa berdasarkan pola jawaban menggunakan algoritma K-Means Clustering.
    
    **Kategori:**
    - 🟢 **Kelompok Mahir**: Skor di atas rata-rata kelas
    - 🟡 **Kelompok Menengah**: Skor sekitar rata-rata kelas  
    - 🔴 **Kelompok Perlu Bimbingan**: Skor di bawah rata-rata kelas
    """)
    
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
    
    st.subheader("📋 Hasil Pengelompokan Siswa")
    segment_table = df[['Responden', 'Total_Skor', 'Segment']].copy()
    st.dataframe(segment_table.head(20), use_container_width=True, hide_index=True)
    
    st.subheader("📊 Ringkasan per Kelompok")
    summary = df.groupby('Segment').agg({
        'Total_Skor': ['count', 'mean', 'min', 'max']
    }).round(2)
    summary.columns = ['Jumlah Siswa', 'Rata-rata Skor', 'Skor Terendah', 'Skor Tertinggi']
    st.dataframe(summary, use_container_width=True)
    
    fig_seg, ax_seg = plt.subplots(figsize=(8, 5))
    seg_counts = df['Segment'].value_counts()
    colors_seg = ['#2ecc71', '#f1c40f', '#e74c3c']
    ax_seg.bar(seg_counts.index, seg_counts.values, color=colors_seg, alpha=0.8)
    ax_seg.set_ylabel('Jumlah Siswa')
    ax_seg.set_title('Distribusi Siswa per Kelompok')
    ax_seg.tick_params(axis='x', rotation=45)
    st.pyplot(fig_seg)
    
    st.success("📌 Segmentasi berhasil – gunakan hasil ini untuk strategi pembelajaran & remedial yang tepat sasaran")
    st.divider()
    
    # ==========================================================
    # 💡 KESIMPULAN & REKOMENDASI
    # ==========================================================
    st.header("💡 Kesimpulan & Rekomendasi")
    
    prioritas_gap = (4 - mean_scores).idxmax()
    gap_max = (4 - mean_scores).max()
    top_corr = corr_total.index[0].split('_')[1]
    r_val = corr_total.iloc[0]
    
    st.markdown(f"""
    ### 📊 Temuan Utama:
    1. **Kinerja Kelas**: Rata-rata skor {avg_total:.1f} dari {max_score} ({avg_percentage:.1f}%) — kategori **{kategori_ikm(avg_percentage)}**.
    2. **Soal Prioritas**: {prioritas_gap} memiliki GAP terbesar ({gap_max:.2f}) — perlu revisi materi/pembuatan soal.
    3. **Korelasi Kuat**: Soal dengan korelasi tertinggi terhadap total skor adalah Soal_{top_corr} (r = {r_val:.3f}).
    4. **Segmentasi**: Siswa terbagi menjadi 3 kelompok — strategi pembelajaran perlu disesuaikan.

    ### 🎯 Rekomendasi Tindak Lanjut:
    | Prioritas | Aksi | Target |
    |-----------|------|--------|
    | 🔴 Tinggi | Revisi soal {prioritas_gap} | Tim Penyusun Soal |
    | 🟡 Sedang | Remedial untuk Kelompok Perlu Bimbingan | Guru Mata Pelajaran |
    | 🟢 Rendah | Pertahankan soal dengan kinerja baik | Semua Pihak |

    ### ⚠️ Catatan Analisis:
    - Data berasal dari {len(df)} siswa dengan 20 butir soal (skala 1-4)
    - Segmentasi menggunakan K-Means Clustering (3 cluster)
    - Analisis regresi menjelaskan {r2*100:.1f}% variansi hasil
    """)
    
    st.success("✅ Dashboard ini dapat digunakan sebagai dasar pengambilan keputusan pembelajaran.")
    
    # ==========================================================
    # SIDEBAR & FOOTER
    # ==========================================================
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard Analisis Butir Soal Fisika • Evaluasi kualitas soal & pemetaan kemampuan siswa")
    
    st.markdown("---")
    st.caption("Dashboard Analisis Soal Fisika • **Henni Marsella** • 2026")
    
else:
    st.info("👈 Silakan upload file Excel di sidebar untuk memulai analisis")
