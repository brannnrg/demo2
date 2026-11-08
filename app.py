import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- KONFIGURASI HALAMAN (POLOS TANPA IKON) ---
st.set_page_config(
    page_title="Klasifikasi KNN - Machine Learning UNP Kediri",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ELEGAN ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 22px 26px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 26px;
        font-weight: 700;
    }
    .main-header p {
        color: #e0e7ff !important;
        margin: 5px 0 0 0;
        font-size: 14px;
    }
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER APLIKASI ---
st.markdown("""
<div class="main-header">
    <h1>Aplikasi Klasifikasi K-Nearest Neighbors (KNN) - Kelompok 2</h1>
</div>
""", unsafe_allow_html=True)

# --- LOAD DATASETS DENGAN MULTI-PATH ROBUST ---
@st.cache_data
def load_datasets():
    candidate_dirs = [
        os.path.dirname(__file__),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        r"D:\TUGAS KULIAH\Machine Learning\Tugas Klasifikasi KKN",
        r"C:\Users\ergi2\OneDrive\Desktop\web_aplikasi_knn"
    ]
    df_iris = None
    df_bal = None
    for d in candidate_dirs:
        p_iris = os.path.join(d, "iris.csv")
        p_bal = os.path.join(d, "balance-scale.csv")
        if df_iris is None and os.path.exists(p_iris):
            df_iris = pd.read_csv(p_iris)
        if df_bal is None and os.path.exists(p_bal):
            df_bal = pd.read_csv(p_bal)
    return df_iris, df_bal

df_iris, df_bal = load_datasets()

# --- SIDEBAR KONTROL ---
with st.sidebar:
    st.header("Pengaturan Model")
    
    dataset_choice = st.selectbox(
        "Pilih Dataset:",
        ["Balance Scale Dataset", "Iris Dataset"]
    )
    
    st.markdown("---")
    st.subheader("Parameter Pembagian Data")
    
    split_mode = st.radio(
        "Rasio Data Latih : Uji",
        ["80% : 20% (Skenario 1)", "70% : 30% (Skenario 2)", "60% : 40% (Skenario 3)"]
    )
    
    if split_mode == "80% : 20% (Skenario 1)":
        test_size = 0.20
    elif split_mode == "70% : 30% (Skenario 2)":
        test_size = 0.30
    else:
        test_size = 0.40

    k_val = st.slider("Nilai K (Jumlah Tetangga):", min_value=1, max_value=20, value=3, step=1)
    
    # Metode jarak murni Euclidean sesuai instruksi soal
    metric_choice = "euclidean"
    st.markdown("---")
    st.info("Metrik Jarak: Euclidean Distance")
    
    st.markdown("---")
    st.caption("Kelompok 2")

# --- PROSES DATASET TERPILIH SESUAI KODING DAFFA & LAPORAN ---
if dataset_choice == "Balance Scale Dataset":
    df_active = df_bal
    target_col = "Class"
    feature_cols = ['L-Weight', 'L-Distance', 'R-Weight', 'R-Distance']
    if df_active is not None:
        x_raw = np.array(df_active[feature_cols].values, dtype=np.float64)
        y = np.array(df_active[target_col].tolist())
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x_raw)
        class_labels = ['B', 'L', 'R']
else:
    df_active = df_iris
    target_col = "Species"
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    if df_active is not None:
        x = np.array(df_active[feature_cols].values, dtype=np.float64)
        y = np.array(df_active[target_col].tolist())
        scaler = None
        class_labels = sorted(list(set(y)))

if df_active is None:
    st.error("⚠️ File dataset (.csv) tidak ditemukan. Pastikan file 'iris.csv' dan 'balance-scale.csv' ada di folder aplikasi.")
    st.stop()

# Split Data konsisten dengan notebook Daffa (Cell 12: random_state=0 tanpa stratify)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=0
)

# Training KNN
knn = KNeighborsClassifier(n_neighbors=k_val, metric=metric_choice).fit(x_train, y_train)
y_pred = knn.predict(x_test)
akurasi_aktif = accuracy_score(y_test, y_pred)

# --- TAB NAVIGASI UTAMA (MURNI TEKS TANPA IKON) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Eksplorasi Data",
    "Evaluasi Model & K Terbaik",
    "Kurva & 3 Skenario",
    "Klasifikasi Data Baru"
])

# ==================== TAB 1 ====================
with tab1:
    st.subheader(f"Dataset Aktif: {dataset_choice}")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Total Sampel", f"{df_active.shape[0]} baris")
    col_d2.metric("Jumlah Fitur", f"{len(feature_cols)} atribut")
    col_d3.metric("Data Latih / Uji", f"{len(x_train)} / {len(x_test)}")
    
    st.markdown("#### Preview Data (Head & Info)")
    st.dataframe(df_active.head(10), use_container_width=True)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"#### Sebaran Kelas Target (`{target_col}`.value_counts())")
        vc = df_active[target_col].value_counts().reset_index()
        vc.columns = [target_col, "Jumlah"]
        st.dataframe(vc, use_container_width=True)
        st.caption(f"Kelas Unik: `{list(df_active[target_col].unique())}`")
    with col_info2:
        st.markdown("#### Statistik Deskriptif (`.describe()`)")
        st.dataframe(df_active[feature_cols].describe(), use_container_width=True)
        if dataset_choice == "Balance Scale Dataset":
            st.info(f"Feature Scaling (MinMaxScaler) -> Nilai Min: {x.min():.1f}, Nilai Max: {x.max():.1f}")

# ==================== TAB 2 ====================
with tab2:
    st.subheader("Hasil Evaluasi Model KNN")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown(f'<div class="card"><p style="margin:0;color:#64748b;">Akurasi Model Aktif</p><div class="metric-value">{akurasi_aktif*100:.2f}%</div><p style="margin:0;font-size:12px;color:#059669;">K = {k_val} | Split {int((1-test_size)*100)}:{int(test_size*100)}</p></div>', unsafe_allow_html=True)
    
    scores_k = []
    for k_i in range(1, 21):
        m_i = KNeighborsClassifier(n_neighbors=k_i, metric=metric_choice).fit(x_train, y_train)
        scores_k.append(accuracy_score(y_test, m_i.predict(x_test)))
    best_acc = max(scores_k)
    best_k = scores_k.index(best_acc) + 1
    
    col_m2.markdown(f'<div class="card"><p style="margin:0;color:#64748b;">K Terbaik (Akurasi Teratas)</p><div class="metric-value">K = {best_k}</div><p style="margin:0;font-size:12px;color:#2563eb;">Akurasi: {best_acc*100:.2f}%</p></div>', unsafe_allow_html=True)
    col_m3.markdown(f'<div class="card"><p style="margin:0;color:#64748b;">Metrik Jarak</p><div class="metric-value" style="font-size:22px;text-transform:capitalize;">{metric_choice}</div><p style="margin:0;font-size:12px;color:#64748b;">Euclidean Distance</p></div>', unsafe_allow_html=True)
    
    col_eval1, col_eval2 = st.columns([1, 1])
    with col_eval1:
        st.markdown("#### Visualisasi Confusion Matrix dengan Seaborn")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=class_labels, yticklabels=class_labels)
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        st.pyplot(fig_cm)
        
    with col_eval2:
        st.markdown("#### Classification Report (`classification_report`)")
        rep = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)

# ==================== TAB 3 ====================
with tab3:
    st.subheader("Kurva K & Eksperimen 3 Skenario (Sesuai Koding Daffa & Soal.txt)")
    
    # 1. Kurva K 1 s/d 20 dengan marker 'o' sesuai gaya Daffa
    n_akurasi = []
    for k_i in range(1, 21):
        knn_i = KNeighborsClassifier(n_neighbors=k_i, metric=metric_choice).fit(x_train, y_train)
        n_akurasi.append(accuracy_score(y_test, knn_i.predict(x_test)))
    
    fig_curve, ax_curve = plt.subplots(figsize=(14, 6))
    ax_curve.plot(range(1, 21), n_akurasi, marker='o', color='#1f77b4', linewidth=2)
    ax_curve.set_xticks(range(1, 21))
    ax_curve.set_xlabel("Nilai K")
    ax_curve.set_ylabel("Nilai Akurasi")
    ax_curve.set_title(f"Grafik K Terbaik pada {dataset_choice} (Split Aktif)", fontsize=13, fontweight='bold')
    ax_curve.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_curve)
    
    st.markdown("---")
    st.subheader("Eksperimen 3 Skenario Train-Test Split")
    st.caption("Implementasi dictionary `skenario_split` dan pemformatan `lambda v: f\"{v*100:.2f}%\"` persis notebook Daffa:")
    
    # Logika persis koding Daffa
    skenario_split = {
        "Skenario 1 (Train 80% : Test 20%)": 0.2,
        "Skenario 2 (Train 70% : Test 30%)": 0.3,
        "Skenario 3 (Train 60% : Test 40%)": 0.4
    }
    
    rekap_skenario = {}
    summary_text = ""
    
    for nama_skenario, ukuran_test in skenario_split.items():
        xtr_s, xts_s, ytr_s, yts_s = train_test_split(x, y, test_size=ukuran_test, random_state=42, stratify=y)
        accs = []
        for k_s in range(1, 21):
            knn_s = KNeighborsClassifier(n_neighbors=k_s, metric='euclidean').fit(xtr_s, ytr_s)
            accs.append(accuracy_score(yts_s, knn_s.predict(xts_s)))
        
        rekap_skenario[nama_skenario] = accs
        best_a = max(accs)
        best_k = accs.index(best_a) + 1
        
        summary_text += f"{nama_skenario}\n"
        summary_text += f" -> K Terbaik       : {best_k}\n"
        summary_text += f" -> Akurasi         : {best_a:.4f} ({best_a*100:.2f}%)\n\n"
        
    st.code(summary_text, language="text")
    
    st.markdown("#### Tabel Komparasi Seluruh K (1 s/d 20) Format Persentase Lambda:")
    df_skenario = pd.DataFrame(rekap_skenario, index=[f'K = {k}' for k in range(1, 21)])
    
    # Pemformatan lambda persis notebook Daffa
    df_tampil = df_skenario.map(lambda v: f"{v*100:.2f}%")
    st.dataframe(df_tampil, use_container_width=True)

# ==================== TAB 4 ====================
with tab4:
    st.subheader("Klasifikasi Data Baru (`new_data`)")
    st.write("Masukkan nilai fitur di bawah ini untuk menguji model klasifikasi secara instan:")
    
    cols_in = st.columns(len(feature_cols))
    input_values = []
    
    for idx_f, f_name in enumerate(feature_cols):
        with cols_in[idx_f]:
            if dataset_choice == "Balance Scale Dataset":
                default_vals = [4, 3, 2, 1]
                val = st.number_input(f_name, min_value=1, max_value=5, value=default_vals[idx_f], step=1)
            else:
                default_vals = [5.1, 3.5, 1.4, 0.2]
                val = st.number_input(f_name, min_value=0.1, max_value=10.0, value=default_vals[idx_f], step=0.1)
            input_values.append(val)
            
    if st.button("Prediksi Data Baru", use_container_width=True):
        raw_arr = np.array([input_values])
        if scaler is not None:
            processed_input = scaler.transform(raw_arr)
        else:
            processed_input = raw_arr
            
        pred_new = knn.predict(processed_input)[0]
        
        st.success(f"### Hasil Prediksi Kelas: **{pred_new}**")
        if dataset_choice == "Balance Scale Dataset":
            ket = "Seimbang (Balanced)" if pred_new == 'B' else ("Condong ke Kiri (Left)" if pred_new == 'L' else "Condong ke Kanan (Right)")
            st.info(f"Keterangan Fisik: **{ket}**")
