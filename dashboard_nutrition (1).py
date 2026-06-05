# dashboard_nutrition_id.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi Kalori Resep Ayam",
    page_icon="🍗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-top: 4px solid #2E7D32;
    }
    .metric-card h3 {
        color: #2E7D32;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .metric-card h2 {
        color: #1B5E20;
        font-size: 1.8rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #c62828;
        margin: 1rem 0;
    }
    .badge-success {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #FF9800;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .badge-danger {
        background-color: #f44336;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .rec-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        height: 100%;
    }
    .rec-card h4 {
        color: #2E7D32;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #2E7D32;
        padding-bottom: 0.5rem;
    }
    .dataframe-container {
        max-height: 400px;
        overflow: auto;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Load data dengan caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_nutrition_data.csv')
        return df
    except:
        st.warning("Dataset tidak ditemukan. Menggunakan data contoh...")
        np.random.seed(42)
        n_samples = 5000
        df = pd.DataFrame({
            'Title': [f'Resep Ayam {i}' for i in range(n_samples)],
            'Ingredients': ['bahan1--bahan2--bahan3'] * n_samples,
            'Steps': ['langkah1--langkah2--langkah3'] * n_samples,
            'Loves': np.random.randint(0, 1000, n_samples),
            'URL': ['/id/resep/1'] * n_samples,
            'jenis_makanan': ['ayam'] * n_samples,
            'usia': np.random.randint(18, 65, n_samples),
            'jumlah_kalori': np.random.normal(780, 400, n_samples)
        })
        df['jumlah_kalori'] = df['jumlah_kalori'].clip(100, 1500)
        return df

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_nutrision.pkl')
        scaler = joblib.load('scaler_food.pkl')
        return model, scaler
    except:
        try:
            with open('best_calorie_model.pkl', 'rb') as f:
                model = pickle.load(f)
            scaler = None
            return model, scaler
        except:
            st.warning("Model tidak ditemukan. Menggunakan prediksi sederhana.")
            return None, None

# Feature engineering
def engineer_features(df):
    df_copy = df.copy()
    df_copy['title_length'] = df_copy['Title'].fillna('').apply(len)
    df_copy['title_word_count'] = df_copy['Title'].fillna('').apply(lambda x: len(str(x).split()))
    
    def count_ingredients(ing_str):
        if pd.isna(ing_str):
            return 0
        return len(str(ing_str).split('--'))
    
    df_copy['num_ingredients'] = df_copy['Ingredients'].fillna('').apply(count_ingredients)
    df_copy['ingredients_length'] = df_copy['Ingredients'].fillna('').apply(len)
    df_copy['num_steps'] = df_copy['Steps'].fillna('').apply(lambda x: len(str(x).split('--')))
    df_copy['steps_length'] = df_copy['Steps'].fillna('').apply(len)
    df_copy['url_length'] = df_copy['URL'].fillna('').apply(len)
    df_copy['loves_usia_interaction'] = df_copy['Loves'] * df_copy['usia']
    
    return df_copy

# Load data
df = load_data()
df_featured = engineer_features(df)

# Feature columns
feature_columns = [
    'usia', 'Loves', 'title_length', 'title_word_count',
    'num_ingredients', 'ingredients_length', 'num_steps', 'steps_length',
    'url_length', 'loves_usia_interaction'
]

# Load model
model, scaler = load_model()

# Header
st.markdown('<div class="main-header">🍗 Dashboard Prediksi Kalori Resep Ayam</div>', 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🧭 Navigasi")
    page = st.radio(
        "",
        ["📊 Ikhtisar Data", "📈 Kinerja Model", "🔍 Prediksi Kalori", "📉 Data & Insights"]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.info(
        "Aplikasi ini menganalisis resep ayam untuk memprediksi jumlah kalori. "
        "Target bisnis: **MAE ≤ 75 kkal** dan **R² ≥ 0.75**."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistik Cepat")
    st.metric("Total Resep", f"{len(df):,}")
    st.metric("Rata-rata Kalori", f"{df['jumlah_kalori'].mean():.0f} kkal")
    st.metric("Rata-rata Likes", f"{df['Loves'].mean():.1f}")

# ==================== HALAMAN 1: IKHTISAR DATA ====================
if page == "📊 Ikhtisar Data":
    st.header("📊 Ikhtisar Data Resep Ayam")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Total Resep</h3>
            <h2>{len(df):,}</h2>
            <p>Resep ayam unik</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Rata-rata Kalori</h3>
            <h2>{df['jumlah_kalori'].mean():.0f} kkal</h2>
            <p>per porsi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Rentang Kalori</h3>
            <h2>{df['jumlah_kalori'].min():.0f} - {df['jumlah_kalori'].max():.0f}</h2>
            <p>minimum ke maksimum</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>❤️ Rata-rata Popularitas</h3>
            <h2>{df['Loves'].mean():.1f}</h2>
            <p>likes per resep</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Distribusi Kalori")
        fig = px.histogram(
            df, x='jumlah_kalori', 
            nbins=50, 
            title='Persebaran Jumlah Kalori Resep',
            labels={'jumlah_kalori': 'Kalori (kkal)', 'count': 'Jumlah Resep'},
            color_discrete_sequence=['#2E7D32']
        )
        fig.add_vline(x=df['jumlah_kalori'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Rata-rata: {df['jumlah_kalori'].mean():.0f}")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 10 Resep Paling Populer")
        top_recipes = df.nlargest(10, 'Loves')[['Title', 'Loves', 'jumlah_kalori']]
        top_recipes.columns = ['Judul Resep', 'Jumlah Likes', 'Kalori']
        fig = px.bar(
            top_recipes, 
            x='Jumlah Likes', 
            y='Judul Resep',
            orientation='h',
            title='Resep dengan Likes Terbanyak',
            labels={'Jumlah Likes': 'Jumlah Likes', 'Judul Resep': ''},
            color='Kalori',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tampilkan data mentah
    with st.expander("📋 Lihat Data Mentah"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Menampilkan 100 dari {len(df)} baris data")

# ==================== HALAMAN 2: KINERJA MODEL ====================
elif page == "📈 Kinerja Model":
    st.header("📈 Analisis Kinerja Model")
    
    # Hasil model
    model_results = {
        'Model': ['Regresi Linear', 'Random Forest', 'XGBoost', 'Random Forest Tuned', 'XGBoost Tuned'],
        'MAE': [362.29, 366.46, 376.53, 361.69, 362.40],
        'RMSE': [418.19, 425.52, 442.42, 418.67, 418.43],
        'R²': [-0.0018, -0.0372, -0.1212, -0.0041, -0.0030],
        'MAPE': [108.96, 109.95, 111.05, 109.03, 108.96]
    }
    results_df = pd.DataFrame(model_results)
    
    TARGET_MAE = 75
    TARGET_R2 = 0.75
    
    # Ringkasan pencapaian target
    st.markdown("### 🎯 Pencapaian Target")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_mae = results_df.loc[results_df['MAE'].idxmin(), 'MAE']
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Target MAE</h3>
            <h2 style="color: #f44336">{best_mae:.2f} / {TARGET_MAE} kkal</h2>
            <p>❌ TARGET TIDAK TERCAPAI</p>
            <p>Selisih: {best_mae - TARGET_MAE:.2f} kkal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_r2 = results_df.loc[results_df['R²'].idxmax(), 'R²']
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Target R²</h3>
            <h2 style="color: #f44336">{best_r2:.4f} / {TARGET_R2}</h2>
            <p>❌ TARGET TIDAK TERCAPAI</p>
            <p>R² negatif → lebih buruk dari rata-rata</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏆 Model Terbaik</h3>
            <h2>{best_model_name}</h2>
            <p>R²: {results_df.loc[results_df['Model'] == best_model_name, 'R²'].values[0]:.4f}</p>
            <p>MAE: {results_df.loc[results_df['Model'] == best_model_name, 'MAE'].values[0]:.2f} kkal</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Perbandingan MAE (Semakin kecil semakin baik)")
        fig = px.bar(
            results_df,
            x='Model',
            y='MAE',
            title='Mean Absolute Error per Model',
            color='MAE',
            color_continuous_scale='Reds',
            text='MAE'
        )
        fig.add_hline(y=TARGET_MAE, line_dash="dash", line_color="green",
                      annotation_text=f"Target: {TARGET_MAE}")
        fig.update_traces(textposition='outside')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Perbandingan R² (Semakin besar semakin baik)")
        fig = px.bar(
            results_df,
            x='Model',
            y='R²',
            title='R² Score per Model',
            color='R²',
            color_continuous_scale='RdYlGn',
            range_color=[-0.2, 0.1],
            text='R²'
        )
        fig.add_hline(y=TARGET_R2, line_dash="dash", line_color="green",
                      annotation_text=f"Target: {TARGET_R2}")
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                      annotation_text="Baseline (rata-rata)")
        fig.update_traces(textposition='outside')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tabel metrik detail
    st.subheader("📋 Tabel Metrik Detail")
    display_df = results_df.copy()
    display_df.columns = ['Model', 'MAE (kkal)', 'RMSE (kkal)', 'R²', 'MAPE (%)']
    st.dataframe(display_df, use_container_width=True)
    
    # Analisis kegagalan
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Analisis Kinerja</h3>
        <ul>
            <li><strong>MAE Terbaik: 361.69 kkal</strong> - Hampir <strong>5 kali lebih buruk</strong> dari target 75 kkal</li>
            <li><strong>R² Bernilai Negatif</strong> - Model lebih buruk dari memprediksi nilai rata-rata</li>
            <li><strong>MAPE ~109%</strong> - Prediksi meleset lebih dari 100% rata-rata</li>
            <li><strong>Semua model gagal</strong> mencapai target bisnis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== HALAMAN 3: PREDIKSI KALORI ====================
elif page == "🔍 Prediksi Kalori":
    st.header("🔍 Prediksi Kalori Resep")
    st.markdown("Masukkan informasi resep untuk memprediksi jumlah kalori")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Informasi Resep")
        
        judul_resep = st.text_input("Judul Resep", placeholder="Contoh: Ayam Goreng Bumbu Kunyit")
        
        col_a, col_b = st.columns(2)
        with col_a:
            panjang_judul = st.number_input("Panjang Judul (karakter)", min_value=0, max_value=500, value=50)
            jumlah_kata_judul = st.number_input("Jumlah Kata Judul", min_value=0, max_value=50, value=8)
        with col_b:
            jumlah_bahan = st.number_input("Jumlah Bahan", min_value=0, max_value=100, value=10)
            panjang_bahan = st.number_input("Panjang Teks Bahan (karakter)", min_value=0, max_value=5000, value=200)
        
    with col2:
        st.markdown("### ⚙️ Informasi Tambahan")
        
        col_c, col_d = st.columns(2)
        with col_c:
            jumlah_langkah = st.number_input("Jumlah Langkah Memasak", min_value=0, max_value=100, value=5)
            panjang_langkah = st.number_input("Panjang Teks Langkah (karakter)", min_value=0, max_value=5000, value=150)
        with col_d:
            jumlah_likes = st.number_input("Jumlah Likes", min_value=0, max_value=10000, value=100)
            usia_pengguna = st.number_input("Usia Pengguna", min_value=1, max_value=100, value=30)
            panjang_url = st.number_input("Panjang URL", min_value=0, max_value=200, value=50)
    
    # Fitur interaksi
    likes_usia_interaction = jumlah_likes * usia_pengguna
    
    # Tombol prediksi
    st.markdown("---")
    predict_button = st.button("🔮 Prediksi Kalori", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Menghitung prediksi kalori..."):
            # Buat feature array
            features = np.array([[
                usia_pengguna, jumlah_likes, panjang_judul, jumlah_kata_judul,
                jumlah_bahan, panjang_bahan, jumlah_langkah, panjang_langkah,
                panjang_url, likes_usia_interaction
            ]])
            
            # Prediksi dengan model jika ada
            if model is not None and scaler is not None:
                try:
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)[0]
                except Exception as e:
                    st.warning(f"Error menggunakan model: {e}. Menggunakan prediksi sederhana.")
                    prediction = 300 + (jumlah_bahan * 25) + (jumlah_langkah * 20) + (panjang_bahan * 0.1)
            else:
                # Prediksi sederhana
                prediction = 300 + (jumlah_bahan * 25) + (jumlah_langkah * 20) + (panjang_bahan * 0.1)
            
            # Batasi range
            prediction = max(50, min(1500, prediction))
            
            # Tampilkan hasil
            st.markdown("---")
            st.markdown("## 📊 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔥 Prediksi Kalori</h3>
                    <h2>{prediction:.0f} kkal</h2>
                    <p>per porsi</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if prediction < 400:
                    badge = 'badge-success'
                    status = 'Rendah Kalori'
                    deskripsi = 'Cocok untuk diet'
                elif prediction < 800:
                    badge = 'badge-warning'
                    status = 'Sedang'
                    deskripsi = 'Cukup mengenyangkan'
                else:
                    badge = 'badge-danger'
                    status = 'Tinggi Kalori'
                    deskripsi = 'Kaya energi'
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏷️ Kategori</h3>
                    <h2><span class="{badge}">{status}</span></h2>
                    <p>{deskripsi}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                est_per_bahan = prediction / max(jumlah_bahan, 1)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Estimasi per Bahan</h3>
                    <h2>{est_per_bahan:.0f} kkal</h2>
                    <p>rata-rata per bahan</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tampilkan input yang digunakan
            with st.expander("📋 Detail Input yang Digunakan"):
                input_data = pd.DataFrame({
                    'Fitur': ['Usia', 'Jumlah Likes', 'Panjang Judul', 'Jumlah Kata Judul', 
                              'Jumlah Bahan', 'Panjang Bahan', 'Jumlah Langkah', 'Panjang Langkah',
                              'Panjang URL', 'Likes × Usia'],
                    'Nilai': [usia_pengguna, jumlah_likes, panjang_judul, jumlah_kata_judul,
                              jumlah_bahan, panjang_bahan, jumlah_langkah, panjang_langkah,
                              panjang_url, likes_usia_interaction]
                })
                st.dataframe(input_data, use_container_width=True)
            
            # Rekomendasi
            if prediction < 400:
                st.success("✅ **Rekomendasi:** Resep ini rendah kalori, cocok untuk menu diet atau makan malam ringan.")
            elif prediction < 800:
                st.info("ℹ️ **Rekomendasi:** Resep ini memiliki kalori sedang, cocok untuk makan siang.")
            else:
                st.warning("⚠️ **Rekomendasi:** Resep ini tinggi kalori, konsumsi dengan bijak atau bagikan untuk 2 porsi.")

# ==================== HALAMAN 4: DATA & INSIGHTS ====================
else:
    st.header("📉 Data & Insights")
    
    # Business question
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 Pertanyaan Bisnis</h3>
        <p><strong>"Dapatkah model machine learning memprediksi kalori resep ayam dengan MAE ≤ 75 kkal dan R² ≥ 0.75?"</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
            <h3>❌ Target TIDAK Tercapai</h3>
            <ul>
                <li><strong>MAE Terbaik:</strong> 361.69 kkal <span style="color:#f44336">(Target: ≤75 kkal)</span></li>
                <li><strong>R² Terbaik:</strong> -0.0041 <span style="color:#f44336">(Target: ≥0.75)</span></li>
                <li><strong>MAPE Terbaik:</strong> ~109% <span style="color:#f44336">(Error sangat tinggi)</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Ringkasan Kinerja Model</h3>
            <p><strong>Random Forest Tuned</strong> adalah model terbaik namun masih jauh dari target:</p>
            <ul style="text-align:left">
                <li>MAE: 361.69 kkal (Target: 75)</li>
                <li>RMSE: 418.67 kkal</li>
                <li>R²: -0.0041 (negatif)</li>
                <li>MAPE: 109%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🔍 Analisis Akar Masalah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <h4>📌 Mengapa Model Gagal?</h4>
            <ul>
                <li><strong>Variansi Kalori Tinggi</strong> - Kalori berkisar 50-1500 kkal (std: 418 kkal)</li>
                <li><strong>Fitur Terbatas</strong> - Korelasi terbaik hanya ~0.3 (lemah)</li>
                <li><strong>Informasi Penting Hilang</strong> - Tidak ada ukuran porsi, metode masak, kuantitas bahan</li>
                <li><strong>Keterbatasan Fitur Teks</strong> - Teks resep tidak secara langsung menunjukkan kalori</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Korelasi fitur
        corr_data = df_featured[feature_columns + ['jumlah_kalori']].copy()
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Matriks Korelasi Fitur',
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1,
            text_auto='.2f',
            aspect='auto'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("💡 Rekomendasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <h4>📊 Untuk Data yang Lebih Baik</h4>
            <ul>
                <li>Kumpulkan informasi ukuran porsi</li>
                <li>Tambahkan kategori metode memasak</li>
                <li>Sertakan kuantitas bahan baku</li>
                <li>Tambahkan rincian nutrisi per bahan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="rec-card">
            <h4>🔧 Untuk Fitur yang Lebih Baik</h4>
            <ul>
                <li>Gunakan embedding bahan (Word2Vec/BERT)</li>
                <li>Ekstrak teknik memasak</li>
                <li>Identifikasi bahan berkalori tinggi</li>
                <li>Hitung rasio bahan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="rec-card">
            <h4>💼 Untuk Bisnis</h4>
            <ul>
                <li>Dataset ini <strong>TIDAK cocok</strong> untuk prediksi kalori</li>
                <li>Gunakan estimasi berbasis aturan</li>
                <li>Bermitra dengan database nutrisi</li>
                <li>Gunakan verifikasi manual untuk aplikasi kritis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Kesimpulan akhir
    st.markdown("""
    <div class="insight-box">
        <h3>📌 Kesimpulan Akhir</h3>
        <p>
        <strong>Semua model gagal mencapai target bisnis.</strong> Model terbaik (Random Forest Tuned) 
        mencapai Mean Absolute Error sebesar <strong style="color:#f44336">361.69 kkal</strong>, 
        hampir <strong>5 kali lebih buruk</strong> dari target 75 kkal. Nilai R² negatif menunjukkan 
        bahwa model bekerja lebih buruk daripada hanya memprediksi nilai rata-rata kalori.
        </p>
        <p>
        <strong>Masalah utamanya bukan pada pemilihan model atau tuning hyperparameter, melainkan pada dataset itu sendiri.</strong> 
        Fitur yang tersedia (teks resep, likes, usia pengguna) memiliki korelasi yang lemah dengan jumlah kalori. 
        Informasi kritis seperti ukuran porsi, kuantitas bahan spesifik, dan metode memasak tidak tersedia.
        </p>
        <p>
        <strong>Untuk prediksi kalori yang akurat, diperlukan pendekatan yang berbeda:</strong> mengumpulkan data nutrisi yang lebih detail 
        atau menggunakan sistem berbasis aturan dengan database bahan baku yang terstandardisasi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan data lengkap
    with st.expander("📋 Lihat Seluruh Data Resep"):
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total {len(df)} resep ayam")
