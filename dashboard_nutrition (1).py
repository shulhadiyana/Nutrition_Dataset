# dashboard_nutrition.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi Kalori Resep Ayam",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
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
            'jenis_makanan': np.random.choice(['ayam', 'daging', 'ikan'], n_samples),
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
        if hasattr(scaler, 'n_features_in_'):
            st.session_state['expected_features'] = scaler.n_features_in_
        else:
            st.session_state['expected_features'] = 11
        return model, scaler
    except:
        try:
            with open('best_calorie_model.pkl', 'rb') as f:
                model = pickle.load(f)
            scaler = None
            st.session_state['expected_features'] = 11
            return model, scaler
        except:
            st.session_state['expected_features'] = 11
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
    
    food_type_map = {'ayam': 0, 'daging': 1, 'ikan': 2, 'sayur': 3}
    df_copy['jenis_makanan_encoded'] = df_copy['jenis_makanan'].map(food_type_map).fillna(0)
    
    return df_copy

# Load data dan model
df = load_data()
df_featured = engineer_features(df)

feature_columns = [
    'usia', 'Loves', 'title_length', 'title_word_count',
    'num_ingredients', 'ingredients_length', 'num_steps', 'steps_length',
    'url_length', 'loves_usia_interaction', 'jenis_makanan_encoded'
]

model, scaler = load_model()

# Header
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🍗 Dashboard Prediksi Kalori Resep Ayam</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### Navigasi")
    page = st.radio("", ["Data Overview", "Model Performance", "Calorie Prediction", "Kesimpulan & Rekomendasi"])
    
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.info(
        "Aplikasi ini menganalisis resep ayam untuk memprediksi jumlah kalori.\n\n"
        "Target Bisnis:\n"
        "- MAE ≤ 75 kkal\n"
        "- R² ≥ 0.75"
    )
    
    st.markdown("---")
    st.markdown("### Statistik Cepat")
    st.metric("Total Resep", f"{len(df):,}")
    st.metric("Rata-rata Kalori", f"{df['jumlah_kalori'].mean():.0f} kkal")
    st.metric("Rata-rata Likes", f"{df['Loves'].mean():.1f}")

# ==================== PAGE 1: DATA OVERVIEW ====================
if page == "Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resep", f"{len(df):,}")
    col2.metric("Rata-rata Kalori", f"{df['jumlah_kalori'].mean():.0f} kkal")
    col3.metric("Rentang Kalori", f"{df['jumlah_kalori'].min():.0f} - {df['jumlah_kalori'].max():.0f}")
    col4.metric("Rata-rata Popularitas", f"{df['Loves'].mean():.1f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Kalori")
        fig = px.histogram(
            df, x='jumlah_kalori', nbins=50,
            title='Distribusi Kalori Resep',
            labels={'jumlah_kalori': 'Kalori (kkal)', 'count': 'Jumlah Resep'},
            color_discrete_sequence=['#2E7D32']
        )
        fig.add_vline(x=df['jumlah_kalori'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['jumlah_kalori'].mean():.0f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("10 Resep Paling Populer")
        top_recipes = df.nlargest(10, 'Loves')[['Title', 'Loves', 'jumlah_kalori']]
        top_recipes.columns = ['Judul Resep', 'Jumlah Likes', 'Kalori']
        fig = px.bar(
            top_recipes, x='Jumlah Likes', y='Judul Resep', orientation='h',
            title='Resep dengan Likes Terbanyak',
            color='Kalori', color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Lihat Data Mentah"):
        st.dataframe(df.head(100), use_container_width=True)

# ==================== PAGE 2: MODEL PERFORMANCE ====================
elif page == "Model Performance":
    st.header("📈 Model Performance")
    
    model_results = {
        'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Random Forest Tuned', 'XGBoost Tuned'],
        'MAE': [362.29, 366.46, 376.53, 361.69, 362.40],
        'RMSE': [418.19, 425.52, 442.42, 418.67, 418.43],
        'R2': [-0.0018, -0.0372, -0.1212, -0.0041, -0.0030],
        'MAPE': [108.96, 109.95, 111.05, 109.03, 108.96]
    }
    results_df = pd.DataFrame(model_results)
    
    TARGET_MAE = 75
    TARGET_R2 = 0.75
    
    best_mae = results_df['MAE'].min()
    best_r2 = results_df['R2'].max()
    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE Terbaik", f"{best_mae:.2f} kkal", delta=f"Target: {TARGET_MAE} kkal", delta_color="off")
    col2.metric("R² Terbaik", f"{best_r2:.4f}", delta=f"Target: {TARGET_R2}", delta_color="off")
    col3.metric("Model Terbaik", best_model_name)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(results_df, x='Model', y='MAE', title='Perbandingan MAE (Semakin Kecil Semakin Baik)',
                     color='MAE', color_continuous_scale='Reds', text='MAE')
        fig.add_hline(y=TARGET_MAE, line_dash="dash", line_color="green", annotation_text=f"Target: {TARGET_MAE}")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(results_df, x='Model', y='R2', title='Perbandingan R² (Semakin Besar Semakin Baik)',
                     color='R2', color_continuous_scale='RdYlGn', text='R2')
        fig.add_hline(y=TARGET_R2, line_dash="dash", line_color="green", annotation_text=f"Target: {TARGET_R2}")
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Baseline")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Tabel Metrik Detail")
    display_df = results_df.copy()
    display_df.columns = ['Model', 'MAE (kcal)', 'RMSE (kcal)', 'R²', 'MAPE (%)']
    st.dataframe(display_df, use_container_width=True)
    
    st.warning("""
    **Analisis Kinerja:**
    - Best MAE: 361.69 kcal (hampir 5 kali lebih buruk dari target 75 kcal)
    - R² bernilai negatif (model lebih buruk dari memprediksi nilai rata-rata)
    - MAPE ~109% (prediksi meleset lebih dari 100% rata-rata)
    - Semua model gagal mencapai target bisnis
    """)

# ==================== PAGE 3: CALORIE PREDICTION ====================
elif page == "Calorie Prediction":
    st.header("🔮 Prediksi Kalori")
    st.markdown("Masukkan informasi resep untuk memprediksi jumlah kalori")
    
    expected_features = st.session_state.get('expected_features', 11)
    st.info(f"ℹ️ Model membutuhkan {expected_features} fitur untuk prediksi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Informasi Resep")
        recipe_title = st.text_input("Judul Resep", placeholder="Contoh: Ayam Goreng Bumbu Kunyit")
        
        col_a, col_b = st.columns(2)
        with col_a:
            title_length = st.number_input("Panjang Judul (karakter)", min_value=0, max_value=500, value=50)
            title_word_count = st.number_input("Jumlah Kata Judul", min_value=0, max_value=50, value=8)
        with col_b:
            num_ingredients = st.number_input("Jumlah Bahan", min_value=0, max_value=100, value=10)
            ingredients_length = st.number_input("Panjang Teks Bahan (karakter)", min_value=0, max_value=5000, value=200)
        
    with col2:
        st.markdown("### Informasi Tambahan")
        col_c, col_d = st.columns(2)
        with col_c:
            num_steps = st.number_input("Jumlah Langkah Memasak", min_value=0, max_value=100, value=5)
            steps_length = st.number_input("Panjang Teks Langkah (karakter)", min_value=0, max_value=5000, value=150)
        with col_d:
            num_likes = st.number_input("Jumlah Likes", min_value=0, max_value=10000, value=100)
            user_age = st.number_input("Usia Pengguna (tahun)", min_value=1, max_value=100, value=30)
            url_length = st.number_input("Panjang URL (karakter)", min_value=0, max_value=200, value=50)
    
    likes_age_interaction = num_likes * user_age
    
    st.markdown("### Jenis Makanan")
    food_type = st.selectbox("Pilih Jenis Makanan", ["ayam", "daging", "ikan", "sayur"], index=0)
    food_type_mapping = {'ayam': 0, 'daging': 1, 'ikan': 2, 'sayur': 3}
    food_type_encoded = food_type_mapping.get(food_type, 0)
    
    st.markdown("---")
    predict_button = st.button("🔮 Prediksi Kalori", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Menghitung prediksi kalori..."):
            features = np.array([[
                float(user_age), float(num_likes), float(title_length), float(title_word_count),
                float(num_ingredients), float(ingredients_length), float(num_steps), float(steps_length),
                float(url_length), float(likes_age_interaction), float(food_type_encoded)
            ]])
            
            if model is not None and scaler is not None:
                try:
                    if features.shape[1] == scaler.n_features_in_:
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)[0]
                    else:
                        prediction = None
                except:
                    prediction = None
            else:
                prediction = None
            
            if prediction is None:
                prediction = 300 + (num_ingredients * 25) + (num_steps * 20) + (ingredients_length * 0.1)
                prediction = max(50, min(1500, prediction))
                st.info("Menggunakan prediksi sederhana (model utama tidak tersedia)")
            else:
                prediction = max(50, min(1500, prediction))
            
            st.markdown("---")
            st.subheader("Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediksi Kalori", f"{prediction:.0f} kkal")
            
            if prediction < 400:
                col2.metric("Kategori", "Rendah Kalori", delta="Cocok untuk diet")
            elif prediction < 800:
                col2.metric("Kategori", "Sedang", delta="Cocok untuk makan siang")
            else:
                col2.metric("Kategori", "Tinggi Kalori", delta="Konsumsi dengan bijak")
            
            est_per_ingredient = prediction / max(num_ingredients, 1)
            col3.metric("Estimasi per Bahan", f"{est_per_ingredient:.0f} kkal")
            
            with st.expander("Lihat Detail Input"):
                input_data = pd.DataFrame({
                    'Fitur': ['Usia', 'Likes', 'Panjang Judul', 'Jumlah Kata Judul', 'Jumlah Bahan',
                              'Panjang Bahan', 'Jumlah Langkah', 'Panjang Langkah', 'Panjang URL',
                              'Likes x Usia', 'Jenis Makanan'],
                    'Nilai': [user_age, num_likes, title_length, title_word_count, num_ingredients,
                              ingredients_length, num_steps, steps_length, url_length,
                              likes_age_interaction, f"{food_type} ({food_type_encoded})"]
                })
                st.dataframe(input_data, use_container_width=True)

# ==================== PAGE 4: KESIMPULAN ====================
else:
    st.header("📝 Kesimpulan & Rekomendasi")
    
    st.info("**Pertanyaan Bisnis:** Dapatkah model machine learning memprediksi kalori resep ayam dengan MAE ≤ 75 kkal dan R² ≥ 0.75?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("❌ Target TIDAK Tercapai")
        st.write("""
        - **Best MAE:** 361.69 kcal (Target: ≤ 75 kcal)
        - **Best R²:** -0.0018 (Target: ≥ 0.75)
        - **Best MAPE:** ~109% (Error sangat tinggi)
        """)
        
        st.markdown("---")
        st.subheader("📊 Perbandingan Target")
        target_df = pd.DataFrame({
            'Metrik': ['MAE', 'R²', 'MAPE'],
            'Target': ['≤ 75 kcal', '≥ 0.75', '≤ 10%'],
            'Hasil': ['361.69 kcal', '-0.0018', '~109%'],
            'Status': ['❌ Gagal', '❌ Gagal', '❌ Gagal']
        })
        st.dataframe(target_df, use_container_width=True)
    
    with col2:
        st.subheader("📊 Ringkasan Kinerja Model")
        st.write("**Random Forest Tuned** adalah model terbaik namun masih jauh dari optimal:")
        
        summary_df = pd.DataFrame({
            'Metrik': ['MAE', 'RMSE', 'R²', 'MAPE'],
            'Nilai': ['361.69 kcal', '418.67 kcal', '-0.0041', '109%'],
            'Keterangan': ['Target: 75 kcal', 'Standar deviasi tinggi', 'Negatif → buruk', 'Error > 100%']
        })
        st.dataframe(summary_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🏆 Peringkat Model")
        st.write("""
        1. **Random Forest Tuned** (R² = -0.0041)
        2. **XGBoost Tuned** (R² = -0.0030)
        3. **Linear Regression** (R² = -0.0018)
        4. **Random Forest** (R² = -0.0372)
        5. **XGBoost** (R² = -0.1212)
        """)
    
    st.markdown("---")
    st.subheader("🔍 Analisis Akar Masalah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mengapa Model Gagal?**")
        st.write("""
        1. **Variansi Kalori Tinggi** - Kalori berkisar 50-1500 kcal (std: 418 kcal)
        2. **Fitur Terbatas** - Korelasi terbaik hanya ~0.3 (lemah)
        3. **Informasi Penting Hilang** - Tidak ada ukuran porsi, metode masak, kuantitas bahan
        4. **Keterbatasan Fitur Teks** - Teks resep tidak secara langsung menunjukkan kalori
        """)
    
    with col2:
        # Korelasi fitur
        corr_data = df_featured[feature_columns + ['jumlah_kalori']].copy()
        corr_values = corr_data.corr()['jumlah_kalori'].sort_values(ascending=False)
        corr_df = pd.DataFrame({
            'Fitur': ['Usia', 'Likes', 'Panjang Judul', 'Jumlah Kata', 'Jumlah Bahan',
                      'Panjang Bahan', 'Jumlah Langkah', 'Panjang Langkah', 'Panjang URL',
                      'Likes x Usia', 'Jenis Makanan', 'Jumlah Kalori'],
            'Korelasi': corr_values.values
        })
        st.write("**Korelasi Fitur dengan Kalori:**")
        st.dataframe(corr_df.head(6), use_container_width=True)
    
    st.markdown("---")
    st.subheader("💡 Rekomendasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Untuk Data yang Lebih Baik**")
        st.write("""
        - Kumpulkan informasi ukuran porsi
        - Tambahkan kategori metode memasak
        - Sertakan kuantitas bahan baku
        - Tambahkan rincian nutrisi per bahan
        """)
    
    with col2:
        st.write("**Untuk Fitur yang Lebih Baik**")
        st.write("""
        - Gunakan embedding bahan (Word2Vec/BERT)
        - Ekstrak teknik memasak dari teks
        - Identifikasi bahan berkalori tinggi
        - Hitung rasio antar bahan
        """)
    
    with col3:
        st.write("**Untuk Bisnis**")
        st.write("""
        - Dataset ini **TIDAK COCOK** untuk prediksi kalori
        - Gunakan estimasi berbasis aturan
        - Bermitra dengan database nutrisi
        - Gunakan verifikasi manual
        """)
    
    st.markdown("---")
    
    st.subheader("📌 Kesimpulan Akhir")
    st.write("""
    **Semua model gagal mencapai target bisnis.** Model terbaik (Random Forest Tuned) 
    mencapai Mean Absolute Error sebesar **361.69 kkal**, hampir **5 kali lebih buruk** 
    dari target 75 kkal. Nilai R² yang negatif menunjukkan bahwa model lebih buruk daripada 
    hanya memprediksi nilai rata-rata kalori.
    
    **Masalah utamanya bukan pada pemilihan model atau tuning hyperparameter, melainkan pada dataset itu sendiri.** 
    Fitur yang tersedia (teks resep, jumlah likes, usia pengguna) memiliki korelasi yang lemah dengan jumlah kalori. 
    Informasi kritis seperti ukuran porsi, kuantitas bahan spesifik, dan metode memasak tidak tersedia dalam dataset.
    
    **Untuk prediksi kalori yang akurat, diperlukan pendekatan yang berbeda:** mengumpulkan data nutrisi yang lebih detail 
    atau menggunakan sistem berbasis aturan dengan database bahan baku yang terstandardisasi.
    """)
    
    with st.expander("📋 Lihat Seluruh Data Resep"):
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total {len(df)} resep ayam")
