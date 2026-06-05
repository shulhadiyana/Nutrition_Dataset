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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 20px;
        padding: 15px;
        background-color: #E8F5E9;
        border-radius: 10px;
        border: 1px solid #A5D6A7;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
        border: 1px solid #E0E0E0;
    }
    
    .metric-card h3 {
        color: #2E7D32;
        margin-bottom: 10px;
        font-size: 16px;
        font-weight: normal;
    }
    
    .metric-card .value {
        font-size: 32px;
        font-weight: bold;
        color: #1B5E20;
        margin: 10px 0;
    }
    
    .metric-card .target {
        font-size: 14px;
        color: #666666;
        margin-top: 5px;
    }
    
    .metric-card .status-success {
        color: #4CAF50;
        font-weight: bold;
        margin-top: 5px;
    }
    
    .metric-card .status-fail {
        color: #F44336;
        font-weight: bold;
        margin-top: 5px;
    }
    
    .info-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 15px 0;
    }
    
    .info-box h3 {
        color: #1565C0;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 18px;
    }
    
    .info-box p {
        color: #333333;
        line-height: 1.6;
        margin: 0;
    }
    
    .warning-box {
        background-color: #FFEBEE;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 15px 0;
    }
    
    .warning-box h3 {
        color: #C62828;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 18px;
    }
    
    .warning-box ul {
        color: #333333;
        margin: 10px 0;
        padding-left: 20px;
    }
    
    .warning-box li {
        margin: 8px 0;
    }
    
    .success-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 15px 0;
    }
    
    .success-box h3 {
        color: #2E7D32;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 18px;
    }
    
    .rec-card {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #E0E0E0;
        height: 100%;
    }
    
    .rec-card h4 {
        color: #2E7D32;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #A5D6A7;
    }
    
    .rec-card ul {
        padding-left: 20px;
        margin: 0;
    }
    
    .rec-card li {
        margin: 8px 0;
        color: #555555;
    }
    
    .divider {
        margin: 30px 0;
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #A5D6A7, transparent);
    }
    
    .stButton > button {
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1B5E20;
        color: white;
    }
    
    .feature-info {
        background-color: #FFF3E0;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 12px;
        border-left: 3px solid #FF9800;
    }
    
    .page-subtitle {
        font-size: 20px;
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #A5D6A7;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Encode jenis_makanan
    food_type_map = {'ayam': 0, 'daging': 1, 'ikan': 2, 'sayur': 3}
    df_copy['jenis_makanan_encoded'] = df_copy['jenis_makanan'].map(food_type_map).fillna(0)
    
    return df_copy

# Load data dan model
df = load_data()
df_featured = engineer_features(df)

# Daftar fitur (11 fitur)
feature_columns = [
    'usia',
    'Loves', 
    'title_length', 
    'title_word_count',
    'num_ingredients', 
    'ingredients_length', 
    'num_steps', 
    'steps_length',
    'url_length', 
    'loves_usia_interaction',
    'jenis_makanan_encoded'
]

model, scaler = load_model()

# Header
st.markdown('<div class="main-header">🍗 Dashboard Prediksi Kalori Resep Ayam</div>', 
            unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Sidebar - Menggunakan Bahasa Indonesia
with st.sidebar:
    st.markdown("### Navigasi")
    page = st.radio(
        "",
        ["Data Overview", "Model Performance", "Calorie Prediction", "Kesimpulan & Rekomendasi"]
    )
    
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
    
    if model is not None:
        st.markdown("---")
        st.markdown("### Info Model")
        st.caption(f"Jumlah fitur: {st.session_state.get('expected_features', 11)}")

# ==================== PAGE 1: DATA OVERVIEW ====================
if page == "Data Overview":
    st.markdown('<div class="page-subtitle">📊 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Resep</h3>
            <div class="value">{len(df):,}</div>
            <div class="target">resep ayam unik</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Rata-rata Kalori</h3>
            <div class="value">{df['jumlah_kalori'].mean():.0f} kkal</div>
            <div class="target">per porsi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Rentang Kalori</h3>
            <div class="value">{df['jumlah_kalori'].min():.0f} - {df['jumlah_kalori'].max():.0f}</div>
            <div class="target">minimum ke maksimum</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Rata-rata Popularitas</h3>
            <div class="value">{df['Loves'].mean():.1f}</div>
            <div class="target">jumlah likes per resep</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Kalori")
        fig = px.histogram(
            df, x='jumlah_kalori', 
            nbins=50, 
            title='Distribution of Recipe Calories',
            labels={'jumlah_kalori': 'Calories (kcal)', 'count': 'Number of Recipes'},
            color_discrete_sequence=['#2E7D32']
        )
        fig.add_vline(x=df['jumlah_kalori'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['jumlah_kalori'].mean():.0f}")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("10 Resep Paling Populer")
        top_recipes = df.nlargest(10, 'Loves')[['Title', 'Loves', 'jumlah_kalori']]
        top_recipes.columns = ['Judul Resep', 'Jumlah Likes', 'Kalori (kkal)']
        fig = px.bar(
            top_recipes, 
            x='Jumlah Likes', 
            y='Judul Resep',
            orientation='h',
            title='Most Loved Recipes',
            labels={'Jumlah Likes': 'Number of Likes', 'Judul Resep': ''},
            color='Kalori (kkal)',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Feature correlation
    st.subheader("Korelasi Fitur dengan Kalori")
    corr_data = df_featured[feature_columns + ['jumlah_kalori']].copy()
    corr_matrix = corr_data.corr()['jumlah_kalori'].sort_values(ascending=False)
    corr_df = pd.DataFrame({
        'Feature': corr_matrix.index,
        'Correlation': corr_matrix.values
    }).iloc[1:]
    
    # Rename features untuk tampilan yang lebih baik
    feature_names_id = {
        'usia': 'Age',
        'Loves': 'Likes',
        'title_length': 'Title Length',
        'title_word_count': 'Title Word Count',
        'num_ingredients': 'Number of Ingredients',
        'ingredients_length': 'Ingredients Length',
        'num_steps': 'Number of Steps',
        'steps_length': 'Steps Length',
        'url_length': 'URL Length',
        'loves_usia_interaction': 'Likes × Age',
        'jenis_makanan_encoded': 'Food Type'
    }
    corr_df['Feature'] = corr_df['Feature'].map(feature_names_id).fillna(corr_df['Feature'])
    
    fig = px.bar(
        corr_df,
        x='Correlation',
        y='Feature',
        orientation='h',
        title='Correlation with Calorie Content',
        color='Correlation',
        color_continuous_scale='RdBu',
        range_color=[-0.5, 0.5]
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan data
    with st.expander("Lihat Data Mentah"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Menampilkan 100 dari {len(df)} baris data")

# ==================== PAGE 2: MODEL PERFORMANCE ====================
elif page == "Model Performance":
    st.markdown('<div class="page-subtitle">📈 Model Performance</div>', unsafe_allow_html=True)
    
    # Hasil model
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
    
    best_mae = results_df.loc[results_df['MAE'].idxmin(), 'MAE']
    best_r2 = results_df.loc[results_df['R2'].idxmax(), 'R2']
    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    
    # Ringkasan pencapaian target
    st.markdown("### Business Target Achievement")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_mae = "FAILED" if best_mae > TARGET_MAE else "ACHIEVED"
        warna_mae = "status-fail" if best_mae > TARGET_MAE else "status-success"
        st.markdown(f"""
        <div class="metric-card">
            <h3>MAE Target</h3>
            <div class="value">{best_mae:.2f} kcal</div>
            <div class="target">Target: ≤ {TARGET_MAE} kcal</div>
            <div class="{warna_mae}">Status: {status_mae}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status_r2 = "FAILED" if best_r2 < TARGET_R2 else "ACHIEVED"
        warna_r2 = "status-fail" if best_r2 < TARGET_R2 else "status-success"
        st.markdown(f"""
        <div class="metric-card">
            <h3>R² Target</h3>
            <div class="value">{best_r2:.4f}</div>
            <div class="target">Target: ≥ {TARGET_R2}</div>
            <div class="{warna_r2}">Status: {status_r2}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Best Model</h3>
            <div class="value">{best_model_name}</div>
            <div class="target">Berdasarkan R² score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MAE Comparison (Lower is Better)")
        fig = px.bar(
            results_df,
            x='Model',
            y='MAE',
            title='Mean Absolute Error by Model',
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
        st.subheader("R² Comparison (Higher is Better)")
        fig = px.bar(
            results_df,
            x='Model',
            y='R2',
            title='R² Score by Model',
            color='R2',
            color_continuous_scale='RdYlGn',
            range_color=[-0.2, 0.1],
            text='R2'
        )
        fig.add_hline(y=TARGET_R2, line_dash="dash", line_color="green",
                      annotation_text=f"Target: {TARGET_R2}")
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                      annotation_text="Baseline (predicting mean)")
        fig.update_traces(textposition='outside')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Tabel metrik
    st.subheader("Tabel Metrik Detail")
    display_df = results_df.copy()
    display_df.columns = ['Model', 'MAE (kcal)', 'RMSE (kcal)', 'R²', 'MAPE (%)']
    st.dataframe(display_df, use_container_width=True)
    
    # Analisis
    st.markdown("""
    <div class="warning-box">
        <h3>Analisis Kinerja</h3>
        <ul>
            <li><strong>Best MAE: 361.69 kcal</strong> - Hampir <strong>5 kali lebih buruk</strong> dari target 75 kcal</li>
            <li><strong>R² bernilai negatif</strong> - Model lebih buruk dari memprediksi nilai rata-rata kalori</li>
            <li><strong>MAPE ~109%</strong> - Prediksi meleset lebih dari 100% rata-rata</li>
            <li><strong>Semua model gagal</strong> mencapai target bisnis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGE 3: CALORIE PREDICTION ====================
elif page == "Calorie Prediction":
    st.markdown('<div class="page-subtitle">🔮 Calorie Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown("Masukkan informasi resep untuk memprediksi jumlah kalori")
    
    # Informasi fitur yang dibutuhkan
    expected_features = st.session_state.get('expected_features', 11)
    st.markdown(f"""
    <div class="feature-info">
        <small>ℹ️ Model membutuhkan <strong>{expected_features} fitur</strong> untuk prediksi</small>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Fitur interaksi
    likes_age_interaction = num_likes * user_age
    
    # Food type selection (fitur ke-11)
    st.markdown("### Jenis Makanan")
    food_type = st.selectbox(
        "Pilih Jenis Makanan", 
        ["ayam", "daging", "ikan", "sayur"], 
        index=0,
        help="Jenis makanan utama dalam resep"
    )
    
    # Encoding untuk jenis_makanan
    food_type_mapping = {
        'ayam': 0,
        'daging': 1,
        'ikan': 2,
        'sayur': 3
    }
    food_type_encoded = food_type_mapping.get(food_type, 0)
    
    # Tombol prediksi
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    predict_button = st.button("🔮 Prediksi Kalori", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Menghitung prediksi kalori..."):
            
            # Buat feature array dengan 11 fitur
            features = np.array([[
                float(user_age),                    # usia
                float(num_likes),                   # Loves
                float(title_length),                # title_length
                float(title_word_count),            # title_word_count
                float(num_ingredients),             # num_ingredients
                float(ingredients_length),          # ingredients_length
                float(num_steps),                   # num_steps
                float(steps_length),                # steps_length
                float(url_length),                  # url_length
                float(likes_age_interaction),       # loves_usia_interaction
                float(food_type_encoded)            # jenis_makanan_encoded
            ]])
            
            # Prediksi dengan model jika ada
            if model is not None and scaler is not None:
                try:
                    if features.shape[1] == scaler.n_features_in_:
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)[0]
                    else:
                        st.error(f"Jumlah fitur tidak sesuai: {features.shape[1]} fitur diberikan, {scaler.n_features_in_} fitur diharapkan")
                        prediction = None
                except Exception as e:
                    st.warning(f"Error model: {str(e)[:200]}. Menggunakan prediksi sederhana.")
                    prediction = None
            else:
                prediction = None
            
            # Fallback jika prediksi gagal
            if prediction is None:
                # Prediksi sederhana berdasarkan bahan dan langkah
                prediction = 300 + (num_ingredients * 25) + (num_steps * 20) + (ingredients_length * 0.1)
                prediction = max(50, min(1500, prediction))
                st.info("Menggunakan model prediksi sederhana karena model utama tidak tersedia.")
            else:
                # Batasi range
                prediction = max(50, min(1500, prediction))
            
            # Tampilkan hasil
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("## Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Prediksi Kalori</h3>
                    <div class="value">{prediction:.0f} kkal</div>
                    <div class="target">per porsi</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if prediction < 400:
                    status = "Rendah Kalori"
                    warna = "status-success"
                    message = "Cocok untuk diet"
                elif prediction < 800:
                    status = "Sedang"
                    warna = "status-success"
                    message = "Cocok untuk makan siang"
                else:
                    status = "Tinggi Kalori"
                    warna = "status-fail"
                    message = "Konsumsi dengan bijak"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Kategori</h3>
                    <div class="{warna}" style="font-size:24px; font-weight:bold;">{status}</div>
                    <div class="target">{message}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                est_per_ingredient = prediction / max(num_ingredients, 1)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Estimasi per Bahan</h3>
                    <div class="value">{est_per_ingredient:.0f} kkal</div>
                    <div class="target">rata-rata per bahan</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detail input
            with st.expander("Lihat Detail Input"):
                input_data = pd.DataFrame({
                    'Fitur': [
                        'Usia Pengguna', 
                        'Jumlah Likes', 
                        'Panjang Judul', 
                        'Jumlah Kata Judul', 
                        'Jumlah Bahan', 
                        'Panjang Teks Bahan', 
                        'Jumlah Langkah', 
                        'Panjang Teks Langkah',
                        'Panjang URL', 
                        'Interaksi Likes x Usia', 
                        'Jenis Makanan'
                    ],
                    'Nilai': [
                        user_age, 
                        num_likes, 
                        title_length, 
                        title_word_count,
                        num_ingredients, 
                        ingredients_length, 
                        num_steps, 
                        steps_length,
                        url_length, 
                        likes_age_interaction, 
                        f"{food_type} ({food_type_encoded})"
                    ]
                })
                st.dataframe(input_data, use_container_width=True)
            
            # Rekomendasi
            if prediction < 400:
                st.markdown("""
                <div class="success-box">
                    <h3>Rekomendasi</h3>
                    <p>Resep ini memiliki kalori rendah, cocok untuk menu diet atau makan malam ringan.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction < 800:
                st.markdown("""
                <div class="info-box">
                    <h3>Rekomendasi</h3>
                    <p>Resep ini memiliki kalori sedang, cocok untuk menu makan siang.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h3>Rekomendasi</h3>
                    <p>Resep ini memiliki kalori tinggi. Konsumsi dengan bijak atau bagi untuk dua porsi.</p>
                </div>
                """, unsafe_allow_html=True)

# ==================== PAGE 4: KESIMPULAN ====================
else:
    st.markdown('<div class="page-subtitle">📝 Kesimpulan & Rekomendasi</div>', unsafe_allow_html=True)
    
    # Business question
    st.markdown("""
    <div class="info-box">
        <h3>Pertanyaan Bisnis</h3>
        <p><strong>"Dapatkah model machine learning memprediksi kalori resep ayam dengan MAE ≤ 75 kkal dan R² ≥ 0.75?"</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
            <h3>Target TIDAK Tercapai</h3>
            <ul>
                <li><strong>Best MAE:</strong> 361.69 kcal (Target: ≤ 75 kcal)</li>
                <li><strong>Best R²:</strong> -0.0018 (Target: ≥ 0.75)</li>
                <li><strong>Best MAPE:</strong> ~109% (Error sangat tinggi)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Ringkasan Kinerja Model</h3>
            <p><strong>Random Forest Tuned</strong> adalah model terbaik namun masih jauh dari optimal:</p>
            <ul style="text-align:left; margin-top:10px;">
                <li>MAE: 361.69 kcal (Target: 75)</li>
                <li>RMSE: 418.67 kcal</li>
                <li>R²: -0.0041 (negatif)</li>
                <li>MAPE: 109%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    st.subheader("Analisis Akar Masalah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <h4>Mengapa Model Gagal?</h4>
            <ul>
                <li><strong>Variansi Kalori Tinggi</strong> - Kalori berkisar 50-1500 kcal (std: 418 kcal)</li>
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
        
        # Rename untuk tampilan lebih baik
        corr_matrix = corr_matrix.rename(columns=feature_names_id, index=feature_names_id)
        
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
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    st.subheader("Rekomendasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <h4>Untuk Data yang Lebih Baik</h4>
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
            <h4>Untuk Fitur yang Lebih Baik</h4>
            <ul>
                <li>Gunakan embedding bahan (Word2Vec/BERT)</li>
                <li>Ekstrak teknik memasak dari teks</li>
                <li>Identifikasi bahan berkalori tinggi</li>
                <li>Hitung rasio antar bahan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="rec-card">
            <h4>Untuk Bisnis</h4>
            <ul>
                <li>Dataset ini <strong>TIDAK COCOK</strong> untuk prediksi kalori</li>
                <li>Gunakan estimasi berbasis aturan</li>
                <li>Bermitra dengan database nutrisi</li>
                <li>Gunakan verifikasi manual untuk aplikasi kritis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Kesimpulan akhir
    st.markdown("""
    <div class="info-box">
        <h3>Kesimpulan Akhir</h3>
        <p>
        <strong>Semua model gagal mencapai target bisnis.</strong> Model terbaik (Random Forest Tuned) 
        mencapai Mean Absolute Error sebesar <strong>361.69 kkal</strong>, hampir <strong>5 kali lebih buruk</strong> 
        dari target 75 kkal. Nilai R² yang negatif menunjukkan bahwa model lebih buruk daripada hanya 
        memprediksi nilai rata-rata kalori.
        </p>
        <p>
        <strong>Masalah utamanya bukan pada pemilihan model atau tuning hyperparameter, melainkan pada dataset itu sendiri.</strong> 
        Fitur yang tersedia (teks resep, jumlah likes, usia pengguna) memiliki korelasi yang lemah dengan jumlah kalori. 
        Informasi kritis seperti ukuran porsi, kuantitas bahan spesifik, dan metode memasak tidak tersedia dalam dataset.
        </p>
        <p>
        <strong>Untuk prediksi kalori yang akurat, diperlukan pendekatan yang berbeda:</strong> mengumpulkan data nutrisi yang lebih detail 
        atau menggunakan sistem berbasis aturan dengan database bahan baku yang terstandardisasi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan data lengkap
    with st.expander("Lihat Seluruh Data Resep"):
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total {len(df)} resep ayam")
