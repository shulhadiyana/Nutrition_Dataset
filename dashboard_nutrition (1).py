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
    page_title="Chicken Recipe Calorie Prediction Dashboard",
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
    }
    
    .stButton > button:hover {
        background-color: #1B5E20;
        color: white;
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
        st.warning("Dataset not found. Using sample data...")
        np.random.seed(42)
        n_samples = 5000
        df = pd.DataFrame({
            'Title': [f'Chicken Recipe {i}' for i in range(n_samples)],
            'Ingredients': ['ingredient1--ingredient2--ingredient3'] * n_samples,
            'Steps': ['step1--step2--step3'] * n_samples,
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

# Load data dan model
df = load_data()
df_featured = engineer_features(df)

feature_columns = [
    'usia', 'Loves', 'title_length', 'title_word_count',
    'num_ingredients', 'ingredients_length', 'num_steps', 'steps_length',
    'url_length', 'loves_usia_interaction'
]

model, scaler = load_model()

# Header
st.markdown('<div class="main-header">Chicken Recipe Calorie Prediction Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Data Overview", "Model Performance", "Calorie Prediction", "Conclusions & Recommendations"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This application analyzes chicken recipes to predict calorie content.\n\n"
        "Business Targets:\n"
        "- MAE ≤ 75 kcal\n"
        "- R² ≥ 0.75"
    )
    
    st.markdown("---")
    st.markdown("### Quick Statistics")
    st.metric("Total Recipes", f"{len(df):,}")
    st.metric("Average Calories", f"{df['jumlah_kalori'].mean():.0f} kcal")
    st.metric("Average Likes", f"{df['Loves'].mean():.1f}")

# ==================== PAGE 1: DATA OVERVIEW ====================
if page == "Data Overview":
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Recipes</h3>
            <div class="value">{len(df):,}</div>
            <div class="target">unique chicken recipes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Calories</h3>
            <div class="value">{df['jumlah_kalori'].mean():.0f} kcal</div>
            <div class="target">per serving</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Calorie Range</h3>
            <div class="value">{df['jumlah_kalori'].min():.0f} - {df['jumlah_kalori'].max():.0f}</div>
            <div class="target">min to max</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Popularity</h3>
            <div class="value">{df['Loves'].mean():.1f}</div>
            <div class="target">average likes per recipe</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calorie Distribution")
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
        st.subheader("Top 10 Most Popular Recipes")
        top_recipes = df.nlargest(10, 'Loves')[['Title', 'Loves', 'jumlah_kalori']]
        top_recipes.columns = ['Recipe Title', 'Number of Likes', 'Calories (kcal)']
        fig = px.bar(
            top_recipes, 
            x='Number of Likes', 
            y='Recipe Title',
            orientation='h',
            title='Most Loved Recipes',
            labels={'Number of Likes': 'Number of Likes', 'Recipe Title': ''},
            color='Calories (kcal)',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Tampilkan data
    with st.expander("View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing 100 out of {len(df)} rows")

# ==================== PAGE 2: MODEL PERFORMANCE ====================
elif page == "Model Performance":
    st.header("Model Performance Analysis")
    
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
            <div class="target">Based on R² score</div>
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
    st.subheader("Detailed Metrics Table")
    display_df = results_df.copy()
    display_df.columns = ['Model', 'MAE (kcal)', 'RMSE (kcal)', 'R²', 'MAPE (%)']
    st.dataframe(display_df, use_container_width=True)
    
    # Analisis
    st.markdown("""
    <div class="warning-box">
        <h3>Performance Analysis</h3>
        <ul>
            <li><strong>Best MAE: 361.69 kcal</strong> - Nearly <strong>5 times worse</strong> than the target of 75 kcal</li>
            <li><strong>R² is negative</strong> - Models perform worse than simply predicting the mean calorie value</li>
            <li><strong>MAPE ~109%</strong> - Predictions are off by more than 100% on average</li>
            <li><strong>All models failed</strong> to achieve business targets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGE 3: CALORIE PREDICTION ====================
elif page == "Calorie Prediction":
    st.header("Calorie Prediction Tool")
    st.markdown("Enter recipe information to predict calorie content")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Recipe Information")
        
        recipe_title = st.text_input("Recipe Title", placeholder="Example: Fried Chicken with Turmeric")
        
        col_a, col_b = st.columns(2)
        with col_a:
            title_length = st.number_input("Title Length (characters)", min_value=0, max_value=500, value=50)
            title_word_count = st.number_input("Title Word Count", min_value=0, max_value=50, value=8)
        with col_b:
            num_ingredients = st.number_input("Number of Ingredients", min_value=0, max_value=100, value=10)
            ingredients_length = st.number_input("Ingredients Text Length (chars)", min_value=0, max_value=5000, value=200)
        
    with col2:
        st.markdown("### Additional Information")
        
        col_c, col_d = st.columns(2)
        with col_c:
            num_steps = st.number_input("Number of Cooking Steps", min_value=0, max_value=100, value=5)
            steps_length = st.number_input("Steps Text Length (chars)", min_value=0, max_value=5000, value=150)
        with col_d:
            num_likes = st.number_input("Number of Likes", min_value=0, max_value=10000, value=100)
            user_age = st.number_input("User Age (years)", min_value=1, max_value=100, value=30)
            url_length = st.number_input("URL Length (chars)", min_value=0, max_value=200, value=50)
    
    # Fitur interaksi
    likes_age_interaction = num_likes * user_age
    
    # Tombol prediksi
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    predict_button = st.button("Predict Calories", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Calculating calorie prediction..."):
            # Buat feature array
            features = np.array([[
                user_age, num_likes, title_length, title_word_count,
                num_ingredients, ingredients_length, num_steps, steps_length,
                url_length, likes_age_interaction
            ]])
            
            # Prediksi dengan model jika ada
            if model is not None and scaler is not None:
                try:
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)[0]
                except Exception as e:
                    st.warning(f"Model error: {e}. Using simplified prediction.")
                    prediction = 300 + (num_ingredients * 25) + (num_steps * 20) + (ingredients_length * 0.1)
            else:
                # Prediksi sederhana
                prediction = 300 + (num_ingredients * 25) + (num_steps * 20) + (ingredients_length * 0.1)
            
            # Batasi range
            prediction = max(50, min(1500, prediction))
            
            # Tampilkan hasil
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("## Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Predicted Calories</h3>
                    <div class="value">{prediction:.0f} kcal</div>
                    <div class="target">per serving</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if prediction < 400:
                    status = "Low Calorie"
                    warna = "status-success"
                elif prediction < 800:
                    status = "Moderate"
                    warna = "status-success"
                else:
                    status = "High Calorie"
                    warna = "status-fail"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Category</h3>
                    <div class="{warna}" style="font-size:24px; font-weight:bold;">{status}</div>
                    <div class="target">based on calorie count</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                est_per_ingredient = prediction / max(num_ingredients, 1)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Estimate per Ingredient</h3>
                    <div class="value">{est_per_ingredient:.0f} kcal</div>
                    <div class="target">average per ingredient</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detail input
            with st.expander("Input Details Used"):
                input_data = pd.DataFrame({
                    'Feature': ['User Age', 'Number of Likes', 'Title Length', 'Title Word Count', 
                              'Number of Ingredients', 'Ingredients Text Length', 'Number of Steps', 'Steps Text Length',
                              'URL Length', 'Likes × Age Interaction'],
                    'Value': [user_age, num_likes, title_length, title_word_count,
                              num_ingredients, ingredients_length, num_steps, steps_length,
                              url_length, likes_age_interaction]
                })
                st.dataframe(input_data, use_container_width=True)
            
            # Rekomendasi
            if prediction < 400:
                st.markdown("""
                <div class="success-box">
                    <h3>Recommendation</h3>
                    <p>This recipe has low calories, suitable for diet menus or light dinners.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction < 800:
                st.markdown("""
                <div class="info-box">
                    <h3>Recommendation</h3>
                    <p>This recipe has moderate calories, suitable for lunch menus.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h3>Recommendation</h3>
                    <p>This recipe has high calories. Consume wisely or share for two servings.</p>
                </div>
                """, unsafe_allow_html=True)

# ==================== PAGE 4: CONCLUSIONS ====================
else:
    st.header("Conclusions and Recommendations")
    
    # Business question
    st.markdown("""
    <div class="info-box">
        <h3>Business Question</h3>
        <p><strong>"Can a machine learning model predict chicken recipe calories with MAE ≤ 75 kcal and R² ≥ 0.75?"</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
            <h3>Target NOT Achieved</h3>
            <ul>
                <li><strong>Best MAE:</strong> 361.69 kcal (Target: ≤ 75 kcal)</li>
                <li><strong>Best R²:</strong> -0.0018 (Target: ≥ 0.75)</li>
                <li><strong>Best MAPE:</strong> ~109% (Extremely high error)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Model Performance Summary</h3>
            <p><strong>Random Forest Tuned</strong> was the best performer but still far from optimal:</p>
            <ul style="text-align:left; margin-top:10px;">
                <li>MAE: 361.69 kcal (Target: 75)</li>
                <li>RMSE: 418.67 kcal</li>
                <li>R²: -0.0041 (negative)</li>
                <li>MAPE: 109%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    st.subheader("Root Cause Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <h4>Why Did Models Fail?</h4>
            <ul>
                <li><strong>High Calorie Variance</strong> - Calories range from 50 to 1,500 kcal (std: 418 kcal)</li>
                <li><strong>Limited Predictive Features</strong> - Best correlation with calories is only ~0.3 (weak)</li>
                <li><strong>Missing Key Information</strong> - No portion sizes, cooking methods, or ingredient quantities</li>
                <li><strong>Text Features Limitations</strong> - Recipe text doesn't directly indicate calories</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Korelasi fitur
        corr_data = df_featured[feature_columns + ['jumlah_kalori']].copy()
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1,
            text_auto='.2f',
            aspect='auto'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    st.subheader("Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <h4>For Better Data</h4>
            <ul>
                <li>Collect portion size information</li>
                <li>Add cooking method categories</li>
                <li>Include standardized ingredient quantities</li>
                <li>Add nutritional breakdown per ingredient</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="rec-card">
            <h4>For Better Features</h4>
            <ul>
                <li>Use ingredient embeddings (Word2Vec/BERT)</li>
                <li>Extract cooking techniques</li>
                <li>Identify high-calorie ingredients</li>
                <li>Calculate ingredient ratios</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="rec-card">
            <h4>For Business</h4>
            <ul>
                <li>This dataset is <strong>NOT suitable</strong> for calorie prediction</li>
                <li>Consider rule-based estimation instead</li>
                <li>Partner with nutrition databases</li>
                <li>Use human verification for critical applications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Kesimpulan akhir
    st.markdown("""
    <div class="info-box">
        <h3>Final Conclusion</h3>
        <p>
        <strong>All models failed to achieve the business targets.</strong> The best performing model (Random Forest Tuned) 
        achieved a Mean Absolute Error of <strong>361.69 kcal</strong>, which is nearly <strong>5 times worse</strong> 
        than the target of 75 kcal. The negative R² scores indicate that models perform worse than simply predicting 
        the mean calorie value.
        </p>
        <p>
        <strong>The primary issue is not model selection or hyperparameter tuning, but rather the dataset itself.</strong> 
        The available features (recipe text, likes, user age) have weak correlations with calorie content. 
        Critical information such as portion sizes, specific ingredient quantities, and cooking methods are missing.
        </p>
        <p>
        <strong>For accurate calorie prediction, a different approach is needed:</strong> either collect more detailed nutritional data 
        or use a rule-based system with standardized ingredient databases.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan data lengkap
    with st.expander("View All Recipe Data"):
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total {len(df)} chicken recipes")
