# dashboard_nutrition.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nutrition Prediction Dashboard",
    page_icon="🍗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 10px;
    }
    
    /* Metric cards */
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
        font-size: 1.1rem;
    }
    
    .metric-card h2 {
        color: #1B5E20;
        font-size: 2rem;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        color: #666;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Insight box */
    .insight-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .insight-box h3 {
        color: #2E7D32;
        margin-top: 0;
        margin-bottom: 0.8rem;
    }
    
    .insight-box p {
        color: #333;
        line-height: 1.6;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #c62828;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .warning-box h3 {
        color: #c62828;
        margin-top: 0;
        margin-bottom: 0.8rem;
    }
    
    .warning-box ul {
        color: #333;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .warning-box li {
        margin: 0.5rem 0;
    }
    
    /* Success/Info/Warning badges */
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
    
    /* Target indicator */
    .target-met {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .target-not-met {
        color: #f44336;
        font-weight: bold;
    }
    
    /* Comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .comparison-table th {
        background-color: #2E7D32;
        color: white;
        padding: 0.8rem;
        text-align: center;
    }
    
    .comparison-table td {
        padding: 0.6rem;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }
    
    .comparison-table tr:hover {
        background-color: #f5f5f5;
    }
    
    /* Recommendations section */
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
    
    .rec-card ul {
        padding-left: 1.2rem;
        margin: 0;
    }
    
    .rec-card li {
        margin: 0.5rem 0;
        color: #555;
    }
    
    /* Divider */
    .custom-divider {
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #2E7D32, transparent);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f5f5f5;
    }
    
    /* Tooltip style */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 0.3rem 0.6rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        white-space: nowrap;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        # Try to load from local first
        df = pd.read_csv('cleaned_nutrition_data.csv')
        return df
    except:
        # If not found, create sample data for demo
        st.warning("Dataset not found. Using sample data for demonstration.")
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'Title': [f'Recipe {i}' for i in range(n_samples)],
            'Ingredients': ['ingredient1--ingredient2'] * n_samples,
            'Steps': ['step1--step2'] * n_samples,
            'Loves': np.random.randint(0, 1000, n_samples),
            'URL': ['/id/resep/1'] * n_samples,
            'jenis_makanan': ['ayam'] * n_samples,
            'usia': np.random.randint(18, 70, n_samples),
            'jumlah_kalori': np.random.normal(700, 200, n_samples)
        })
        df['jumlah_kalori'] = df['jumlah_kalori'].clip(100, 1500)
        return df

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

# Title
st.markdown('<div class="main-header">🍗 Chicken Recipe Calorie Prediction Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "",
        ["📊 Overview", "📈 Model Performance", "🔍 Prediction Tool", "📉 Insights & Conclusions"]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This dashboard analyzes chicken recipes to predict calorie content. "
        "The goal was to achieve **MAE ≤ 75 kkal** and **R² ≥ 0.75**."
    )
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [Dataset Info](#)")
    st.markdown("- [Model Details](#)")
    st.markdown("- [Technical Report](#)")

# Page 1: Overview
if page == "📊 Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Total Recipes</h3>
            <h2>{len(df):,}</h2>
            <p>Unique chicken recipes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Avg Calories</h3>
            <h2>{df['jumlah_kalori'].mean():.0f} kkal</h2>
            <p>per serving</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Calorie Range</h3>
            <h2>{df['jumlah_kalori'].min():.0f} - {df['jumlah_kalori'].max():.0f}</h2>
            <p>min to max</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>❤️ Avg Popularity</h3>
            <h2>{df['Loves'].mean():.1f}</h2>
            <p>average loves per recipe</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Calorie Distribution")
        fig = px.histogram(
            df, x='jumlah_kalori', 
            nbins=50, 
            title='Distribution of Recipe Calories',
            labels={'jumlah_kalori': 'Calories (kcal)', 'count': 'Number of Recipes'},
            color_discrete_sequence=['#2E7D32']
        )
        fig.add_vline(x=df['jumlah_kalori'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['jumlah_kalori'].mean():.0f}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top 10 Most Popular Recipes")
        top_recipes = df.nlargest(10, 'Loves')[['Title', 'Loves', 'jumlah_kalori']]
        fig = px.bar(
            top_recipes, 
            x='Loves', 
            y='Title',
            orientation='h',
            title='Most Loved Recipes',
            labels={'Loves': 'Number of Loves', 'Title': ''},
            color='jumlah_kalori',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    st.subheader("🔗 Feature Correlations with Calories")
    corr_data = df_featured[feature_columns + ['jumlah_kalori']].copy()
    corr_matrix = corr_data.corr()['jumlah_kalori'].sort_values(ascending=False)
    corr_df = pd.DataFrame({
        'Feature': corr_matrix.index,
        'Correlation': corr_matrix.values
    }).iloc[1:]
    
    fig = px.bar(
        corr_df,
        x='Correlation',
        y='Feature',
        orientation='h',
        title='Correlation with Calorie Content',
        color='Correlation',
        color_continuous_scale='RdYlGn',
        range_color=[-1, 1]
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# Page 2: Model Performance
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Analysis")
    
    # Model results
    model_results = {
        'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Random Forest Tuned', 'XGBoost Tuned'],
        'MAE': [362.29, 366.46, 376.53, 361.69, 362.40],
        'RMSE': [418.19, 425.52, 442.42, 418.67, 418.43],
        'R²': [-0.0018, -0.0372, -0.1212, -0.0041, -0.0030],
        'MAPE': [108.96, 109.95, 111.05, 109.03, 108.96]
    }
    results_df = pd.DataFrame(model_results)
    
    TARGET_MAE = 75
    TARGET_R2 = 0.75
    
    # Target achievement summary
    st.markdown("### 🎯 Target Achievement")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_mae = results_df.loc[results_df['MAE'].idxmin(), 'MAE']
        mae_achieved = best_mae <= TARGET_MAE
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 MAE Target</h3>
            <h2 style="color: {'#4CAF50' if mae_achieved else '#f44336'}">{best_mae:.2f} / {TARGET_MAE} kkal</h2>
            <p>{'✅ TARGET ACHIEVED' if mae_achieved else '❌ TARGET NOT ACHIEVED'}</p>
            <p>Best MAE is {abs(best_mae - TARGET_MAE):.2f} kkal above target</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_r2 = results_df.loc[results_df['R²'].idxmax(), 'R²']
        r2_achieved = best_r2 >= TARGET_R2
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 R² Target</h3>
            <h2 style="color: {'#4CAF50' if r2_achieved else '#f44336'}">{best_r2:.4f} / {TARGET_R2}</h2>
            <p>{'✅ TARGET ACHIEVED' if r2_achieved else '❌ TARGET NOT ACHIEVED'}</p>
            <p>R² is negative → worse than mean baseline</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_model = results_df.loc[results_df['R²'].idxmax(), 'Model']
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏆 Best Model</h3>
            <h2>{best_model}</h2>
            <p>R²: {results_df.loc[results_df['Model'] == best_model, 'R²'].values[0]:.4f}</p>
            <p>MAE: {results_df.loc[results_df['Model'] == best_model, 'MAE'].values[0]:.2f} kkal</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Comparison - MAE")
        fig = px.bar(
            results_df,
            x='Model',
            y='MAE',
            title='Mean Absolute Error by Model (Lower is Better)',
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
        st.subheader("📈 Model Comparison - R² Score")
        fig = px.bar(
            results_df,
            x='Model',
            y='R²',
            title='R² Score by Model (Higher is Better)',
            color='R²',
            color_continuous_scale='RdYlGn',
            range_color=[-0.2, 0.1],
            text='R²'
        )
        fig.add_hline(y=TARGET_R2, line_dash="dash", line_color="green",
                      annotation_text=f"Target: {TARGET_R2}")
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                      annotation_text="Baseline (mean)")
        fig.update_traces(textposition='outside')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Performance Metrics")
    
    # Style the dataframe
    def color_r2(val):
        if val < 0:
            return 'color: #f44336; font-weight: bold'
        elif val >= TARGET_R2:
            return 'color: #4CAF50; font-weight: bold'
        return 'color: #FF9800'
    
    def color_mae(val):
        if val <= TARGET_MAE:
            return 'color: #4CAF50; font-weight: bold'
        return 'color: #f44336'
    
    styled_df = results_df.style.format({
        'MAE': '{:.2f}',
        'RMSE': '{:.2f}',
        'R²': '{:.4f}',
        'MAPE': '{:.2f}%'
    }).applymap(color_r2, subset=['R²']).applymap(color_mae, subset=['MAE'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Performance explanation
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Performance Analysis</h3>
        <ul>
            <li><strong>Best MAE: 361.69 kkal</strong> - This is nearly <strong>5 times worse</strong> than the target of 75 kkal</li>
            <li><strong>R² is negative</strong> - Models perform worse than simply predicting the mean calorie value</li>
            <li><strong>MAPE ~109%</strong> - Predictions are off by more than 100% on average</li>
            <li><strong>All models failed</strong> to achieve business targets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Page 3: Prediction Tool
elif page == "🔍 Prediction Tool":
    st.header("🔍 Calorie Prediction Tool")
    st.markdown("Enter recipe details to predict its calorie content")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card" style="text-align:left">', unsafe_allow_html=True)
        st.subheader("📝 Recipe Information")
        title_length = st.number_input("Title Length (characters)", min_value=0, max_value=500, value=50)
        title_word_count = st.number_input("Title Word Count", min_value=0, max_value=50, value=8)
        num_ingredients = st.number_input("Number of Ingredients", min_value=0, max_value=100, value=10)
        ingredients_length = st.number_input("Ingredients Text Length", min_value=0, max_value=5000, value=200)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card" style="text-align:left">', unsafe_allow_html=True)
        st.subheader("⚙️ Metadata")
        num_steps = st.number_input("Number of Steps", min_value=0, max_value=100, value=5)
        steps_length = st.number_input("Steps Text Length", min_value=0, max_value=5000, value=150)
        loves = st.number_input("Number of Loves", min_value=0, max_value=10000, value=100)
        usia = st.number_input("User Age", min_value=1, max_value=100, value=30)
        url_length = st.number_input("URL Length", min_value=0, max_value=200, value=50)
        st.markdown('</div>', unsafe_allow_html=True)
    
    loves_usia_interaction = loves * usia
    
    # Simplified prediction model (since actual model may not be available)
    # This is a demonstration using a simplified formula
    predicted_calories = (
        300 + 
        (num_ingredients * 25) + 
        (num_steps * 20) + 
        (ingredients_length * 0.1) +
        (steps_length * 0.05) +
        (title_length * 0.5)
    )
    predicted_calories = max(100, min(1500, predicted_calories))
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🍽️ Predicted Calories</h3>
            <h2>{predicted_calories:.0f} kkal</h2>
            <p>Based on recipe features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if predicted_calories < 400:
            badge = 'badge-success'
            status = 'Low Calorie Meal'
        elif predicted_calories < 800:
            badge = 'badge-warning'
            status = 'Moderate Calorie Meal'
        else:
            badge = 'badge-danger'
            status = 'High Calorie Meal'
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏷️ Meal Category</h3>
            <h2><span class="{badge}">{status}</span></h2>
            <p>{'< 400 kcal' if predicted_calories < 400 else '< 800 kcal' if predicted_calories < 800 else '> 800 kcal'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        est_per_ingredient = predicted_calories / max(num_ingredients, 1)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Calories per Ingredient</h3>
            <h2>{est_per_ingredient:.0f} kkal</h2>
            <p>Average per ingredient</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("ℹ️ **Note:** This is a simplified prediction for demonstration. The actual trained model requires additional preprocessing.")

# Page 4: Insights & Conclusions
elif page == "📉 Insights & Conclusions":
    st.header("📉 Key Insights & Business Conclusions")
    
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 Business Question</h3>
        <p><strong>"Can a machine learning model predict chicken recipe calories with MAE ≤ 75 kkal and R² ≥ 0.75?"</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
            <h3>❌ Target NOT Achieved</h3>
            <ul>
                <li><strong>Best MAE:</strong> 361.69 kkal <span style="color:#f44336">(Target: ≤75 kkal)</span></li>
                <li><strong>Best R²:</strong> -0.0041 <span style="color:#f44336">(Target: ≥0.75)</span></li>
                <li><strong>Best MAPE:</strong> ~109% <span style="color:#f44336">(Extremely high error)</span></li>
                <li><strong>RMSE:</strong> 418.67 kkal <span style="color:#f44336">(Very high deviation)</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Model Performance Summary</h3>
            <p><strong>Random Forest Tuned</strong> was the best performer but still far from optimal:</p>
            <ul style="text-align:left; margin-top:0.5rem">
                <li>MAE: 361.69 kkal (Target: 75)</li>
                <li>RMSE: 418.67 kkal</li>
                <li>R²: -0.0041 (negative → worse than mean baseline)</li>
                <li>MAPE: 109% (more than 100% error)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    st.subheader("🔍 Root Cause Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <h4>📌 Why Did Models Fail?</h4>
            <ul>
                <li><strong>High Calorie Variance</strong> - Calories range from 50 to 1,500 kkal (std: 418 kkal)</li>
                <li><strong>Limited Predictive Features</strong> - Best correlation with calories is only ~0.3 (weak)</li>
                <li><strong>Missing Key Information</strong> - No portion sizes, cooking methods, or ingredient quantities</li>
                <li><strong>Text Features Limitations</strong> - Recipe text doesn't directly indicate calories</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Simple correlation visualization
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
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    st.subheader("💡 Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <h4>📊 For Better Data</h4>
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
            <h4>🔧 For Better Features</h4>
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
            <h4>💼 For Business</h4>
            <ul>
                <li>This dataset is <strong>not suitable</strong> for calorie prediction</li>
                <li>Consider rule-based estimation instead</li>
                <li>Partner with nutrition databases</li>
                <li>Use human verification for critical apps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # Final conclusion
    st.markdown("""
    <div class="insight-box">
        <h3>📌 Final Conclusion</h3>
        <p>
        <strong>All models failed to achieve the business targets.</strong> The best performing model (Random Forest Tuned) 
        achieved a Mean Absolute Error of <strong style="color:#f44336">361.69 kkal</strong>, which is nearly 
        <strong>5 times worse</strong> than the target of 75 kkal. The negative R² scores indicate that the models perform 
        worse than simply predicting the mean calorie value.
        </p>
        <p>
        <strong>The primary issue is not model selection or hyperparameter tuning, but rather the dataset itself.</strong> 
        The available features (recipe text, loves, user age) have weak correlations with calorie content. 
        Critical information such as portion sizes, specific ingredient quantities, and cooking methods are missing.
        </p>
        <p>
        <strong>For accurate calorie prediction, a different approach is needed:</strong> either collect more detailed nutritional data 
        or use a rule-based system with standardized ingredient databases.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key takeaway
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Key Takeaway</h3>
        <p style="font-size:1.1rem">
        <strong>This dataset is NOT suitable for calorie prediction.</strong> The features available do not contain 
        enough nutritional information to make accurate predictions. Future work should focus on collecting 
        ingredient-level nutritional data rather than improving model complexity.
        </p>
    </div>
    """, unsafe_allow_html=True)
