"""
BioInsight Lite - Streamlit Application
Main application for data exploration and bioactivity prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import sys

# Page configuration
st.set_page_config(
    page_title="BioInsight Lite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file_path='bioactivity_data.csv', sample_size=None):
    """Load and cache dataset"""
    df = pd.read_csv(file_path)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    return df


@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    model_dir = Path('models')
    
    if model_dir.exists():
        for model_file in model_dir.glob('*.pkl'):
            if model_file.stem != 'scaler':
                try:
                    models[model_file.stem] = joblib.load(model_file)
                except Exception as e:
                    st.error(f"Error loading {model_file.stem}: {e}")
        
        # Load scaler
        scaler_path = model_dir / 'scaler.pkl'
        if scaler_path.exists():
            models['scaler'] = joblib.load(scaler_path)
        
        # Load feature names
        feature_path = model_dir / 'feature_names.json'
        if feature_path.exists():
            with open(feature_path) as f:
                models['feature_names'] = json.load(f)
    
    return models


def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 BioInsight Lite</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Chemical & Biological Data Explorer and Bioactivity Predictor</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📊 Data Explorer", "🔬 Bioactivity Predictor", 
         "📈 Model Performance", "🔍 Advanced Search"]
    )
    
    # Load data and models
    data_path = Path('bioactivity_data.csv')

    if not data_path.exists():
        st.sidebar.warning("Default dataset `bioactivity_data.csv` not found.")
        uploaded = st.sidebar.file_uploader("Upload a bioactivity CSV file", type=['csv'])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                st.stop()
        else:
            st.error("No dataset available. Upload a CSV in the sidebar to continue.")
            st.stop()
    else:
        try:
            df = load_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    try:
        models = load_models()
    except Exception as e:
        st.warning(f"Models could not be loaded: {e}")
        models = {}
    
    # Page routing
    if page == "🏠 Home":
        show_home(df, models)
    elif page == "📊 Data Explorer":
        show_data_explorer(df)
    elif page == "🔬 Bioactivity Predictor":
        show_predictor(models)
    elif page == "📈 Model Performance":
        show_model_performance(models)
    elif page == "🔍 Advanced Search":
        show_advanced_search(df)


def show_home(df, models):
    """Home page with overview"""
    st.header("Welcome to BioInsight Lite")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Compounds", f"{df['molregno'].nunique():,}" if 'molregno' in df.columns else "N/A")
    
    with col2:
        st.metric("Total Activities", f"{len(df):,}")
    
    with col3:
        st.metric("Unique Targets", f"{df['target_chembl_id'].nunique():,}" if 'target_chembl_id' in df.columns else "N/A")
    
    with col4:
        st.metric("Models Trained", len([k for k in models.keys() if k not in ['scaler', 'feature_names']]))
    
    st.markdown("---")
    
    # About section
    st.subheader("About This Application")
    st.markdown("""
    BioInsight Lite is a comprehensive platform for exploring chemical and biological datasets
    and predicting compound bioactivity using machine learning.
    
    **Key Features:**
    - 📊 **Data Explorer**: Visualize and analyze ChEMBL bioactivity data
    - 🔬 **Bioactivity Predictor**: Predict if a compound is active using ML models
    - 📈 **Model Performance**: Compare trained models and their metrics
    - 🔍 **Advanced Search**: Search compounds with flexible criteria
    
    **Dataset:** ChEMBL v36 - One of the largest publicly available databases of bioactive molecules
    """)
    
    # Quick stats
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'standard_type' in df.columns:
            fig = px.pie(
                df['standard_type'].value_counts().head(5).reset_index(),
                values='count',
                names='standard_type',
                title='Top 5 Activity Types'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'target_type' in df.columns:
            fig = px.bar(
                df['target_type'].value_counts().head(10).reset_index(),
                x='count',
                y='target_type',
                orientation='h',
                title='Top 10 Target Types'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


def show_data_explorer(df):
    """Data exploration page"""
    st.header("📊 Data Explorer")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Relationships", "Raw Data"])
    
    with tab1:
        st.subheader("Dataset Summary")
        
        # Basic statistics
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", len(df.columns))
        
        # Missing values
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if not missing_df.empty:
            fig = px.bar(
                missing_df.head(15),
                x='Percentage',
                y='Column',
                orientation='h',
                title='Top 15 Columns with Missing Values'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values in the dataset!")
    
    with tab2:
        st.subheader("Feature Distributions")
        
        # Select numerical column
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            selected_col = st.selectbox("Select feature to visualize", numerical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    df,
                    x=selected_col,
                    nbins=50,
                    title=f'Distribution of {selected_col}',
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistics
                st.write("**Statistics:**")
                stats_df = df[selected_col].describe()
                st.dataframe(stats_df)
                
                # Box plot by activity type
                if 'standard_type' in df.columns:
                    fig = px.box(
                        df,
                        x='standard_type',
                        y=selected_col,
                        title=f'{selected_col} by Activity Type'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Relationships")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis", numerical_cols, index=0)
            with col2:
                y_col = st.selectbox("Y-axis", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)
            
            # Color by categorical variable
            color_col = None
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                color_col = st.selectbox("Color by (optional)", ['None'] + categorical_cols)
                if color_col == 'None':
                    color_col = None
            
            # Scatter plot
            fig = px.scatter(
                df.sample(min(5000, len(df))),  # Sample for performance
                x=x_col,
                y=y_col,
                color=color_col,
                title=f'{x_col} vs {y_col}',
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            corr_cols = st.multiselect(
                "Select features for correlation",
                numerical_cols,
                default=numerical_cols[:min(10, len(numerical_cols))]
            )
            
            if len(corr_cols) >= 2:
                corr_matrix = df[corr_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title='Correlation Heatmap'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Raw Data View")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            n_rows = st.slider("Number of rows to display", 10, 500, 100)
        
        with col2:
            column_filter = st.multiselect(
                "Select columns",
                df.columns.tolist(),
                default=df.columns.tolist()[:10]
            )
        
        if column_filter:
            st.dataframe(df[column_filter].head(n_rows), use_container_width=True)
        else:
            st.dataframe(df.head(n_rows), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset as CSV",
            data=csv,
            file_name="bioactivity_data.csv",
            mime="text/csv"
        )


def show_predictor(models):
    """Bioactivity prediction page"""
    st.header("🔬 Bioactivity Predictor")
    
    if not models or 'feature_names' not in models:
        st.error("Models not loaded. Please train models first.")
        return
    
    st.markdown("""
    Enter compound properties to predict bioactivity. The model will classify 
    whether the compound is likely to be **Active** or **Inactive**.
    """)
    
    # Model selection
    available_models = [k for k in models.keys() if k not in ['scaler', 'feature_names']]
    
    if not available_models:
        st.error("No trained models found.")
        return
    
    selected_model = st.selectbox("Select Model", available_models)
    
    # Input form
    st.subheader("Input Compound Properties")
    
    feature_names = models['feature_names']
    
    # Create input fields dynamically
    col1, col2, col3 = st.columns(3)
    
    input_values = {}
    
    # Common features with defaults
    feature_defaults = {
        'mw_freebase': 350.0,
        'alogp': 2.5,
        'hba': 5,
        'hbd': 2,
        'psa': 75.0,
        'rtb': 5,
        'aromatic_rings': 2,
        'heavy_atoms': 25,
        'num_ro5_violations': 0,
        'mw_per_heavy_atom': 14.0,
        'hb_total': 7,
        'hba_hbd_ratio': 2.5,
        'lipinski_violations': 0
    }
    
    cols = [col1, col2, col3]
    for idx, feature in enumerate(feature_names):
        col = cols[idx % 3]
        
        with col:
            default_val = feature_defaults.get(feature, 0.0)
            
            if 'encoded' in feature:
                input_values[feature] = st.number_input(
                    feature,
                    value=int(default_val),
                    step=1
                )
            else:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(default_val),
                    step=0.1 if isinstance(default_val, float) else 1.0
                )
    
    # Prediction button
    if st.button("🔮 Predict Bioactivity", type="primary"):
        try:
            # Prepare input
            X_input = np.array([list(input_values.values())])
            
            # Get model
            model = models[selected_model]
            
            # Scale if needed
            if selected_model == 'logistic_regression' and 'scaler' in models:
                X_input = models['scaler'].transform(X_input)
            
            # Predict
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "Active" if prediction == 1 else "Inactive")
            
            with col2:
                st.metric("Confidence", f"{max(probability):.1%}")
            
            with col3:
                st.metric("Active Probability", f"{probability[1]:.1%}")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability[1] * 100,
                title={'text': "Active Probability (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for this prediction (simplified)
            st.subheader("Top Contributing Features")
            feature_contrib = pd.DataFrame({
                'Feature': feature_names,
                'Value': list(input_values.values())
            })
            st.dataframe(feature_contrib, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")


def show_model_performance(models):
    """Model performance comparison page"""
    st.header("📈 Model Performance")
    
    # Load results
    results_path = Path('models') / 'results_summary.json'
    
    if not results_path.exists():
        st.error("Model results not found. Please train models first.")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Display comparison table
    st.subheader("Model Comparison")
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.reset_index()
    comparison_df.columns = ['Model'] + list(comparison_df.columns[1:])
    
    # Style the dataframe
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, subset=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']),
        use_container_width=True
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of metrics
        metrics_df = comparison_df.melt(
            id_vars=['Model'],
            var_name='Metric',
            value_name='Score'
        )
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            color='Model',
            barmode='group',
            title='Model Performance Metrics'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar chart
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for model_name in results.keys():
            values = [results[model_name][m] for m in metrics]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Model Performance Radar Chart'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model recommendation
    st.subheader("🏆 Recommendation")
    
    best_f1_model = comparison_df.loc[comparison_df['f1_score'].idxmax(), 'Model']
    best_auc_model = comparison_df.loc[comparison_df['roc_auc'].idxmax(), 'Model']
    
    st.success(f"**Best F1-Score:** {best_f1_model}")
    st.success(f"**Best ROC-AUC:** {best_auc_model}")


def show_advanced_search(df):
    """Advanced search page with flexible criteria"""
    st.header("🔍 Advanced Search")
    
    st.markdown("""
    Search for compounds and bioactivities using multiple criteria.
    """)
    
    # Search filters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Compound Filters")
        
        compound_name = st.text_input("Compound Name (partial match)")
        
        if 'mw_freebase' in df.columns:
            mw_range = st.slider(
                "Molecular Weight Range",
                float(df['mw_freebase'].min()),
                float(df['mw_freebase'].max()),
                (float(df['mw_freebase'].min()), float(df['mw_freebase'].max()))
            )
        else:
            mw_range = None
        
        if 'alogp' in df.columns:
            alogp_range = st.slider(
                "LogP Range",
                float(df['alogp'].min()),
                float(df['alogp'].max()),
                (float(df['alogp'].min()), float(df['alogp'].max()))
            )
        else:
            alogp_range = None
    
    with col2:
        st.subheader("Activity Filters")
        
        if 'target_name' in df.columns:
            target_name = st.text_input("Target Name (partial match)")
        else:
            target_name = ""
        
        if 'standard_type' in df.columns:
            activity_types = df['standard_type'].dropna().unique().tolist()
            selected_activity = st.multiselect("Activity Type", activity_types)
        else:
            selected_activity = []
        
        if 'max_phase' in df.columns:
            max_phase = st.selectbox("Minimum Development Phase", [None, 0, 1, 2, 3, 4])
        else:
            max_phase = None
    
    # Apply filters
    filtered_df = df.copy()
    
    if compound_name:
        if 'compound_name' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['compound_name'].str.contains(compound_name, case=False, na=False)
            ]
    
    if mw_range and 'mw_freebase' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['mw_freebase'] >= mw_range[0]) &
            (filtered_df['mw_freebase'] <= mw_range[1])
        ]
    
    if alogp_range and 'alogp' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['alogp'] >= alogp_range[0]) &
            (filtered_df['alogp'] <= alogp_range[1])
        ]
    
    if target_name and 'target_name' in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df['target_name'].str.contains(target_name, case=False, na=False)
        ]
    
    if selected_activity and 'standard_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['standard_type'].isin(selected_activity)]
    
    if max_phase is not None and 'max_phase' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['max_phase'] >= max_phase]
    
    # Display results
    st.markdown("---")
    st.subheader(f"Search Results ({len(filtered_df)} matches)")
    
    if len(filtered_df) > 0:
        # Select columns to display
        display_cols = [
            'compound_chembl_id', 'compound_name', 'target_name',
            'standard_type', 'standard_value', 'pchembl_value',
            'mw_freebase', 'alogp', 'max_phase'
        ]
        
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        st.dataframe(filtered_df[available_cols].head(100), use_container_width=True)
        
        # Download results
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Search Results",
            data=csv,
            file_name="search_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No matches found. Try adjusting your search criteria.")


if __name__ == "__main__":
    main()