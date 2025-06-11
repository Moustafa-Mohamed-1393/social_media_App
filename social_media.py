import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import re
import string
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from math import sqrt

# Set page config
st.set_page_config(layout="wide", page_title="Social Media Engagement Analysis")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("./Social Media Engagement Dataset.csv")
    
    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)  # Set timestamp as index for time series
    df['hour'] = df.index.hour
    df['month'] = df.index.month_name()
    df['year'] = df.index.year
    df['day_of_week'] = df.index.day_name()
    df['total_engagement'] = df['likes_count'] + df['shares_count'] + df['comments_count']
    df['engagement_rate'] = df['total_engagement'] / df['impressions']
    
    # Handle missing values
    df.dropna(subset=['text_content'], inplace=True)
    numerical_cols = ['sentiment_score', 'toxicity_score', 'likes_count', 'shares_count', 'comments_count']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    categorical_cols = ['platform', 'brand_name', 'campaign_phase']
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", 
                        ["Data Overview", "Exploratory Analysis", "Modeling", "Time Series Analysis"])

# Main content
if section == "Data Overview":
    st.title("Social Media Engagement Analysis")
    st.header("Data Overview")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.subheader("Data Summary")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())
    
    with col2:
        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str))
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)

elif section == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Stats", "Engagement Analysis", "Sentiment Analysis", "Geographic Analysis"])

    with tab1:
        st.header("📈 Basic Statistics", divider="rainbow")  # Only this tab gets the rainbow divider
    
        # Section 1: Correlation Analysis
        st.subheader("🔍 Correlation Insights")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("##### 🔗 Feature Correlation Heatmap")
            with st.expander("About this visualization", expanded=True):
                st.write("""
                    This heatmap shows how different metrics in your data relate to each other.
                    - Strong positive correlation (red): Metrics that increase together
                    - Strong negative correlation (blue): Metrics that move in opposite directions
                    """)

                corr_threshold = st.slider(
                    "Minimum correlation to display:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    help="Filter out weak correlations"
                )

                corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
                mask = (abs(corr_matrix) >= corr_threshold) | (corr_matrix.isna())

                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix.where(mask), 
                        annot=True, 
                        cmap='coolwarm', 
                        fmt=".2f",
                        vmin=-1, 
                        vmax=1,
                        linewidths=0.5)
                plt.title(f"Feature Correlations (|r| ≥ {corr_threshold})")
                st.pyplot(plt)

        # Section 2: Sentiment Distribution (ONLY in Basic Stats)
        st.markdown("##### 😊 Sentiment Distribution")
        with st.expander("About this chart", expanded=True):
            st.write("""
            See how sentiment is distributed across different brands.
            Use the filters below to focus on specific brands.
            """)
            
        # Brand selector
        selected_brands = st.multiselect(
            "Select brands to display:",
            options=df['brand_name'].unique(),
            default=df['brand_name'].unique()[:3],
            key="brand_filter_tab1"
        )

        if selected_brands:
            filtered_df = df[df['brand_name'].isin(selected_brands)]
            
            # Sentiment distribution chart
            fig = px.histogram(
                filtered_df, 
                x='brand_name', 
                color='sentiment_label', 
                title='',
                labels={'brand_name': 'Brand', 'sentiment_label': 'Sentiment'},
                category_orders={'sentiment_label': ['Positive', 'Neutral', 'Negative']},
                color_discrete_map={
                    'Positive': '#4CAF50',
                    'Neutral': '#9E9E9E',
                    'Negative': '#F44336'
                }
            )
            fig.update_layout(
                barmode='stack', 
                xaxis={'categoryorder':'total descending'},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one brand")

        # Section 3: Quick Stats (ONLY in Basic Stats)
        if selected_brands and len(selected_brands) > 0:
            st.markdown("##### 📊 Quick Stats")
            pos_pct = (filtered_df['sentiment_label'] == 'Positive').mean() * 100
            neg_pct = (filtered_df['sentiment_label'] == 'Negative').mean() * 100
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric(
                    "Total Posts",
                    value=f"{len(filtered_df):,}",
                    help="Number of posts for selected brands"
                )
            with stat_col2:
                st.metric(
                    "Positive Sentiment",
                    value=f"{pos_pct:.1f}%",
                    delta=f"{(pos_pct - (df['sentiment_label'] == 'Positive').mean() * 100):.1f}% vs average",
                    delta_color="normal",
                    help="Percentage of positive posts"
                )
            with stat_col3:
                st.metric(
                    "Negative Sentiment",
                    value=f"{neg_pct:.1f}%",
                    delta=f"{(neg_pct - (df['sentiment_label'] == 'Negative').mean() * 100):.1f}% vs average",
                    delta_color="inverse",
                    help="Percentage of negative posts"
                )
    with tab2:
        st.header("📊 Engagement Analysis", divider="rainbow")
        
        # Section 1: Platform Comparison
        st.subheader("📱 Platform Performance")
        
        # Interactive controls
        platform_col1, platform_col2 = st.columns([1, 2])
        
        with platform_col1:
            metric_choice = st.selectbox(
                "Primary engagement metric:",
                ['likes_count', 'shares_count', 'comments_count', 'engagement_rate'],
                format_func=lambda x: x.replace('_', ' ').title(),
                key="platform_metric"
            )
            
            normalize = st.checkbox(
                "Normalize by post count",
                value=True,
                help="Show averages instead of totals"
            )
        
        # Prepare data
        if normalize:
            engagement_metrics = df.groupby('platform')[['likes_count', 'shares_count', 'comments_count', 'engagement_rate']].mean().reset_index()
        else:
            engagement_metrics = df.groupby('platform')[['likes_count', 'shares_count', 'comments_count', 'engagement_rate']].sum().reset_index()
        
        # Visualization
        fig = px.bar(
            engagement_metrics,
            x='platform',
            y=metric_choice,
            color='platform',
            title='',
            text_auto='.2s',
            labels={metric_choice: metric_choice.replace('_', ' ').title()},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Platform",
            yaxis_title=metric_choice.replace('_', ' ').title(),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Section 2: Campaign Analysis
        st.subheader("🎯 Campaign Performance")
        
        campaign_col1, campaign_col2 = st.columns(2)
        
        with campaign_col1:
            st.markdown("##### 📅 Engagement by Campaign Phase")
            with st.expander("About this chart"):
                st.write("Compare how engagement varies across different campaign phases")
            
            fig = px.box(
                df,
                x='campaign_phase',
                y='engagement_rate',
                color='campaign_phase',
                title='',
                labels={'engagement_rate': 'Engagement Rate', 'campaign_phase': 'Campaign Phase'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with campaign_col2:
            st.markdown("##### ⏰ Best Posting Times")
            with st.expander("About this chart"):
                st.write("See when engagement is typically highest throughout the day/week")
            
            time_metric = st.selectbox(
                "Time aggregation:",
                ['hour', 'day_of_week'],
                format_func=lambda x: "Hour of Day" if x == "hour" else "Day of Week",
                key="time_metric"
            )
            
            time_data = df.groupby(time_metric)['engagement_rate'].mean().reset_index()
            
            if time_metric == 'hour':
                category_order = list(range(24))
                x_title = "Hour of Day (24h)"
            else:
                category_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                time_data[time_metric] = pd.Categorical(time_data[time_metric], categories=category_order, ordered=True)
                x_title = "Day of Week"
            
            fig = px.line(
                time_data,
                x=time_metric,
                y='engagement_rate',
                title='',
                markers=True,
                labels={'engagement_rate': 'Avg. Engagement Rate'},
                color_discrete_sequence=['#4285F4']
            )
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title="Average Engagement Rate",
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            if time_metric == 'hour':
                fig.update_xaxes(tickvals=list(range(0, 24, 2)))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("📊 Sentiment & Brand Analysis", divider="rainbow")
        
        # Section 1: Sentiment Insights
        st.subheader("🔍 Sentiment Insights")
        
        # Row 1: Sentiment visualizations
        sent_col1, sent_col2 = st.columns([2, 1])
        
        with sent_col1:
            st.markdown("##### 🎭 Toxicity vs Sentiment Relationship")
            with st.expander("About this chart", expanded=False):
                st.write("This scatter plot shows how toxic content relates to sentiment. Hover for details!")
            
            # Interactive controls
            size_by = st.selectbox(
                "Bubble size represents:",
                ["likes_count", "comments_count", "shares_count"],
                key="bubble_size"
            )
            
            fig = px.scatter(
                df, 
                x='sentiment_score', 
                y='toxicity_score', 
                color='sentiment_label',
                size=size_by,
                title='',
                hover_data=['brand_name', 'text_content'],
                color_discrete_map={
                    'Positive': '#4CAF50',
                    'Neutral': '#9E9E9E',
                    'Negative': '#F44336'
                },
                size_max=15
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Sentiment Score",
                yaxis_title="Toxicity Score",
                hoverlabel=dict(bgcolor="white", font_size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with sent_col2:
            st.markdown("##### 😃 Emotion Distribution")
            with st.expander("About this chart", expanded=False):
                st.write("See what emotions dominate your brand conversations")
            
            # Emotion type selector
            selected_emotions = st.multiselect(
                "Filter emotions:",
                options=df['emotion_type'].unique(),
                default=df['emotion_type'].unique(),
                key="emotion_filter"
            )
            
            if selected_emotions:
                emotion_data = df[df['emotion_type'].isin(selected_emotions)]
                fig = px.pie(
                    emotion_data, 
                    names='emotion_type', 
                    title='',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}"
                )
                fig.update_layout(
                    showlegend=False, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one emotion type")
        
        # Section 2: Brand Performance (full width)
        st.subheader("🏆 Brand Performance Comparison")
        
        # Interactive controls for brand analysis
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            metric = st.selectbox(
                "Primary metric to compare:",
                ['sentiment_score', 'engagement_rate', 'likes_count'],
                format_func=lambda x: x.replace('_', ' ').title(),
                key="primary_metric"
            )
        
        with control_col2:
            compare_metric = st.selectbox(
                "Compare with:",
                [None, 'sentiment_score', 'engagement_rate', 'likes_count'],
                format_func=lambda x: "None" if x is None else x.replace('_', ' ').title(),
                key="compare_metric"
            )
        
        with control_col3:
            min_posts = st.slider(
                "Minimum posts per brand:",
                min_value=1,
                max_value=50,
                value=5,
                help="Filter out brands with few posts"
            )
        
        # Calculate and display brand metrics
        brand_counts = df['brand_name'].value_counts()
        valid_brands = brand_counts[brand_counts >= min_posts].index
        filtered_df = df[df['brand_name'].isin(valid_brands)]
        
        if not filtered_df.empty:
            brand_metrics = filtered_df.groupby('brand_name').agg({
                'sentiment_score': 'mean',
                'engagement_rate': 'mean',
                'likes_count': 'mean',
                'text_content': 'count'
            }).rename(columns={'text_content': 'post_count'}).reset_index()
            
            # Sort by selected metric
            brand_metrics = brand_metrics.sort_values(metric, ascending=False)
            
            # Create visualization based on selection
            if compare_metric:
                # Dual-axis comparison chart
                fig = go.Figure()
                
                # Primary metric (left axis)
                fig.add_trace(go.Bar(
                    x=brand_metrics['brand_name'],
                    y=brand_metrics[metric],
                    name=metric.replace('_', ' ').title(),
                    marker_color='#4285F4'
                ))
                
                # Comparison metric (right axis)
                fig.add_trace(go.Scatter(
                    x=brand_metrics['brand_name'],
                    y=brand_metrics[compare_metric],
                    name=compare_metric.replace('_', ' ').title(),
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color='#EA4335', width=3)
                ))
                
                fig.update_layout(
                    yaxis=dict(title=metric.replace('_', ' ').title()),
                    yaxis2=dict(
                        title=compare_metric.replace('_', ' ').title(),
                        overlaying='y',
                        side='right'
                    ),
                    hovermode="x unified",
                    margin=dict(l=50, r=50, t=30, b=100),
                    height=500,
                    xaxis=dict(tickangle=45)
                )
            else:
                # Single metric bar chart
                fig = px.bar(
                    brand_metrics,
                    x='brand_name',
                    y=metric,
                    color='brand_name',
                    title='',
                    text_auto='.2f',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_layout(
                    yaxis_title=metric.replace('_', ' ').title(),
                    showlegend=False,
                    margin=dict(l=50, r=50, t=30, b=100),
                    height=500,
                    xaxis=dict(tickangle=45)
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table with performance metrics
            st.markdown("##### 📋 Detailed Brand Metrics")
            st.dataframe(
                brand_metrics.style
                    .background_gradient(subset=[metric], cmap='Blues')
                    .format({
                        'sentiment_score': '{:.2f}',
                        'engagement_rate': '{:.2f}',
                        'likes_count': '{:.0f}'
                    }),
                use_container_width=True,
                height=400
            )
        else:
            st.warning(f"No brands found with at least {min_posts} posts. Try reducing the minimum posts filter.")

    with tab4:
        st.header("🌍 Geographic Analysis", divider="rainbow")
        
        # Section 1: Global Distribution
        st.subheader("🗺 Global Presence")
        
        geo_col1, geo_col2 = st.columns([3, 1])
        
        with geo_col1:
            st.markdown("##### World Map of Engagement")
            with st.expander("About this visualization", expanded=True):
                st.write("""
                This map shows where your social media engagement is coming from.
                - Darker colors indicate higher engagement
                - Hover over countries for details
                """)
            
            # Extract country and aggregate data
            df['country'] = df['location'].apply(lambda x: x.split(',')[-1].strip() if pd.notna(x) else 'Unknown')
            country_data = df.groupby('country').agg(
                total_engagement=('total_engagement', 'sum'),
                post_count=('text_content', 'count'),
                avg_sentiment=('sentiment_score', 'mean')
            ).reset_index()
            country_counts = df['country'].value_counts().reset_index()
            country_counts.columns = ['country', 'count']
            
            fig = px.choropleth(country_counts, 
                            locations='country', 
                            locationmode='country names',
                            color='count', 
                            title='Geographic Distribution of Posts',
                            color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            # Interactive controls
            min_posts = st.slider(
                "Minimum posts per country:",
                min_value=1,
                max_value=50,
                value=5,
                help="Filter out countries with few posts"
            )
            
            filtered_countries = country_data[country_data['post_count'] >= min_posts]
            
            # Create choropleth map
            fig = px.choropleth(
                filtered_countries,
                locations='country',
                locationmode='country names',
                color='total_engagement',
                hover_name='country',
                hover_data=['post_count', 'avg_sentiment'],
                title='',
                color_continuous_scale='Viridis',
                labels={
                    'total_engagement': 'Total Engagement',
                    'post_count': 'Post Count',
                    'avg_sentiment': 'Avg. Sentiment'
                }
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=500,
                coloraxis_colorbar=dict(
                    title="Engagement",
                    thicknessmode="pixels",
                    lenmode="pixels",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=0.01
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with geo_col2:
            st.markdown("##### 🌎 Top Countries")
            with st.expander("About this data", expanded=True):
                st.write("Top countries by engagement and post volume")
            
            top_countries = country_data.sort_values('total_engagement', ascending=False).head(10)
            
            # Display top countries table
            st.dataframe(
                top_countries.style
                    .background_gradient(subset=['total_engagement'], cmap='Blues')
                    .format({
                        'total_engagement': '{:,.0f}',
                        'post_count': '{:,.0f}',
                        'avg_sentiment': '{:.2f}'
                    }),
                use_container_width=True,
                height=400
            )
        
        # Section 2: Hashtag Analysis
        st.subheader("🏷 Hashtag Performance")
        
        hashtag_col1, hashtag_col2 = st.columns([1, 1])
        
        with hashtag_col1:
            st.markdown("##### 🔥 Top Hashtags")
            with st.expander("About this chart"):
                st.write("Most frequently used hashtags across all posts")
            
            # Extract hashtags
            hashtags = df['hashtags'].str.split(', ').explode().str.replace('#', '')
            hashtag_counts = hashtags.value_counts().reset_index()
            hashtag_counts.columns = ['hashtag', 'count']
            
            # Interactive control
            top_n = st.slider(
                "Number of hashtags to show:",
                min_value=5,
                max_value=50,
                value=15,
                key="top_n_hashtags"
            )
            
            fig = px.bar(
                hashtag_counts.head(top_n),
                x='hashtag',
                y='count',
                color='hashtag',
                title='',
                labels={'count': 'Frequency', 'hashtag': 'Hashtag'},
                color_discrete_sequence=px.colors.qualitative.Alphabet
            )
            fig.update_layout(
                xaxis={'categoryorder':'total descending'},
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with hashtag_col2:
            st.markdown("##### 📈 Hashtag Engagement")
            with st.expander("About this chart"):
                st.write("See which hashtags drive the most engagement")
            
            # Calculate hashtag engagement
            hashtag_engagement = df.explode('hashtags').groupby('hashtags')['total_engagement'].mean().reset_index()
            hashtag_engagement.columns = ['hashtag', 'avg_engagement']
            hashtag_engagement = hashtag_engagement.sort_values('avg_engagement', ascending=False)
            
            fig = px.bar(
                hashtag_engagement.head(top_n),
                x='hashtag',
                y='avg_engagement',
                color='hashtag',
                title='',
                labels={'avg_engagement': 'Avg. Engagement', 'hashtag': 'Hashtag'},
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig.update_layout(
                xaxis={'categoryorder':'total descending'},
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

elif section == "Modeling":
    st.header("🤖 Predictive Modeling", divider="rainbow")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Classification", "📈 Regression", "🧩 Clustering"])
    
    with tab1:
        st.subheader("Sentiment Classification")
        
        # Prepare data section with expander
        with st.expander("⚙️ Data Preparation", expanded=False):
            st.write("""
            **Features used for classification:**
            - Platform, day of week, hour
            - Sentiment score, toxicity score
            - Engagement metrics (likes, shares, comments)
            - User history metrics
            """)
            
            # Define features
            features = ['platform', 'day_of_week', 'hour', 'sentiment_score', 'toxicity_score',
                    'likes_count', 'shares_count', 'comments_count', 'impressions',
                    'engagement_rate', 'total_engagement']
            
            # Show sample of features
            if st.checkbox("Show feature samples", key="show_feature_samples"):
                st.dataframe(df[features].head(5))
        
        # Model training section
        st.markdown("##### 🏋️‍♂️ Model Training")
        train_col1, train_col2 = st.columns(2)
        
        with train_col1:
            test_size = st.slider(
                "Test set size (%)",
                min_value=10,
                max_value=40,
                value=20,
                help="Percentage of data to use for testing"
            )
        
        with train_col2:
            random_state = st.number_input(
                "Random seed",
                min_value=0,
                max_value=100,
                value=42,
                help="For reproducible results"
            )
        
        # Initialize models with progress bar
        with st.spinner("Preparing models..."):
            # Prepare data
            target_classification = 'sentiment_label'
            
            # Encode categorical variables
            label_encoders = {}
            for col in ['platform', 'day_of_week', 'sentiment_label', 'emotion_type', 'brand_name', 'campaign_phase']:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le
            
            X = df[features]
            y = df[target_classification]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
            
            # Scale numerical features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize models
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(kernel='rbf', probability=True, random_state=random_state),
                "KNN": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": xgb.XGBClassifier(),
                "Naive Bayes": GaussianNB()
            }
            
            # Train and evaluate models
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results.append({
                    "Model": name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1
                })
                
                progress_bar.progress((i + 1) / len(models))
            
            status_text.text("Training complete!")
            progress_bar.empty()
        
        # Results visualization
        st.markdown("##### 📊 Model Performance Comparison")
        
        # Interactive metric selection
        metric_options = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        selected_metrics = st.multiselect(
            "Select metrics to display:",
            options=metric_options,
            default=metric_options,
            key="class_metrics"
        )
        
        if selected_metrics:
            # Melt dataframe for visualization
            results_df = pd.DataFrame(results)
            melted_results = results_df.melt(id_vars="Model", value_vars=selected_metrics,var_name="Metric", value_name="Score")
            
            fig = px.bar(
                melted_results,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="",
                labels={"Score": "Performance", "Model": ""},
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text_auto=".2f"
            )
            fig.update_layout(
                xaxis={'categoryorder':'total descending'},
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed results table
            st.dataframe(
                results_df.style
                    .background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], cmap='Blues')
                    .format({
                        'Accuracy': '{:.2%}',
                        'Precision': '{:.2%}',
                        'Recall': '{:.2%}',
                        'F1 Score': '{:.2%}'
                    }),
                use_container_width=True
            )
        
        # Feature Importance
        st.markdown("##### 🔍 Feature Importance")
        
        model_choice = st.selectbox(
            "Select model to analyze:",
            options=list(models.keys()),
            index=0,
            key="feat_imp_model"
        )
        
        if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            # Get feature importances
            model = models[model_choice]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create visualization
            fig = px.bar(
                x=[features[i] for i in indices],
                y=importances[indices],
                color=importances[indices],
                color_continuous_scale='Blues',
                labels={'x': 'Feature', 'y': 'Importance'},
                title=f'{model_choice} Feature Importance'
            )
            fig.update_layout(
                yaxis_title="Importance Score",
                xaxis_title="Feature",
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=40, b=100),
                height=500
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Feature importance not available for {model_choice}. Try Random Forest, Gradient Boosting, or XGBoost.")
    
    with tab2:
        st.subheader("Engagement Regression")
        
        # Prepare data
        target_regression = 'total_engagement'
        y_reg = df[target_regression]
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_reg_scaled = scaler.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler.transform(X_test_reg)
        
        # Initialize regression models
        reg_models = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": xgb.XGBRegressor(),
        }
        
        # Train and evaluate regression models
        reg_results = {}
        for name, model in reg_models.items():
            model.fit(X_train_reg_scaled, y_train_reg)
            y_pred_reg = model.predict(X_test_reg_scaled)
            mse = mean_squared_error(y_test_reg, y_pred_reg)
            r2 = r2_score(y_test_reg, y_pred_reg)
            reg_results[name] = {'MSE': mse, 'R2': r2}
        
        st.write("### Regression Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### MSE Comparison")
            mse_values = [v['MSE'] for v in reg_results.values()]
            fig = plt.figure(figsize=(8, 6))
            plt.bar(reg_results.keys(), mse_values)
            plt.title("Model MSE Comparison")
            plt.ylabel("MSE")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.write("#### R2 Comparison")
            r2_values = [v['R2'] for v in reg_results.values()]
            fig = plt.figure(figsize=(8, 6))
            plt.bar(reg_results.keys(), r2_values)
            plt.title("Model R2 Comparison")
            plt.ylabel("R2 Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Post Clustering")
        
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_train_scaled)
        
        # Determine optimal number of clusters
        inertia = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_pca)
            inertia.append(kmeans.inertia_)
        
        st.write("### Elbow Method for Optimal k")
        fig = plt.figure(figsize=(8, 5))
        plt.plot(range(1, 10), inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        st.pyplot(fig)
        
        # Apply K-means clustering
        optimal_k = st.slider("Select number of clusters", 2, 6, 3)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        
        st.write("### Cluster Visualization")
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                    s=200, c='red', label='Centroids')
        plt.title('K-means Clustering of Social Media Posts')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        st.pyplot(fig)
        
        # Analyze cluster characteristics
        df_cluster = X_train.copy()
        df_cluster['cluster'] = clusters
        cluster_stats = df_cluster.groupby('cluster').mean()
        
        st.write("### Cluster Characteristics")
        st.dataframe(cluster_stats)

elif section == "Time Series Analysis":
    st.title("Time Series Analysis")
    
    # Check if index is datetime, if not try to convert or create one
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to find a datetime column if one exists
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if datetime_cols:
            # Use the first datetime column found
            df = df.set_index(pd.to_datetime(df[datetime_cols[0]]))
            df = df.sort_index()  # Sort by datetime
        else:
            # If no datetime column exists, create a dummy index (not ideal but will work)
            st.warning("No datetime column found - creating a synthetic datetime index")
            df = df.set_index(pd.date_range(start='2020-01-01', periods=len(df), freq='D'))
    
    # Create daily engagement time series
    daily_engagement = df['total_engagement'].resample('D').sum()
    daily_engagement = daily_engagement.asfreq('D').fillna(0)
    
    st.write("### Daily Engagement Over Time")
    fig = plt.figure(figsize=(12, 6))
    daily_engagement.plot()
    plt.title('Daily Total Engagement Over Time')
    plt.ylabel('Total Engagement')
    plt.xlabel('Date')
    plt.grid()
    st.pyplot(fig)
    
    # Check stationarity
    st.write("### Stationarity Check")
    st.write("Results of Dickey-Fuller Test:")
    dftest = adfuller(daily_engagement, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    st.write(dfoutput)
    
    # Seasonal decomposition
    st.write("### Seasonal Decomposition")
    try:
        result = seasonal_decompose(daily_engagement, model='additive', period=7)
        fig = result.plot()
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"Seasonal decomposition failed: {str(e)}")
    
    # Split into train and test
    train_size = int(len(daily_engagement) * 0.8)
    train, test = daily_engagement[:train_size], daily_engagement[train_size:]
    
    # Model comparison
    st.write("### Model Comparison")
    
    # ARIMA
    st.write("#### ARIMA Model")
    try:
        arima_model = ARIMA(train, order=(2,1,2))
        arima_fit = arima_model.fit()
        st.text(arima_fit.summary())
        
        arima_forecast = arima_fit.forecast(steps=len(test))
        arima_rmse = sqrt(mean_squared_error(test, arima_forecast))
        st.write(f'ARIMA RMSE: {arima_rmse:.2f}')
    except Exception as e:
        st.error(f"ARIMA modeling failed: {str(e)}")
    
    # SARIMA
    st.write("#### SARIMA Model")
    try:
        sarima_model = SARIMAX(train, 
                            order=(1,1,1), 
                            seasonal_order=(1,1,1,7))
        sarima_fit = sarima_model.fit(disp=False)
        st.text(sarima_fit.summary())
        
        sarima_forecast = sarima_fit.forecast(steps=len(test))
        sarima_rmse = sqrt(mean_squared_error(test, sarima_forecast))
        st.write(f'SARIMA RMSE: {sarima_rmse:.2f}')
    except Exception as e:
        st.error(f"SARIMA modeling failed: {str(e)}")
    
    # SARIMAX
    st.write("#### SARIMAX Model (with Exogenous Variables)")
    try:
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        daily_exog = df[['sentiment_score', 'toxicity_score', 'is_weekend']].resample('D').mean()
        daily_exog = daily_exog.asfreq('D').fillna(method='ffill')
        exog_train, exog_test = daily_exog[:train_size], daily_exog[train_size:]
        
        sarimax_model = SARIMAX(train,
                            exog=exog_train,
                            order=(1,1,1),
                            seasonal_order=(1,1,1,7),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        sarimax_fit = sarimax_model.fit(disp=False)
        st.text(sarimax_fit.summary())
        
        sarimax_forecast = sarimax_fit.forecast(steps=len(test), exog=exog_test)
        sarimax_rmse = sqrt(mean_squared_error(test, sarimax_forecast))
        st.write(f'SARIMAX RMSE: {sarimax_rmse:.2f}')
    except Exception as e:
        st.error(f"SARIMAX modeling failed: {str(e)}")
    
    # Plot forecasts
    st.write("### Forecast Comparison")
    try:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Training')
        plt.plot(test.index, test, label='Actual')
        
        if 'arima_forecast' in locals():
            plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
        if 'sarima_forecast' in locals():
            plt.plot(test.index, sarima_forecast, label='SARIMA Forecast')
        if 'sarimax_forecast' in locals():
            plt.plot(test.index, sarimax_forecast, label='SARIMAX Forecast')
            
        plt.title('Forecast Comparison')
        plt.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Forecast plotting failed: {str(e)}")
    
    # Model comparison
    models = {}
    if 'arima_rmse' in locals():
        models['ARIMA'] = arima_rmse
    if 'sarima_rmse' in locals():
        models['SARIMA'] = sarima_rmse
    if 'sarimax_rmse' in locals():
        models['SARIMAX'] = sarimax_rmse
    
    if models:
        st.write("### Model RMSE Comparison")
        fig = plt.figure(figsize=(10, 5))
        plt.bar(models.keys(), models.values())
        plt.title('Model Comparison (Lower RMSE is Better)')
        plt.ylabel('Root Mean Squared Error')
        st.pyplot(fig)
    else:
        st.warning("No models were successfully trained for comparison")
