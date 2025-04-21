import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import re
import plotly.figure_factory as ff
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config with wider layout
st.set_page_config(
    page_title="LinkedIn Content Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with dark theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00CED1;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 16px;
        color: #E0E0E0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .insight-box {
        background-color: #1e2130;
        border-left: 4px solid #00CED1;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    div[data-testid="stExpander"] {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to convert relative time to datetime
def convert_relative_time(time_str):
    if pd.isna(time_str):
        return None
    
    time_str = time_str.replace('•', '').strip()
    match = re.match(r'(\d+)d', time_str)
    
    if match:
        days = int(match.group(1))
        return datetime.now() - timedelta(days=days)
    return None

# Enhanced data processing functions
def extract_hashtags(text):
    if pd.isna(text):
        return []
    return re.findall(r'#(\w+)', str(text))

def get_day_name(date):
    try:
        return pd.to_datetime(date).strftime('%A')
    except:
        return None

def calculate_post_frequency(dates):
    dates = pd.to_datetime(dates)
    return dates.value_counts().resample('D').count().mean()

# Load data with enhanced error handling
@st.cache_data
def load_data():
    try:
        posts_df = pd.read_csv('linkedin_creators_posts.csv')
        engagement_df = pd.read_csv('linkedin_engagement_results.csv')
        
        # Convert dates and add derived features
        posts_df['date_posted'] = posts_df['date_posted'].apply(convert_relative_time)
        posts_df['hour'] = posts_df['time_of_day'].apply(lambda x: int(x.split(':')[0]) if isinstance(x, str) and ':' in x else 0)
        posts_df['engagement_rate'] = (posts_df['likes'] + posts_df['comments'] * 2) / 100
        
        # Extract hashtags and create new features
        posts_df['hashtags'] = posts_df['content'].apply(extract_hashtags)
        posts_df['hashtag_count'] = posts_df['hashtags'].apply(len)
        
        # Add day of week
        posts_df['day_name'] = posts_df['date_posted'].apply(get_day_name)
        
        # Calculate post frequency
        posts_df['post_frequency'] = calculate_post_frequency(posts_df['date_posted'])
        
        return engagement_df, posts_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

engagement_df, posts_df = load_data()

# Custom color palette for dark theme
COLORS = ['#00CED1', '#4FD1C5', '#63B3ED', '#76E4F7', '#90CDF4']

if engagement_df is not None and posts_df is not None:
    # Dashboard Header with Description
    st.title("📊 LinkedIn Content Analytics Dashboard")
    st.markdown("""
        <div class='insight-box'>
        Analyze your LinkedIn content performance with advanced metrics and AI-powered insights.
        Track engagement patterns, optimize posting times, and improve your content strategy.
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar Filters
    st.sidebar.header("📈 Analysis Filters")
    
    # Date filter with improved UI
    if 'date_posted' in posts_df.columns and not posts_df['date_posted'].isna().all():
        min_date = posts_df['date_posted'].min()
        max_date = posts_df['date_posted'].max()
        if min_date is not None and max_date is not None:
            st.sidebar.subheader("📅 Time Period")
            date_range = st.sidebar.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
    
    # Enhanced content filters
    st.sidebar.subheader("🎯 Content Filters")
    if 'tone' in posts_df.columns:
        available_tones = posts_df['tone'].unique().tolist()
        selected_tone = st.sidebar.multiselect(
            "Content Tone",
            available_tones,
            default=available_tones[:3]
        )
    
    # Engagement threshold filter
    min_engagement = int(posts_df['engagement_score'].min())
    max_engagement = int(posts_df['engagement_score'].max())
    engagement_threshold = st.sidebar.slider(
        "Minimum Engagement Score",
        min_engagement, max_engagement,
        value=min_engagement
    )
    
    # Main content area with enhanced metrics
    st.subheader("📊 Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        avg_engagement = posts_df['engagement_score'].mean()
        st.markdown(f"<div class='metric-value'>{avg_engagement:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Average Engagement</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        best_time = posts_df.groupby('time_of_day')['engagement_score'].mean().idxmax()
        st.markdown(f"<div class='metric-value'>{best_time}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Peak Engagement Time</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        best_tone = posts_df.groupby('tone')['engagement_score'].mean().idxmax()
        st.markdown(f"<div class='metric-value'>{best_tone}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Top Performing Tone</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        optimal_length = posts_df.groupby('word_count')['engagement_score'].mean().idxmax()
        st.markdown(f"<div class='metric-value'>{optimal_length}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Optimal Word Count</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced Content Analysis Tabs
    tabs = st.tabs([
        "📈 Engagement Analysis",
        "📝 Content Analysis",
        "⏰ Timing Analysis",
        "#️⃣ Hashtag Analysis",
        "📊 Advanced Metrics"
    ])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Engagement by Content Tone")
            tone_data = posts_df.groupby('tone')['engagement_score'].agg(['mean', 'count']).reset_index()
            fig_tone = px.bar(
                tone_data,
                x='tone',
                y='mean',
                title="Average Engagement by Content Tone",
                color_discrete_sequence=COLORS,
                labels={'mean': 'Average Engagement', 'tone': 'Content Tone'}
            )
            fig_tone.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'},
                xaxis_title="Content Tone",
                yaxis_title="Average Engagement Score"
            )
            st.plotly_chart(fig_tone, use_container_width=True)
        
        with col2:
            st.subheader("Engagement Distribution")
            fig_dist = px.histogram(
                posts_df,
                x='engagement_score',
                nbins=30,
                title="Distribution of Engagement Scores",
                color_discrete_sequence=[COLORS[0]]
            )
            fig_dist.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'},
                xaxis_title="Engagement Score",
                yaxis_title="Number of Posts"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Content Length Impact")
        fig_length = px.scatter(
            posts_df,
            x='word_count',
            y='engagement_score',
            title="Engagement by Content Length",
            trendline="lowess",
            color_discrete_sequence=COLORS
        )
        fig_length.update_layout(
            plot_bgcolor='#1e2130',
            paper_bgcolor='#1e2130',
            font={'color': '#E0E0E0'},
            xaxis_title="Word Count",
            yaxis_title="Engagement Score"
        )
        st.plotly_chart(fig_length, use_container_width=True)
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Engagement Patterns")
            day_data = posts_df.groupby('day_posted')['engagement_score'].mean().reset_index()
            fig_day = px.bar(
                day_data,
                x='day_posted',
                y='engagement_score',
                title="Average Engagement by Day",
                color_discrete_sequence=COLORS
            )
            fig_day.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'},
                xaxis_title="Day of Week",
                yaxis_title="Average Engagement"
            )
            st.plotly_chart(fig_day, use_container_width=True)
        
        with col2:
            st.subheader("Hourly Engagement Heatmap")
            hour_day_data = posts_df.pivot_table(
                values='engagement_score',
                index='day_posted',
                columns='hour',
                aggfunc='mean'
            ).fillna(0)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=hour_day_data.values,
                x=hour_day_data.columns,
                y=hour_day_data.index,
                colorscale='Teal'
            ))
            fig_heatmap.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'},
                title="Engagement Heatmap by Day and Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tabs[3]:
        st.subheader("Hashtag Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hashtag frequency analysis
            all_hashtags = [tag for tags in posts_df['hashtags'] for tag in tags]
            hashtag_freq = pd.Series(all_hashtags).value_counts().head(10)
            
            fig_hashtags = px.bar(
                x=hashtag_freq.index,
                y=hashtag_freq.values,
                title="Top 10 Hashtags",
                labels={'x': 'Hashtag', 'y': 'Frequency'},
                color_discrete_sequence=COLORS
            )
            fig_hashtags.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'}
            )
            st.plotly_chart(fig_hashtags, use_container_width=True)
            
            # Hashtag impact on engagement
            avg_engagement_by_hashtags = posts_df.groupby('hashtag_count')['engagement_score'].mean()
            fig_hashtag_impact = px.line(
                x=avg_engagement_by_hashtags.index,
                y=avg_engagement_by_hashtags.values,
                title="Impact of Hashtag Count on Engagement",
                labels={'x': 'Number of Hashtags', 'y': 'Average Engagement'},
                color_discrete_sequence=COLORS
            )
            fig_hashtag_impact.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'}
            )
            st.plotly_chart(fig_hashtag_impact, use_container_width=True)
        
        with col2:
            # Top performing hashtags by engagement
            hashtag_performance = []
            for idx, row in posts_df.iterrows():
                for tag in row['hashtags']:
                    hashtag_performance.append({
                        'hashtag': tag,
                        'engagement': row['engagement_score']
                    })
            
            if hashtag_performance:
                hashtag_df = pd.DataFrame(hashtag_performance)
                top_hashtags = hashtag_df.groupby('hashtag')['engagement'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
                
                fig_top_hashtags = px.bar(
                    top_hashtags,
                    y=top_hashtags.index,
                    x='mean',
                    title="Top Performing Hashtags by Engagement",
                    labels={'mean': 'Average Engagement', 'y': 'Hashtag'},
                    color='count',
                    orientation='h',
                    color_continuous_scale='Teal'
                )
                fig_top_hashtags.update_layout(
                    plot_bgcolor='#1e2130',
                    paper_bgcolor='#1e2130',
                    font={'color': '#E0E0E0'}
                )
                st.plotly_chart(fig_top_hashtags, use_container_width=True)

    with tabs[4]:
        st.subheader("Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Posting consistency analysis
            daily_posts = posts_df['day_name'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fig_posting_pattern = px.bar(
                x=daily_posts.index,
                y=daily_posts.values,
                title="Posting Consistency by Day",
                labels={'x': 'Day of Week', 'y': 'Number of Posts'},
                color_discrete_sequence=COLORS
            )
            fig_posting_pattern.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'}
            )
            st.plotly_chart(fig_posting_pattern, use_container_width=True)
            
            # Post frequency vs engagement
            fig_freq_engage = px.scatter(
                posts_df,
                x='post_frequency',
                y='engagement_score',
                title="Post Frequency vs Engagement",
                trendline="lowess",
                labels={'post_frequency': 'Posts per Day', 'engagement_score': 'Engagement Score'},
                color_discrete_sequence=COLORS
            )
            fig_freq_engage.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'}
            )
            st.plotly_chart(fig_freq_engage, use_container_width=True)
        
        with col2:
            # Engagement metrics correlation
            engagement_metrics = [col for col in posts_df.columns if col in ['likes', 'comments', 'engagement_score', 'engagement_rate', 'hashtag_count']]
            if len(engagement_metrics) >= 2:
                correlation_data = posts_df[engagement_metrics].corr()
                fig_corr = px.imshow(
                    correlation_data,
                    title="Metrics Correlation Matrix",
                    color_continuous_scale='Teal'
                )
                fig_corr.update_layout(
                    plot_bgcolor='#1e2130',
                    paper_bgcolor='#1e2130',
                    font={'color': '#E0E0E0'}
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Engagement distribution by day
            fig_day_dist = px.box(
                posts_df,
                x='day_name',
                y='engagement_score',
                title="Engagement Distribution by Day",
                category_orders={'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
                color_discrete_sequence=COLORS
            )
            fig_day_dist.update_layout(
                plot_bgcolor='#1e2130',
                paper_bgcolor='#1e2130',
                font={'color': '#E0E0E0'}
            )
            st.plotly_chart(fig_day_dist, use_container_width=True)

    # Enhanced Insights Section
    st.header("🎯 Strategic Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("📈 Content Performance", expanded=True):
            st.markdown(f"""
                ### Key Findings
                
                1. **Best Time to Post**: {best_time}
                   - {posts_df.groupby('time_of_day')['engagement_score'].mean().sort_values(ascending=False).head(3).to_dict()}
                
                2. **Most Engaging Content Type**: {best_tone}
                   - Average engagement: {posts_df[posts_df['tone'] == best_tone]['engagement_score'].mean():.2f}
                
                3. **Optimal Content Length**: {optimal_length} words
                   - Posts between {optimal_length-50} and {optimal_length+50} words perform best
            """)
    
    with col2:
        with st.expander("🏷️ Hashtag Strategy", expanded=True):
            # Calculate optimal hashtag count
            optimal_hashtag_count = posts_df.groupby('hashtag_count')['engagement_score'].mean().idxmax()
            top_hashtags = pd.Series([tag for tags in posts_df['hashtags'] for tag in tags]).value_counts().head(5)
            
            st.markdown(f"""
                ### Hashtag Insights
                
                1. **Optimal Hashtag Count**: {optimal_hashtag_count}
                   - Posts with {optimal_hashtag_count} hashtags perform best
                
                2. **Top Performing Hashtags**:
                   {', '.join(f'#{tag}' for tag in top_hashtags.index)}
                
                3. **Hashtag Impact**:
                   - Posts with hashtags get {(posts_df[posts_df['hashtag_count'] > 0]['engagement_score'].mean() / posts_df[posts_df['hashtag_count'] == 0]['engagement_score'].mean() - 1) * 100:.1f}% more engagement
            """)
    
    with col3:
        with st.expander("📅 Posting Schedule", expanded=True):
            best_day = posts_df.groupby('day_name')['engagement_score'].mean().idxmax()
            avg_posts_per_day = posts_df['post_frequency'].mean()
            
            st.markdown(f"""
                ### Timing Strategy
                
                1. **Best Day**: {best_day}
                   - {posts_df.groupby('day_name')['engagement_score'].mean().sort_values(ascending=False).head(3).to_dict()}
                
                2. **Posting Frequency**:
                   - Current average: {avg_posts_per_day:.1f} posts/day
                   - Recommended: {max(1, min(3, round(avg_posts_per_day * 1.2)))} posts/day
                
                3. **Time Distribution**:
                   - Space posts {24 / max(1, min(3, round(avg_posts_per_day * 1.2))):.1f} hours apart
            """)
else:
    st.error("Unable to load data. Please check if the CSV files exist and are accessible.") 