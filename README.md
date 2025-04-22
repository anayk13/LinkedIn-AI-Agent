# LinkedIn Content Analysis Dashboard

A comprehensive analytics tool for analyzing LinkedIn content performance and engagement patterns. This project provides insights into post performance, content trends, and creator analytics through an interactive Streamlit dashboard.

## Project Overview

This project consists of three main components:
1. **Data Collection Module** (`linkedin.py`): Scrapes LinkedIn posts and engagement data
2. **Analysis Module** (`linkedin_analysis.py`): Processes and analyzes the collected data
3. **Dashboard Interface** (`linkedin_dashboard.py`): Interactive visualization of insights

## Features

### Data Collection
- Automated LinkedIn post scraping
- Engagement metrics tracking (likes, comments, shares)
- Content type detection (text, images, videos)
- Hashtag extraction and analysis
- Sentiment analysis using BERT
- Keyword extraction using KeyBERT

### Analytics
- Post performance metrics
- Engagement rate analysis
- Content type distribution
- Hashtag effectiveness
- Sentiment analysis
- Posting frequency patterns
- Time-based engagement analysis

### Visualization Dashboard
- Interactive metrics dashboard
- Engagement trend analysis
- Content type distribution
- Hashtag cloud visualization
- Sentiment distribution
- Posting schedule heatmap
- Comparative analysis between creators

## Technical Stack

- **Web Scraping**: Selenium, BeautifulSoup
- **Data Processing**: Pandas, NumPy
- **Natural Language Processing**: BERT, KeyBERT, NLTK
- **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
- **Machine Learning**: scikit-learn

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **LinkedIn Account Requirements**:
   - You must have an active LinkedIn account
   - You need to be logged into LinkedIn in your browser
   - The account should have access to view the profiles you want to analyze

2. Configure LinkedIn credentials in `linkedin.py`:
   ```python
   LINKEDIN_USERNAME = 'your_email@example.com'  # Replace with your LinkedIn email
   LINKEDIN_PASSWORD = 'your_password'            # Replace with your LinkedIn password
   ```
   ⚠️ **Note**: These credentials must match your active LinkedIn account. The scraping will not work if:
   - The credentials are incorrect
   - You're not logged into LinkedIn
   - Your account has 2FA enabled (you may need to use a different authentication method)
   -Added a delay while logging in so that if asked for verification code it can be manually entered.

3. Run the data collection script:
   ```bash
   python linkedin.py
   ```

4. Launch the dashboard:
   ```bash
   streamlit run linkedin_dashboard.py
   ```

## Data Outputs

The project generates several output files:
- `linkedin_creators_posts.csv`: Raw post data
- `linkedin_engagement_results.csv`: Engagement metrics
- `linkedin_analysis.png`: Analysis visualizations
- `linkedin_engagement_analysis.png`: Engagement analysis charts

## Project Structure

```
├── linkedin.py                 # Data collection and scraping
├── linkedin_analysis.py        # Data analysis functions
├── linkedin_dashboard.py       # Streamlit dashboard
├── requirements.txt            # Project dependencies
├── linkedin_creators_posts.csv # Raw post data
├── linkedin_engagement_results.csv # Engagement metrics
└── linkedin_insights.txt       # Analysis insights
```

## Security Note

⚠️ **Important**: The current implementation includes hardcoded credentials in `linkedin.py`. In a production environment, these should be moved to environment variables or a secure configuration file.

