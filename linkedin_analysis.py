import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the data
df = pd.read_csv('linkedin_creators_posts.csv')

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Engagement Analysis by Day
plt.subplot(2, 2, 1)
engagement_by_day = df.groupby('day_posted')['engagement_score'].mean().sort_values()
sns.barplot(x=engagement_by_day.index, y=engagement_by_day.values)
plt.title('Average Engagement by Day of Week')
plt.xticks(rotation=45)
plt.ylabel('Average Engagement Score')

# 2. Content Length Analysis
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='word_count', y='engagement_score')
plt.title('Engagement vs. Word Count')
plt.xlabel('Word Count')
plt.ylabel('Engagement Score')

# 3. Time of Day Analysis
plt.subplot(2, 2, 3)
time_engagement = df.groupby('time_of_day')['engagement_score'].mean().sort_values()
sns.barplot(x=time_engagement.index, y=time_engagement.values)
plt.title('Average Engagement by Time of Day')
plt.xticks(rotation=45)
plt.ylabel('Average Engagement Score')

# 4. Sentiment Analysis
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='sentiment_polarity', y='engagement_score')
plt.title('Engagement vs. Sentiment Polarity')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Engagement Score')

plt.tight_layout()
plt.savefig('linkedin_analysis.png')
plt.close()

# Print key insights
print("\n=== Key Insights ===")
print(f"\n1. Total Posts Analyzed: {len(df)}")
print(f"2. Average Engagement Score: {df['engagement_score'].mean():.2f}")
print(f"3. Most Common Tone: {df['tone'].mode()[0]}")
print(f"4. Average Word Count: {df['word_count'].mean():.1f} words")

# Analyze top performing hashtags
all_hashtags = [tag for tags in df['hashtags'].dropna() for tag in tags.split(',')]
hashtag_counts = Counter(all_hashtags)
print("\n5. Top 5 Most Used Hashtags:")
for hashtag, count in hashtag_counts.most_common(5):
    print(f"   - {hashtag.strip()}: {count} times")

# Best performing content length
word_count_ranges = pd.cut(df['word_count'], bins=[0, 50, 100, 150, 200, float('inf')])
best_length = df.groupby(word_count_ranges)['engagement_score'].mean().idxmax()
print(f"\n6. Best Performing Content Length: {best_length}")

# Save the analysis results
with open('linkedin_insights.txt', 'w') as f:
    f.write("LinkedIn Content Analysis Insights\n")
    f.write("================================\n\n")
    f.write(f"Total Posts Analyzed: {len(df)}\n")
    f.write(f"Average Engagement Score: {df['engagement_score'].mean():.2f}\n")
    f.write(f"Most Common Tone: {df['tone'].mode()[0]}\n")
    f.write(f"Average Word Count: {df['word_count'].mean():.1f} words\n\n")
    f.write("Top 5 Most Used Hashtags:\n")
    for hashtag, count in hashtag_counts.most_common(5):
        f.write(f"- {hashtag.strip()}: {count} times\n")
    f.write(f"\nBest Performing Content Length: {best_length}\n") 