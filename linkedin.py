from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
import re
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timedelta
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from keybert import KeyBERT
from textblob import TextBlob
import torch
from collections import Counter

LINKEDIN_USERNAME = 'madethisjustforcursorai@gmail.com'
LINKEDIN_PASSWORD = 'hotcoffee'

# Initialize BERT and keyword models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

chrome_options = Options()
# chrome_options.add_argument('--headless')  # Uncomment if you want to run headless
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('--start-maximized')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

creators = {
    "Archit Anand": "https://www.linkedin.com/in/archit-anand/",
    "Aaron Golbin": "https://www.linkedin.com/in/aarongolbin/",
    "Jaspar Carmichael-Jack": "https://www.linkedin.com/in/jaspar-carmichael-jack/"
}

data = []

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return [kw[0] for kw in keywords[:5]]

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def categorize_tone(text):
    tone_categories = {
        'Inspiring': ['inspire', 'motivate', 'empower', 'transform', 'change', 'impact'],
        'Educational': ['learn', 'teach', 'explain', 'understand', 'knowledge', 'insight'],
        'Promotional': ['offer', 'deal', 'discount', 'limited', 'exclusive', 'sale'],
        'Personal': ['I', 'me', 'my', 'experience', 'journey', 'story'],
        'Professional': ['business', 'industry', 'market', 'strategy', 'growth', 'success']
    }
    text_lower = text.lower()
    tone_scores = {category: sum(1 for word in keywords if word in text_lower)
                   for category, keywords in tone_categories.items()}
    return max(tone_scores.items(), key=lambda x: x[1])[0] if any(tone_scores.values()) else 'Neutral'

def normalize_engagement(likes, comments):
    likes = int(re.sub(r'[^\d]', '', str(likes)) or 0)
    comments = int(re.sub(r'[^\d]', '', str(comments)) or 0)
    engagement_score = (likes * 1.0) + (comments * 2.0)
    normalized_score = min((engagement_score / 1000) * 100, 100)
    return normalized_score

def get_time_features(date_text):
    try:
        # First try the standard relative time formats
        if 'd' in date_text.lower():
            days = int(re.search(r'(\d+)d', date_text.lower()).group(1))
            date = datetime.now() - timedelta(days=days)
        elif 'h' in date_text.lower():
            hours = int(re.search(r'(\d+)h', date_text.lower()).group(1))
            date = datetime.now() - timedelta(hours=hours)
        elif 'm' in date_text.lower():
            minutes = int(re.search(r'(\d+)m', date_text.lower()).group(1))
            date = datetime.now() - timedelta(minutes=minutes)
        elif 'w' in date_text.lower():
            weeks = int(re.search(r'(\d+)w', date_text.lower()).group(1))
            date = datetime.now() - timedelta(weeks=weeks)
        # Try to handle "X days/hours/minutes/weeks ago" format
        elif 'ago' in date_text.lower():
            if 'day' in date_text.lower():
                days = int(re.search(r'(\d+)\s*day', date_text.lower()).group(1))
                date = datetime.now() - timedelta(days=days)
            elif 'hour' in date_text.lower():
                hours = int(re.search(r'(\d+)\s*hour', date_text.lower()).group(1))
                date = datetime.now() - timedelta(hours=hours)
            elif 'minute' in date_text.lower():
                minutes = int(re.search(r'(\d+)\s*minute', date_text.lower()).group(1))
                date = datetime.now() - timedelta(minutes=minutes)
            elif 'week' in date_text.lower():
                weeks = int(re.search(r'(\d+)\s*week', date_text.lower()).group(1))
                date = datetime.now() - timedelta(weeks=weeks)
            else:
                return None, None
        else:
            return None, None

        day_posted = date.strftime('%A')
        hour = date.hour
        if 5 <= hour < 12:
            time_of_day = 'Morning'
        elif 12 <= hour < 17:
            time_of_day = 'Afternoon'
        elif 17 <= hour < 22:
            time_of_day = 'Evening'
        else:
            time_of_day = 'Night'
        return day_posted, time_of_day
    except Exception as e:
        print(f"Error parsing date text: {date_text} - {str(e)}")
        return None, None

def login_to_linkedin():
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)
    driver.find_element(By.ID, "username").send_keys(LINKEDIN_USERNAME)
    driver.find_element(By.ID, "password").send_keys(LINKEDIN_PASSWORD)
    driver.find_element(By.ID, "password").send_keys(Keys.RETURN)
    time.sleep(5)

def scroll_until_end(max_scrolls=20):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(6)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def extract_hashtags(text):
    hashtags = re.findall(r"#\w+", text)
    cleaned_hashtags = [re.sub(r'hashtag', '', tag, flags=re.IGNORECASE).strip() for tag in hashtags]
    return ", ".join(cleaned_hashtags)

def detect_media_type(div):
    if div.find("img"):
        return "Image"
    elif div.find("video"):
        return "Video"
    else:
        return "Text"

def is_reshared_post(parent_div):
    if parent_div:
        class_list = parent_div.get("class", [])
        if "feed-shared-update-v2--is-reshared" in class_list:
            return True
        if "reposted this" in parent_div.get_text(strip=True).lower():
            return True
    return False

def extract_posts(profile_url):
    print(f"Scraping posts for {profile_url}")
    driver.get(profile_url + "recent-activity/shares/")
    time.sleep(5)
    scroll_until_end(max_scrolls=10)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    post_containers = soup.find_all("div", class_="update-components-text")
    post_data = []

    for container in post_containers[:5]:
        content = container.get_text(strip=True)
        parent_div = container.find_parent("div", class_="feed-shared-update-v2")
        if not parent_div:
            continue

        is_reshare = is_reshared_post(parent_div)
        link_tag = parent_div.find("a", href=True)
        post_url = "https://www.linkedin.com" + link_tag['href'] if link_tag else ""

        reactions = parent_div.find("span", class_="social-details-social-counts__reactions-count")
        comments = parent_div.find("span", class_="social-details-social-counts__comments")

        # Try multiple selectors for date element
        date_text = ""
        date_selectors = [
            "span.feed-shared-actor__sub-description span",
            "span.update-components-actor__sub-description span",
            "span.feed-shared-actor__sub-description time",
            "time.feed-shared-actor__sub-description",
            "span.feed-shared-update-v2__creation-time"
        ]
        
        for selector in date_selectors:
            date_elems = parent_div.select(selector)
            for elem in date_elems:
                text = elem.get_text(strip=True)
                if any(char.isdigit() for char in text):
                    date_text = text
                    break
            if date_text:
                break

        media_type = detect_media_type(parent_div)
        hashtags = extract_hashtags(content)

        word_count = len(content.split())
        char_count = len(content)
        num_hashtags = len(re.findall(r"#\w+", content))
        day_posted, time_of_day = get_time_features(date_text)

        engagement_score = normalize_engagement(
            reactions.get_text(strip=True) if reactions else "0",
            comments.get_text(strip=True) if comments else "0"
        )

        keywords = extract_keywords(content)
        sentiment_polarity, sentiment_subjectivity = analyze_sentiment(content)
        tone = categorize_tone(content)
        embeddings = get_bert_embeddings(content)

        post_data.append({
            "post_url": post_url,
            "content": content,
            "likes": reactions.get_text(strip=True) if reactions else "0",
            "comments": comments.get_text(strip=True) if comments else "0",
            "date_posted": date_text,
            "media_type": media_type,
            "hashtags": hashtags,
            "word_count": word_count,
            "char_count": char_count,
            "num_hashtags": num_hashtags,
            "day_posted": day_posted,
            "time_of_day": time_of_day,
            "engagement_score": engagement_score,
            "keywords": ", ".join(keywords),
            "sentiment_polarity": sentiment_polarity,
            "sentiment_subjectivity": sentiment_subjectivity,
            "tone": tone,
            "bert_embeddings": embeddings.tolist(),
            "is_reshare": is_reshare
        })

    return post_data

# Main driver
login_to_linkedin()

for name, profile_url in creators.items():
    posts = extract_posts(profile_url)
    for post in posts:
        data.append({
            "creator": name,
            "post_url": post["post_url"],
            "content": post["content"],
            "likes": post["likes"],
            "comments": post["comments"],
            "date_posted": post["date_posted"],
            "media_type": post["media_type"],
            "hashtags": post["hashtags"],
            "word_count": post["word_count"],
            "char_count": post["char_count"],
            "num_hashtags": post["num_hashtags"],
            "day_posted": post["day_posted"],
            "time_of_day": post["time_of_day"],
            "engagement_score": post["engagement_score"],
            "keywords": post["keywords"],
            "sentiment_polarity": post["sentiment_polarity"],
            "sentiment_subjectivity": post["sentiment_subjectivity"],
            "tone": post["tone"],
            "bert_embeddings": post["bert_embeddings"],
            "is_reshare": post["is_reshare"]
        })

df = pd.DataFrame(data)
df.to_csv("linkedin_creators_posts.csv", index=False)
print("✅ CSV saved as linkedin_creators_posts.csv")

driver.quit()
