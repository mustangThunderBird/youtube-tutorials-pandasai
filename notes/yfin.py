import requests
from bs4 import BeautifulSoup
from yahoo_fin import news
from transformers import pipeline
import warnings
import pandas as pd
import concurrent.futures
import unicodedata
import json
import re

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0)
summarizer = pipeline("summarization")
warnings.filterwarnings('ignore')

def basic_cleanup(text):
    """Performs basic cleanup on the input text"""
    # Normalize Unicode characters
    cleaned_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove special characters except common punctuations
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"()\s]', '', cleaned_text)
    # Remove multiple spaces and newlines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def summarize_article(article_text):
    # Use the summarizer to create a concise summary
    summary = summarizer(article_text[:2056], max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def add_sentiment(df):
    # Apply the sentiment analysis pipeline to each article
    sentiments = df['article_text'].apply(lambda text: sentiment_pipeline(text[:2056])[0]['label'] if text != "N/A" else "UNKNOWN")
    df['sentiment'] = sentiments
    return df

def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs])
            # Normalize and remove special characters
            article_text = basic_cleanup(article_text)
            return article_text
        else:
            print(f"Failed to get article text from {url}: Status code {response.status_code}")
            return "N/A"
    except Exception as e:
        print(f"Failed to get article text from {url}: {e}")
        return "N/A"
    
def prepare_finetuning_data(df: pd.DataFrame):
    training_data = []
    for _, row in df.iterrows():
        if row.get('sentiment') != "UNKNOWN":
            context = f"Title: {basic_cleanup(row['title'])}\nPublished: {row['published']}\nArticle: {row['article_text']}\nSentiment: {row['sentiment']}"
            # Example question about the article
            question = "What is the sentiment of this article and why?"
            # Summarize the article as a reason
            summary = summarize_article(row['article_text'])
            summary = basic_cleanup(summary)
            answer = f"The sentiment is {row['sentiment']} because: {summary}"
            
            training_data.append({"prompt": f"{context}\nQuestion: {question}", "completion": answer})
        
    # Save to JSON
    with open('ollama_training_data.json', 'w') as f:
        json.dump(training_data, f, indent=4)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['summary', 'link', 'published', 'title']]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        article_texts = list(executor.map(get_article_text, df['link']))
    df['article_text'] = article_texts
    df = add_sentiment(df)
    return df

if __name__ == "__main__":  
    stock_news = news.get_yf_rss('MSFT')
    df = pd.DataFrame(stock_news)
    df_clean = preprocess_data(df)
    prepare_finetuning_data(df_clean)