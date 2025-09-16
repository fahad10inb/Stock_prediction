import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import joblib
from tenacity import retry, stop_after_attempt, wait_fixed
from news import fetch_and_return_articles  # Your function to fetch news articles
import math
from decimal import Decimal, ROUND_DOWN
import google.generativeai as genai
import os
from dotenv import load_dotenv


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
load_dotenv()
'''
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")'''

GEMINI_API_KEY = "AIXXXXXXXXXXXXXXXXXXXX"


genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Retry logic for loading sentiment analysis model and vectorizer
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))  # Retry 3 times with 5-second intervals
def load_model_and_vectorizer():
    try:
        sentiment_model = joblib.load('.\model\Latest_Model.joblib')  # Replace with actual path
        vectorizer = joblib.load('.\model\improved_tfidf_vectorizer.joblib')  # Replace with actual path
        return sentiment_model, vectorizer
    except Exception as e:
        logger.error(f"Error loading sentiment model or vectorizer: {e}")
        raise  # Retry if loading fails

sentiment_model, vectorizer = None, None
try:
    sentiment_model, vectorizer = load_model_and_vectorizer()
except Exception as e:
    logger.error("Failed to load sentiment model or vectorizer after retries.")

# Data model for input
class CompanyRequest(BaseModel):
    company_name: str

# Normalize sentiment score to a scale of 0-10
def normalize_sentiment_score(score, min_score=-0.3, max_score=0.45):
    try:
        return (score - min_score) / (max_score - min_score) * 10
    except ZeroDivisionError:
        logger.error(f"ZeroDivisionError in normalize_sentiment_score with score: {score}")
        return 0

# Get sentiment score
def get_sentiment_score(text):
    if not sentiment_model or not vectorizer:
        logger.error("Sentiment model or vectorizer is not loaded.")
        return 0
    try:
        vectorized_text = vectorizer.transform([text])
        sentiment_score = sentiment_model.predict(vectorized_text)[0]
        return sentiment_score
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        return 0

# Summarize articles using Gemini API
def summarize_articles_together(articles):
    try:
        combined_text = " ".join(articles)
        prompt = f"Summarize the key insights from the following news articles in a clear and concise manner (200 words). Focus on the most important developments, trends, and implications, ensuring it is easy to understand for a general audience: {combined_text}"
        response = model_gemini.generate_content(prompt)
        return response.text if response else "Error summarizing articles."
    except Exception as e:
        logger.error(f"Error summarizing with Gemini API: {e}")
        return "Error summarizing articles."

# Function to clean up text by removing special characters, including '*' symbols and article labels
def clean_article_text(text):
    text = re.sub(r'Article \d+:', '', text)  # Remove article labels like "Article 1", "Article 2", etc.
    text = re.sub(r'[*]', '', text)  # Remove all '*' symbols
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Process articles
def process_articles(news_articles):
    min_score = -0.3
    max_score = 0.45
    headlines = [article['Summary'] for article in news_articles[:10] if 'Summary' in article]
    
    if not headlines:
        logger.error("No headlines found in the articles.")
        return {"summary": "No articles available.", "overall_score": 0, "investment_advice": "No data."}

    # Summarizing articles before cleaning
    combined_summary = summarize_articles_together(headlines)
    
    # Now apply the cleaning function after summarizing
    combined_summary = clean_article_text(combined_summary)
    
    sentiment_scores = [get_sentiment_score(headline) for headline in headlines]
    avg_sentiment_score = (math.floor((sum(sentiment_scores) / len(sentiment_scores)) * 100) / 100) if sentiment_scores else 0.00
    normalized_score = normalize_sentiment_score(avg_sentiment_score, min_score, max_score)
    normalized_score = Decimal(normalized_score).quantize(Decimal('0.00'), rounding=ROUND_DOWN)
    
    if normalized_score >= 7:
        investment_advice = """
        Strong positive sentiment. The stock shows a promising outlook, with favorable news 
        and market trends. It suggests strong growth potential in the short and long term. 
        However, while this might be a good investment opportunity, it is important to conduct 
        additional due diligence, analyze company fundamentals, and assess any potential risks 
        before making a final investment decision.
        """
    elif normalized_score >= 4:
        investment_advice = """
        Moderate sentiment. While the stock shows some potential, there are mixed signals in the 
        market. This could indicate both opportunities and risks. It is advisable to conduct in-depth 
        research, analyze the company’s recent performance, review industry trends, and assess broader 
        economic conditions. A thorough risk assessment and possibly waiting for more clarity before 
        making a decision is recommended.
        """
    else:
        investment_advice = """
        Weak sentiment. The company or stock is facing significant challenges, reflected in negative 
        news or market performance. There may be concerns about financial health, market conditions, 
        or management decisions. It’s advisable to be cautious and consider avoiding this investment 
        or reassessing your strategy. If you're already invested, it might be a good idea to review the 
        situation and decide whether to cut losses or wait for improvement.
        """
    
    return {
        "summary": combined_summary,
        "overall_score": normalized_score,
        "investment_advice": investment_advice
    }

@app.post("/analyze_company/")
async def analyze_company(data: CompanyRequest):
    company_name = data.company_name
    try:
        news_articles = fetch_and_return_articles(company_name)
        if not news_articles:
            logger.error(f"No articles found for company: {company_name}")
            raise HTTPException(status_code=404, detail="No articles found for this company.")
        
        result = process_articles(news_articles)
        return result
    except Exception as e:
        logger.error(f"Error in analyze_company: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")
