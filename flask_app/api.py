from flask import Flask, request, jsonify
from utils import get_links, get_article_content, get_llm_response, get_sentiment_distribution, get_topic_overlap, \
    get_comparative_analysis, get_final_sentiment
import os
from ast import literal_eval
from dotenv import load_dotenv

app = Flask(__name__)

# Load API keys from environment variables
load_dotenv()
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HF_TOKEN = os.getenv("HF_TOKEN")


@app.route('/query', methods=['POST'])
def query():
    # get the data/query from streamlit app
    company = request.get_json()['query']
    links = get_links(company)
    articles = [get_article_content(link) for link in links]
    articles_json = [get_llm_response(article) for article in articles]
    data = {'Company Name': company, 'Articles': articles_json}
    sentiment_counts = get_sentiment_distribution(articles_json)
    data["Comparative Sentiment Score"] = {"Sentiment Distribution": sentiment_counts}
    common_topics = get_topic_overlap(articles_json)
    data["Topic Overlap"] = {"Common Topics": common_topics}
    data["Coverage Differences"] = get_comparative_analysis(articles_json)
    data["Final Sentiment Analysis"] = get_final_sentiment(articles_json)

    return jsonify(data)


app.run(host='0.0.0.0', port=8080, debug=True)