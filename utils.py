import os
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, RootModel
from typing import Literal, List
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from ast import literal_eval
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv('SERPER_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class CompanyData(BaseModel):

    Title: str = Field(description='Title of the Article')
    Summary: str = Field(description='A brief summary of the article')
    Sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(description='Sentiment of the article')
    Topics: List[str] = Field(description='Mention key topics from the article')


def get_links(query):
    url = "https://google.serper.dev/news?q={}&gl=in&tbs=qdr%3Am&apiKey={}".format(query,SERPER_API_KEY)
    response = requests.get(url)
    links = [link['link'] for link in response.json()['news']]
    return links


def get_article_content(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/80.0.3987.162 Safari/537.36'}

    webpage = requests.get(url, headers=headers).text
    soup = BeautifulSoup(webpage, 'html.parser')
    data = {'heading': soup.find('h1').text.strip()}

    # implementation of fetching headings and content from the articles
    content = ""
    for tag in soup.find_all('p'):
        content = content + '\n' + tag.text.strip()
    data['content'] = content.strip()
    return data


def get_llm_response(article):

    parser = PydanticOutputParser(pydantic_object=CompanyData)
    format_instruction = parser.get_format_instructions()
    template = PromptTemplate(
        template="Extract the Title, Summary, Sentiment and Topics from the give article."
                 " \n {format_instruction} \n {article}",
        input_variables=['article'],
        partial_variables={'format_instruction': format_instruction}
    )

    model = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='llama3-70b-8192')

    chain = template | model | parser

    final_result = chain.invoke({'article': article})

    return final_result.model_dump_json(indent=2)


def get_sentiment_distribution(articles):
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        article = literal_eval(article)
        if article['Sentiment'] == 'Neutral':
            sentiment_counts['Neutral'] += 1
        elif article['Sentiment'] == 'Positive':
            sentiment_counts['Positive'] += 1
        else:
            sentiment_counts['Negative'] += 1

    return sentiment_counts


def get_topic_overlap(articles):
    total_topics = [literal_eval(article)["Topics"] for article in articles]

    # Find the common elements using set intersection
    common_topics = set(total_topics[0])  # Start with the first list
    for lst in total_topics[1:]:  # Iterate over the remaining lists
        common_topics.intersection_update(lst)

    # Convert the set back to a list (if needed)
    common_topics_list = list(common_topics)

    return common_topics_list


def get_comparative_analysis(articles):
    output_parser = JsonOutputParser()
    format_instruction = output_parser.get_format_instructions()

    template = PromptTemplate(
        partial_variables = {'format_instruction': format_instruction},
        input_variables=["articles_data"],
        template="""
    Generate a coverage differences report comparing the key themes, focus areas, and implications of the following news
    articles:
    \n{format_instruction}\n

    Articles : {articles_data}

    The report should highlight the differences in how each article covers the topic and analyze the potential impact of
    these differences. There can be any number of dictionaries in the output. Use the following output format for the 
    report:

    Example Output:

      
      {{
        "Comparison": "Article 1 highlights Tesla's strong sales, while Article 2 discusses regulatory issues.",
        "Impact": "The first article boosts confidence in Tesla's market growth, while the second raises concerns about 
        future regulatory hurdles."
      }},
      {{
        "Comparison": "Article 1 is focused on financial success and innovation, whereas Article 2 is about legal 
        challenges and risks.",
        "Impact": "Investors may react positively to growth news but stay cautious due to regulatory scrutiny."
      }}
      
    

    Instructions:
    1. Analyze the provided news articles and identify their key themes, focus areas, and tones.
    2. Compare how each article approaches the topic differently.
    3. Assess the potential impact of these differences on the audience, such as investors, stakeholders, or the general 
       public.
    4. Ensure the report is concise, clear, and structured as per the provided format.
    
    Do not give extra information or instruction. Always respond in the mentioned format instruction only. Do not 
    include texts like 'Here is the coverage differences report:' or any other text.
    """
    )

    model = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='llama3-70b-8192')

    chain = template | model | output_parser
    result = chain.invoke({'articles_data': articles})
    return result


def get_final_sentiment(articles):
    text = ""
    for idx, article in enumerate(articles):
        article = literal_eval(article)
        text = "\n" + text + str(idx) + article["Summary"] + "\n\n"

    template = PromptTemplate(
        input_variables=["articles_data"],
        template="Generate a final sentiment analysis of given articles in 2-3 sentences.\n\n{articles_data}"
                 "\n\n"
                 "Example Output: Tesla’s latest news coverage is mostly positive. Potential stock growth expected.\n\n"
                 "Do not give extra information or instruction. Always respond in the mentioned format instruction "
                 "only. Do not include texts like 'Here is a final sentiment analysis:' or any other text."
        )

    model = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='llama3-70b-8192')

    chain = template | model
    result = chain.invoke({'articles_data': text})

    return result.content



