import streamlit as st
import requests
from gtts import gTTS
#from googletrans import Translator
from translate import Translator


def text_to_speech(text, output_file="output.mp3"):
    # Convert text to speech
    translator = Translator(from_lang='en',to_lang='hi')
    print(text)
    print(type(text))
    translation = translator.translate(text)
    print(translation)
    speech = gTTS(text=translation, lang='hi')
    speech.save(output_file)


st.title("News Summarization and Text-to-Speech Application")

company_name = st.text_input('Enter Company Name')
if st.button("Enter"):
    # Make a POST request to the Flask API
    url = "http://localhost:8080/query"
    response = requests.post(url, json={'query': company_name})

    # Return of flask
    if response:
        text_to_speech(response.json()["Final Sentiment Analysis"])
        st.audio("output.mp3", format='audio/mp3')
        st.json(response.json())
