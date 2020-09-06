import streamlit as st
import nltk
nltk.download('punkt')

# NLP Pkgs
import spacy
from textblob import TextBlob
from gensim.summarization import summarize

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

def text_analyzer(mytest):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(mytest)

    tokens = [token.text for token in docx]
    allData = [('Tokens:{},\n Lemma:{}'.format(token.text,token.lemma_)) for token in docx]
    return allData

def entity_analyzer(mytest):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(mytest)
    tokens = [token.text for token in docx] 
    entities = [(entity.text,entity.label_) for entity in docx.ents ]
    allData = ['Tokens: {},\n Entities: {}'.format(tokens,entities)]
    return allData

# Pkgs
from PIL import Image
import base64

def main():
    ''' NLP App with streamlit'''
    st.write("Built by @cindi_star")
    img = Image.open("cindi1.png")
    st.image(img, width=55)

    st.title("NLP with Streamlit")
    html_temp = """
    <div style="background-color:#ff4082;padding:5px">
    <h1 style="color:white;text-align:left;"> Natural Language Processing App </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.markdown("------------")
    

    # Tokenization
    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize Your Text")
        message = st.text_area("Enter Your Text", "Type Here")
    if st.button("Analyze"):
        nlp_result = text_analyzer(message)
        st.json(nlp_result)

    # Named Entity
    if st.checkbox("Show Named Entities"):
        st.subheader("Extract Entities From Your Text")
        message = st.text_area("Enter Your Text", "Type Here the text")
    if st.button("Extract"):
        entity_result = entity_analyzer(message)
        st.json(entity_result)
    
    # Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Sentiment of Your Text")
        message = st.text_area("Enter Your Text", "Type Here Your Text")
    if st.button("Analyze it"):
        blob = TextBlob(message)
        result_sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        polarity = "Negative" if result_sentiment < 0 else "Neutral" if result_sentiment == 0 else "Positive"
        st.success(f'Polarity: {result_sentiment}, Subjectivity: {subjectivity}, {polarity}')
    
    # Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize Your Text")

        message = st.text_area("Enter Text","Type Here ..")
        summary_options = st.selectbox("Choose Summarizer",['sumy','gensim'])
        if st.button("Summarize"):
            if summary_options == 'sumy':
               st.text("Using Sumy Summarizer ..")
               summary_result = sumy_summarizer(message)
            elif summary_options == 'gensim':
                 st.text("Using Gensim Summarizer ..")
                 summary_result = summarize(message)
            else:
                st.warning("Using Default Summarizer")
                st.text("Using Gensim Summarizer ..")
                summary_result = summarize(message)

      
            st.success(summary_result)
    
    st.sidebar.subheader("About The App")
    st.sidebar.text("NLP App With Streamlit")
    st.sidebar.info("Making it was very pleasing :D")
    emoji = """<svg width="5em" height="5em" viewBox="0 0 16 16" class="bi bi-emoji-smile" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
  <path fill-rule="evenodd" d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
  <path fill-rule="evenodd" d="M4.285 9.567a.5.5 0 0 1 .683.183A3.498 3.498 0 0 0 8 11.5a3.498 3.498 0 0 0 3.032-1.75.5.5 0 1 1 .866.5A4.498 4.498 0 0 1 8 12.5a4.498 4.498 0 0 1-3.898-2.25.5.5 0 0 1 .183-.683z"/>
  <path d="M7 6.5C7 7.328 6.552 8 6 8s-1-.672-1-1.5S5.448 5 6 5s1 .672 1 1.5zm4 0c0 .828-.448 1.5-1 1.5s-1-.672-1-1.5S9.448 5 10 5s1 .672 1 1.5z"/>
</svg> """
    st.sidebar.markdown(emoji, unsafe_allow_html=True)
    
html = """<svg width="2em" height="2em" viewBox="0 0 16 16" class="bi bi-arrow-down-square-fill" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
  <path fill-rule="evenodd" d="M2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2zm6.5 4.5a.5.5 0 0 0-1 0v5.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V4.5z"/>
</svg>  """
st.markdown(html, unsafe_allow_html=True)

if __name__ == '__main__':
     main()

