from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st
from dotenv import load_dotenv



load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
if gemini_api_key is None:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
# Load environment variables from .env file
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# langsmith tracing
os.environ['LANGCHAIN_TRACING_V2'] = 'true'



##Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question: {input}"),
    ]
)


##streamlit framework

st.title('LangChain Chatbot with Google Gemini')

input_text = st.text_input("Enter your question:")
llm= ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_api_key,
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"input": input_text}))
