# # Integrate our code OpenAI API :- 
# import os
# from constants import openai_key # This will contain our api_key
# from langchain.llms import OpenAI # Openai model

# # first setup the environment and initialize with openai_key
# os.environ["OPENAI_API_KEY"] = openai_key

# import streamlit as st 

# # streamlit framework
# st.title('LangChain Demo with OpenAI API')

# # We will create a text box whenever we will write any text it will be able to interact with the api .
# input_text = st.text_input("Search the topic you want  :- ")

# # For interacting with api , we first need to initialize OpenAI.
# # OPENAI LLMS
# # temperature controls the randomness or creativity of the model's output. How much balance answer you want to control .
# llm = OpenAI(temperature=0.8)

# if input_text:
#     st.write(llm(input_text)) # input will be given to llm model to do prediction and print that prediction.


import os
from constants import openai_key  # This will contain your OpenAI API key

import streamlit as st
from langchain.chat_models import ChatOpenAI

# Set up environment
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit interface
st.title('LangChain Chat with OpenAI API')

input_text = st.text_input("Search the topic you want:")

# Use Chat model instead of completion model
llm = ChatOpenAI(
    openai_api_key=openai_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

if input_text:
    response = llm.invoke(input_text)
    st.write(response.content)
