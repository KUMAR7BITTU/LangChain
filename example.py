# If you want to create your own use cases custom prompt then you can create with the help prompt template .

import os
from constants import openai_key  # This will contain your OpenAI API key
from langchain import PromptTemplate

from langchain.chains import LLMChain # Whenever you want to execute this prompt template then write this .
# Whenever you will be giving prompt chains then this llm chains will be beneficial because you are executing this because you are giving some kind of input and getting some kind of input. 

# Prompt chaining involves feeding the output of one prompt into the next. It helps structure the reasoning process into discrete, easier-to-handle steps—much like passing a baton through a relay race

import streamlit as st
from langchain.chat_models import ChatOpenAI

# Set up environment
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit interface
st.title('Celebrity Search Results')

input_text = st.text_input("Search the topic you want:")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "tell me about celebrity {name}"
)

# with respect to every prompt template, we will have respective llm chain because we need to execute those things .

# Use Chat model instead of completion model
llm = ChatOpenAI(
    openai_api_key=openai_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# We need to create this llm chain .
chain = LLMChain(llm = llm , prompt = first_input_prompt, verbose = True) # llmchain will specifically run this template .

if input_text:
    st.write(chain.run(input_text))
