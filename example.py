# If you want to create your own use cases custom prompt then you can create with the help prompt template .

import os
from constants import openai_key  # This will contain your OpenAI API key
from langchain import PromptTemplate

from langchain.chains import LLMChain # Whenever you want to execute this prompt template then write this .
# Whenever you will be giving prompt chains then this llm chains will be beneficial because you are executing this because you are giving some kind of input and getting some kind of input. 

# Prompt chaining involves feeding the output of one prompt into the next. It helps structure the reasoning process into discrete, easier-to-handle steps—much like passing a baton through a relay race

import streamlit as st
from langchain.chat_models import ChatOpenAI
#from langchain.chains import SimpleSequentialChain # We can combine the chain and probably set the sequence for that .
from langchain.chains import SequentialChain

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
chain = LLMChain(llm = llm , prompt = first_input_prompt, verbose = True, output_key = 'person') # llmchain will specifically run this template . We use output_key here because the output of first prompt will be used as input in second prompt

second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "when was the {person} born"
)

chain2 = LLMChain(llm = llm , prompt = second_input_prompt, verbose = True, output_key = 'dob')

#parent_chain = SimpleSequentialChain(chains=[chain,chain2],verbose=True)
# The problem with SimpleSequentialChain is that as we are getting inputs, it will only show you the last input/output .


# To show the entire information we will use the SequentialChain.
# parent_chain = SequentialChain(chains=[chain,chain2],input_variables = ['name'], output_variables = ['person','dob'], verbose=True)

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major events happened around {dob} in the world"
)


chain3 = LLMChain(llm = llm , prompt = third_input_prompt, verbose = True, output_key = 'description')

parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables = ['name'], output_variables = ['person','dob','description'], verbose=True)



if input_text:
    # st.write(parent_chain.run(input_text))
    st.write(parent_chain({'name' : input_text})) # Here we have to enter key value pairs .
