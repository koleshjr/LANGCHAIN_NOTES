import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import warnings
warnings.filterwarnings("ignore")

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

tools = load_tools(["llm-math", "wikipedia"], llm = llm)
agent = initialize_agent(
    tools, 
    llm,
    agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    handle_parsing_errors = True,
    verbose = True
)
print("<<<<<<-----------using llm-math tool ---------------->>>>>")
print(agent("What is the 25% of 300?"))

print("<<<<<<<< ------------------Use Wikipedia Tool--------->>>>>>>")
print(agent("who was the first female programmer?"))


print("<<<<<<<-----creating a python repl code ---------------------->>>>>> ")
# import langchain
# langchain.debug = True
agent = create_python_agent(
    llm,
    tool = PythonREPLTool(),
    verbose=True
)
customer_list = [['Harrison','Chase'],
                 ['lang','chain'],
                 ['Dolly','Too'],
                 ['Elle','Elon'],
                 ['Geoff','Fusion'],
                 ['Trance','Former'],
                 ['Jen','Ayal']]

print(agent.run(f"""sorth these customers by last name and then first name\
          and print the output: {customer_list}"""))

print("<<<<<<<<------------Creating your own Custom tool ------------------------->>>>>>>")
from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    #when and how the agent will use this tool
    """Returns todays date, use this for any \
        questions related to knowing todays date.\
            The input should always be an empty string, \
                and this function will always return todays \
                    date - any date mathematics should occur \
                        outside this function. """
    return str(date.today())


agent = initialize_agent(
    tools + [time],
    llm,
    agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors= True,
    verbose = True
)

print(agent.run("Whats the date today?"))

