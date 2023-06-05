import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #read local .env file

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
llm = ChatOpenAI(temperature=0.9)
df = pd.read_csv('Data\Train.csv')

print("<<<<<-----------------------------A very simple LLM CHAIN -------------------->>>>")

prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

chain = LLMChain(llm=llm, prompt = prompt)
product = "Queen Size Sheet Set"
print(chain.run(product))

print("<<<<<-------------------SimpleSequentialChain(single input/output)-------------------------->>>>>")
from langchain.chains import SimpleSequentialChain
llm = ChatOpenAI(temperature=0.9)

first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
        a company that makes {prodcut}?"
)
chain_one = LLMChain(llm=llm, prompt = first_prompt)

second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
        company: {company_name}"
        
)
chain_two = LLMChain(llm = llm, prompt = second_prompt)

overall_simple_chain = SimpleSequentialChain(chains = [chain_one, chain_two],
                                             verbose = True)
print(overall_simple_chain.run(product))

print("<<<<<----------------------SequentialChain(Multiple inputs/ outputs)-------------------------->>>>>")
from langchain.chains import SequentialChain
llm = ChatOpenAI(temperature=0.9)
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following category to swahili:"
    "\n\n{Category}"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key= "Swahili_Review")

second_prompt = ChatPromptTemplate.from_template(
    "can you explain what the following category means in one sentence:"
    "\n\n{Swahili_Review}"
)

chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review: \n\n{Category}"
)

chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

fourth_prompt = ChatPromptTemplate.from_template(
    "Write a sentence to show how this Category can be used in a sentence"
    "Category in the specified language:"
    "\n\nCategory: {Category}\n\nLanguage: {language}"
)

chain_four = LLMChain(llm = llm, prompt = fourth_prompt, output_key="followup_message")

overall_chain = SequentialChain(
    chains = [chain_one, chain_two, chain_three, chain_four],
    input_variables=["Category"],
    output_variables=["Swahili_Review", "Category","followup_message"],
    verbose = True,
)

category = df.MERCHANT_CATEGORIZED_AS[0]
print(overall_chain(category))

print("<<<<<----------------------------------Router Chain -------------------------------->>>>")
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""


prompt_infos = [
    {
        "name":"physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },

    {
        "name":"math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name":"History",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    },
    {
        "name":"computer science",
        "description":"Good for answering computer science questions",
        "prompt_template": computerscience_template
    }
]

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0)
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template= prompt_template)
    chain = LLMChain(llm=llm, prompt = prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt= default_prompt)


MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations = destinations_str
)

router_prompt = PromptTemplate(
    template = router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),

)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains = destination_chains,
                         default_chain=default_chain, verbose = True)

print(chain.run("What is black body radiation?"))
print(chain.run("What is 2+2"))
print(chain.run("Why does every cell in our body contain DNA?"))