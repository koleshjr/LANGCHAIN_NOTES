"""
Notes: LLMS can only inspect a few thousand words at a time
Embeddings: Numerical representations of text
Embedding vector captures content/meaning
Text with similar content will have similar vectors
Vector-Database - Populate it with chunks(broken from large docs - create an embedding for each chunk)- index
so that we can pass the mostt relevant to the llm

create an embedding for the query, compare the embeddings to the stored ones, pick n most similar
chain_types:
    stuff-> simplest method, stuff all the data into the prompt as context to pass to the language model
            -> doesnt work in lots of chucks due to contexxt length
    map_reduce - > takes all the chunks passees them along with the question to llm gets back the response
        and use another llm to summarize the individual responses to a final answer
        Treats all docs as independent and takes a lot more calls (summarization)

    Refine: Builds upon the answer from the previous docs unlike map reduce
        Not as fast , takes as many calls as map_reduce

    Map_rerank: 
        Do a single call to a llm for each doc and ask it to return a score and select the highest score
        The score is dependent to the llm
        all calls are independent and expensive tooo




"""


import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

file = "Data\Train.csv"
loader = CSVLoader(file_path = file)
docs = loader.load()
# print("<<<<----------------------docs[0]--------->>>>>>>>>")
# print(docs[0])
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query(str(docs[0]))

# print("<<<<----------------------len(embed) of docs[0]--------->>>>>>>>>")
# print("There are ",len(embed)," different embeddings")
# print(embed[:5])


print("<<<<----------------------list(docs)--------->>>>>>>>>")
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

query = "Get all merchants categorized as shopping and summarizze it in a table with merchanr_name and merchant_categorized as columns"
docs = db.similarity_search(query)
# print(docs[0])

print("<<<<< ------------ How to use db as retriever----------->>>>>>>")
retriever = db.as_retriever()
llm = ChatOpenAI(temperature= 0.0)

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  #alts; map_reduce
    retriever = retriever,
    verbose = True
)
query = "Get all merchants categorized as shopping and summarizze it in a table with merchanr_name and merchant_categorized as columns"
response  = qa_stuff.run(query)
print(response)

print("<<<<<,----------------------------One liner code that does all these things -------------->>>>")
from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    #additional arguments to make it flexible
).from_loaders([loader])

query = "Get all merchants categorized as shopping and summarizze it in a table with merchanr_name and merchant_categorized as columns"
response = index.query(query,llm = llm)
print(response)

