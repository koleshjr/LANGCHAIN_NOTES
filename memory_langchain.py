import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
import warnings
warnings.filterwarnings('ignore')
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

print("<<< ---------------------------------First conversation with conversational buffer memeory-------------------->>>>")
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory() #store memory
# memory.save_context({"input": "Hi"}, {"output": "What's up"}) #save context explicitly
conversation = ConversationChain(
    llm = llm,
    memory = memory,
    verbose = False
)
for i in range(5):
    user_input = str(input())
    print(conversation.predict(input= user_input))
    print(memory.buffer) # shows the current stored conversation
    print(memory.load_memory_variables({})) # a dict of the history
    i+=1


print("<<< --------------------------------- ConversationalBufferWindowMemory-------------------->>>>")
#to save on memory based on a window size: 
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm = llm,
    memory= memory,
    verbose = False
)
for i in range(5):
    user_input = str(input())
    print(conversation.predict(input= user_input))
    print(memory.buffer) # shows the current stored conversation
    print(memory.load_memory_variables({})) # a dict of the history
    i+=1


print("<<< --------------------------------- ConversationTokenBufferMemory-------------------->>>>")
#to save on memory based on tokens: 
from langchain.memory import ConversationTokenBufferMemory
memory = ConversationTokenBufferMemory(llm = llm,max_token_limit=50)
conversation = ConversationChain(
    llm = llm,
    memory= memory,
    verbose = False
)
for i in range(5):
    user_input = str(input())
    print(conversation.predict(input= user_input))
    print(memory.buffer) # shows the current stored conversation
    print(memory.load_memory_variables({})) # a dict of the history
    i+=1


print("<<< --------------------------------- ConversationSummaryBufferMemory-------------------->>>>")
#saves on memory based on the summarized conversation
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm = llm,max_token_limit=500)
conversation = ConversationChain(
    llm = llm,
    memory= memory,
    verbose = False
)
for i in range(3):
    user_input = str(input())
    print(conversation.predict(input= user_input))
    print(memory.buffer) # shows the current stored conversation
    print(memory.load_memory_variables({})) # a dict of the history
    i+=1

print(" <<< --------------------------------- Additional Notes --------------------------------->>>>")
print("""
    Vector_data_memory: stores text in a vector db and retrieves the most relevant blocks of text
    Entity_Memories: using an LLM it remembers details about specific entities
    Multiple Memories: conversation memory + entity memory to recall individuals
    Store the conversation in a conventional database (such as key-value store or SQL)





""")