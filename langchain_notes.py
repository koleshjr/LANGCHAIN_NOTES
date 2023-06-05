"""
focused on compositions and modularity
Common ways to combine components
Components:
    Models: LLMS 20+ Intergration
        Chat models
        Text embedding models: 10 + integrations

    Prompts:
        Prompt templates
        output parsers: 5+ Implementation
            Retry/ Fixing Logic
    Example selectors: 5+ Implementations
    Indexes:
        Document Loaders: 50+ Implementations
        Text Splitters: 10+ Implementations
        Vector stores: 10+ intergrations
        Retrievers: 5+ Intergrations/Implementations

    Chains:
        Prompt + LLM + Output parsing
        can be used as building blocks for longer chains
        more application specific chains: 20 + types

    Agents:
        Agent Types: 5+ types
            Algorithms for getting llms to use tools
        Agent Toolkits: 10 + implementations
            Agents aimed with specific tools for a specific implementation

    Why do we need a prompt template abstraction like the one we have in Langchain
        Prompts can be long and detailed 
        Reuse good prompts when you can
        Langchain also provides prompts for common operations            

        
    LLMs are stateless - each transaction is independent
    langchain provides several kinds of memory to store and accumulate the conversation
    

"""