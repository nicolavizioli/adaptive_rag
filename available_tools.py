from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from config import embedding_model
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

urls = [
    "https://sebastianraschka.com/blog/2025/the-state-of-reinforcement-learning-for-llm-reasoning.html",
    "https://sebastianraschka.com/blog/2025/understanding-reasoning-llms.html",
    "https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html",
]

docs=[WebBaseLoader(url).load() for url in urls] 
docs_list=[item for sublist in docs for item in sublist]

text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=50
)

docs_split=text_splitter.split_documents(docs_list)

vectostore=Chroma.from_documents(
    documents=docs_split,
    collection_name='Chrome_db',
    embedding=embedding_model
)

retriever=vectostore.as_retriever()