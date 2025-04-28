from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    deployment_name=os.getenv('OPENAI_LLM_DEPLOYMENT_NAME'),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_type=os.getenv('OPENAI_API_TYPE'),
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)