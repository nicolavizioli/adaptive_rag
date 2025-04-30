from langchain_core.prompts import ChatPromptTemplate
from langchain import hub



system_route = """You are an expert at routing a user question to a vectorstore or web search."""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_route),
    ("human", "{question}"),
])


system_grade_doc = """You are a grader assessing relevance of a retrieved document.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"""

grade_doc_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grade_doc),
    ("human", "Document: {document}\n\nQuestion: {question}"),
])


system_hallucination = """You are a grader checking if an answer is grounded in facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts"""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system_hallucination),
    ("human", "Facts: {documents}\n\nAnswer: {generation}"),
])


system_answer = """You are a grader checking if an answer resolves the user's question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resoves the question"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_answer),
    ("human", "Question: {question}\n\nAnswer: {generation}"),
])



system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized \n 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", system_rewrite),
    ("human", "Question: {question}"),
])

prompt = hub.pull("rlm/rag-prompt")
