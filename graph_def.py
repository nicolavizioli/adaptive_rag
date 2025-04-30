from typing import List
from langchain.schema import Document
from typing_extensions import TypedDict
from chain import retrieval_grader, rag_chain, question_rewriter, hallucination_grader, question_router
from available_tools import retriever, web_search_tool


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    


def retrieve(state):
    '''
    Retrieve documents
    Args:
        state (dict): The current graph state
    Return:
        state (dict): New key added to state, documents, that contains retrieved documents
    '''
    print("---RETRIEVE---")
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}

def generate(state):
    '''
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    '''
    print("---GENERATE---")
    answer = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": answer}



def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---GRADE DOCUMENTS---")
    filtered = []
    for d in state["documents"]:
        if retrieval_grader.invoke({"question": state["question"], "document": d.page_content}).binary_score == "yes":
            filtered.append(d)
    return {"documents": filtered, "question": state["question"]}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    better = question_rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": better}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    web_results = web_search_tool.invoke({"query": state["question"]})
    return {"documents": [Document(page_content="\n".join([r["content"] for r in web_results]))], "question": state["question"]}


def grade_generation(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---GRADE GENERATION---")
    grounded = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
    return grounded.binary_score

#####CONDITIONAL EDGE######

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"