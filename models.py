from pydantic import BaseModel, Field
from typing_extensions import Literal

class RouteQuery(BaseModel):
    ''' Route a user query to the most relevant datasource'''
    
    datasource: Literal["vectorstore", "web_search"] = Field(
        description= "given a user questin decide if using retiever or web search "
        )


class GradeDocuments(BaseModel):
    ''' Binary score for relevance check on retrieved documents.'''
    
    binary_score: str = Field(
        description="documents are relevant to answer the user question: answer with 'yes' or 'no' " 
        )

class GradeHallucinations(BaseModel):
    '''Binary score for hallucination present in generation answer.'''
    
    binary_score: str = Field(
        description="the answer gives is grounded in the facts: answe with 'yes' or no" 
    )

class GradeAnswer(BaseModel):
    ''' Binary score to assess answer addresses question.'''
    
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )