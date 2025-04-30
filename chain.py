from config import llm
from models import RouteQuery, GradeDocuments, GradeHallucinations, GradeAnswer
from prompts import route_prompt, grade_doc_prompt, hallucination_prompt, answer_prompt, rewrite_prompt, prompt
from langchain_core.output_parsers import StrOutputParser

question_router = route_prompt | llm.with_structured_output(RouteQuery)
retrieval_grader = grade_doc_prompt | llm.with_structured_output(GradeDocuments)
hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)
answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)
question_rewriter = rewrite_prompt | llm | StrOutputParser()
rag_chain = prompt | llm | StrOutputParser()

