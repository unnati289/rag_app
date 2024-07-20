from langchain_community.vectorstores import Chroma
from chat_box import embedding_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.runnables import RunnableSequence
import streamlit as st
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#query = "can you summarise the content of the entire document?"

db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
def retrieve_docs(retriever, query):
    retrieved_docs = retriever.get_relevant_documents(query)
    return retrieved_docs

#retrieved_docs=retrieve_docs(retriever,query)
"""for doc in retrieved_docs:
    print(doc.page_content)"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#lets see how I can use gemini-pro for generating answers
prompt_template = """
You are an AI assistant.
Please be truthful and give direct answers based on the following retrieved documents.

Retrieved Documents:
{context}

Question:
{question}

Answer:
"""
prompt=PromptTemplate(template=prompt_template)
def prepare_input(query):
    docs = retrieve_docs(retriever, query)
    formatted_docs = format_docs(docs)
    return {
        "context": formatted_docs,
        "question": query
    }
#prep_input=prepare_input(query)
###will be using LangChain
llm = ChatGoogleGenerativeAI(model="gemini-pro")
rag_chain = RunnableSequence( 
            prepare_input|
            prompt| 
            llm|
            StrOutputParser()
)
#result = rag_chain.invoke(query)

#Print the result
#print(result)
def get_answers(rag_chain,query):
    result=rag_chain.invoke(query)
    return result

##streamlit##
st.title("RAG + LLM Chatbot for placement_chronicles")
query = st.text_input("Enter your query:")
if query:
  result=get_answers(rag_chain,query)
  print(result)
  st.write("Response:", result)