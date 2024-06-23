import json,os,sys,boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import numpy as np
from langchain.text_splitter import TextSplitter
from langchain_community.document_loader import PyPDFDirectoryLoader
from langchain.vectorstores import RetrievalQA,FAISS
from langchain.prompts import PromptTemplate

bedrock=boto3.client(service_name='bedrock-runtime')
bedrock_embeddings=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',client=bedrock)

def data_ingestion():
    loader=PyPDFDirectoryLoader('data_pdf')
    documents=loader.load()

    text_splitter=RecursiveCharacterSplitter(chunk_size=10000,
                                             chunk_overlap=200)
    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(docs,bedrock_embeddings)
    #save in local - for the same doc, we will not embedding again
    vectorstore_faiss.save_local('faiss_index')
    return vectorstore_faiss

def load_llama2_llm():
    return Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':512})

def load_claude_llm():
    return Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,
                model_kwargs={'max_gen_len':512})

prompt_template="""
    Human: Use the information provided as context to give a concise answer to the question. 
    Your answer must cover in all complete information not exceeding 250 words in total.
    Just answer the question without pre-amble. 
    <context>
    {context}
    </context>
    Question: {question}
    Assistant: """

def load_prompt_template():
    return PromptTemplate(input_variables=["context","question"],
                          template=prompt_template)

def load_retrieval_qa(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_kwargs={"k":3}),
        chain_type_kwargs={"prompt":load_prompt_template()},
        return_source_documents=True
    )
    return qa("{query}:query")['result']
    
def main():
    st.title("LangChain Retrieval QA")
    st.write("This is a demo of LangChain Retrieval QA by NweNwe")
    

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title('Update or create Vector store')

        if st.button('Vectors update'):
            with st.spinner("Processing"):
                docs=data_ingestion()
                vectorstore_faiss=get_vector_store(docs)
                st.success("Vector store processing is done.")
        if st.button("Cladue LLM"):
            with st.spinner("Processing....."):
                faiss_index=FAISS.load_local('faiss_index',bedrock_embeddings)
                llm=load_claude_llm()
                st.write(load_retrieval_qa(llm,faiss_index),user_question)
                st.success("Done")
        if st.button("Llama2 LLM"):
            with st.spinner("Processing........")
                faiss_index=FAISS.load_local('faiss_index',bedrock_embeddings)
                llm=load_llama2_llm()
                st.write(load_retrieval_qa(llm,faiss_index),user_question)
                st.success("Done")

if __name__=="__main__":
    main()

  
