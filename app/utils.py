# Backend onde serão realizadas as configurações do LangChain
from pathlib import Path
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# Usado para administrar as memorias de estado
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Somente carregar os valores em dotenv
_ = load_dotenv(find_dotenv())

# acessa o caminho atual do diretorio do projeto
folder_files = Path(__file__).parent.parent/"files"

model_name = "gpt-3.5-turbo-0125"

# importação de documentos
def import_documentos():
    documentos = []
    for arquivo in folder_files.glob("*.pdf"):
        loader = PyPDFLoader(arquivo)
        documentos_arquivo = loader.load()
        documentos.extend(documentos_arquivo)
    return documentos

# Quebra [split] de documentos
def split_documentos(documentos):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documentos = recur_splitter.split_documents(documentos)

    # identificando a origem do documento consultado.
    for i, doc in enumerate(documentos):
        doc.metadata["source"] = doc.metadata["source"].split("/")[-1]
        doc.metadata["doc_id"] = i
    return documentos

# Embedding e criando o VectorStore
def cria_vector_store(documentos):
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents = documentos,
        embedding=embedding_model
    )
    return vector_store

# Realizando a importação das funções acima na sequencia necessaria
def cria_chain_conversa():
    documentos = import_documentos()
    documentos = split_documentos(documentos)
    vector_store = cria_vector_store(documentos)

    chat = ChatOpenAI(model=model_name)
    memory = ConversationBufferMemory(return_messages=True,
                                      memory_key="chat_history",
                                      output_key="answer")
    retriever = vector_store.as_retriever()

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    st.session_state["chain"] = chat_chain
    return chat_chain