# ChatPDF: Chatbot com PDFs usando LangChain e OpenAI

Este projeto é um chatbot que permite conversar com documentos PDF enviados pelo usuário. Ele utiliza as bibliotecas LangChain, OpenAI, FAISS e Streamlit para processar, indexar e responder perguntas baseadas no conteúdo dos PDFs.

## Funcionalidades

- Upload de múltiplos arquivos PDF.
- Indexação e divisão dos documentos em trechos relevantes.
- Busca semântica e respostas contextuais usando modelos da OpenAI.
- Interface web interativa via Streamlit.
- Memória de conversação para manter o contexto do chat.

## Como executar

1. Instale as dependências:
   ```sh
   pip install -r requirements.txt
   ```
2. Execute o aplicativo Streamlit:
   ```sh
   streamlit run .\01-app.py
   ```