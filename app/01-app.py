import streamlit as st
from pathlib import Path

# acessa o caminho atual do diretorio do projeto
folder_files = Path(__file__).parent.parent/"files"

# criando a pasta files
folder_files.mkdir(exist_ok=True)

def cria_chain_conversa():
    st.session_state["chain"] = True

def save_uploaded_files(uploaded_files, folder):
    """Salva arquivos enviados na pasta especificada"""
    for file in folder.glob("*.pdf"):
        file.unlink()
    # Salvar novos arquivos enviados
    for file in uploaded_files:
        (folder/file.name).write_bytes(file.read())    

# Função abaixo executa a aplicação
def main():
    with st.sidebar:
        st.header("Upload de PDFs")
        uploaded_pdfs = st.file_uploader("Adicione arquivos PDF",
                                         type="pdf",
                                         accept_multiple_files=True)
        if uploaded_pdfs:
            save_uploaded_files(uploaded_pdfs, folder_files)
            st.success(f"{len(uploaded_pdfs)} arquivo(s) salvo(s) com sucesso")

        label_botao = "Inicializar Chat"
        if "chain" in st.session_state:
            label_botao = "Atualizar Chat"
        if st.button(label_botao, use_container_width=True):
            if len(list(folder_files.glob("*.pdf"))) == 0:
                st.error("Adicione arquivos pdf para inicializar o chat")
            else: 
                st.success("Inicializando o Chat...")
                cria_chain_conversa()
                st.rerun()

if __name__ == "__main__":
    main()