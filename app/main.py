import streamlit as st
import os
import sys
from pathlib import Path
import logging

# Adiciona o diretório do projeto ao PATH para importações relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chat import chat_section, clear_chat_history
from app.upload import upload_section, document_management_section
from app.utils import initialize_session_state
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar a página
st.set_page_config(
    page_title="Invest Guru",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar variáveis de estado da sessão
initialize_session_state()

# Função para verificar se o diretório de PDFs existe e contém arquivos
def check_pdfs_loaded():
    pdf_dir = Path("data/pdfs")
    return pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))

# Função para verificar se o vectorstore está criado
def check_vectorstore_exists():
    index_dir = Path("data/index")
    return index_dir.exists() and any(os.listdir(index_dir)) if index_dir.exists() else False

# Barra lateral
with st.sidebar:
    st.title("📊 Invest Guru 🤖")
    st.markdown("---")
    
    # Status dos documentos
    st.subheader("📈 Status do Sistema")
    
    pdfs_loaded = check_pdfs_loaded()
    vectorstore_exists = check_vectorstore_exists()
    
    st.info(f"📚 PDFs Carregados: {'✅' if pdfs_loaded else '❌'}")
    st.info(f"🔍 Vectorstore Criado: {'✅' if vectorstore_exists else '❌'}")
    
    st.markdown("---")
    
    # Opções
    if st.button("🧹 Limpar Chat", use_container_width=True):
        clear_chat_history()
        st.rerun()
    
    st.markdown("---")
    
    # Informações do sistema
    st.caption("Desenvolvido com LangChain, ChromaDB e Streamlit")
    st.caption("v1.0.0 - 2023")
    st.caption("Criador - Gustavo Gomes Balbino")

# Corpo principal
st.title("📚 Invest Guru 🤖")

# Abas para as diferentes seções
tab1, tab2 = st.tabs(["💬 Chat", "📤 Gerenciamento de Documentos"])

with tab1:
    chat_section()

with tab2:
    upload_section()
    st.markdown("---")
    document_management_section()

if __name__ == "__main__":
    # Aqui poderia ter código adicional para inicialização se necessário
    pass 