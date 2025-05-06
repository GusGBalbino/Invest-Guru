import os
import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
from pathlib import Path
import tempfile

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Salva um arquivo enviado pelo usuário na pasta de PDFs e retorna o caminho."""
    try:
        # Criar o diretório se não existir
        pdf_dir = Path("data/pdfs")
        pdf_dir.mkdir(exist_ok=True, parents=True)
        
        # Caminho completo para o arquivo
        file_path = pdf_dir / uploaded_file.name
        
        # Verificar se já existe um arquivo com o mesmo nome
        if file_path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename, ext = os.path.splitext(uploaded_file.name)
            file_path = pdf_dir / f"{filename}_{timestamp}{ext}"
        
        # Salvar o arquivo
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return str(file_path)
    
    except Exception as e:
        st.error(f"Erro ao salvar o arquivo: {e}")
        return None

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Formata as fontes de uma resposta para exibição."""
    formatted_sources = []
    
    for i, source in enumerate(sources, 1):
        metadata = source.metadata
        source_name = metadata.get("source", "Desconhecido")
        page_num = metadata.get("page", "N/A")
        module = metadata.get("module", "Desconhecido")
        
        formatted_source = f"{i}. **{source_name}** (Pág. {page_num}, {module})"
        formatted_sources.append(formatted_source)
    
    return "\n".join(formatted_sources)

def get_llm_model(model_name: str, api_key: Optional[str] = None):
    """Retorna o modelo LLM adequado com base no nome."""
    from langchain_openai import ChatOpenAI
    
    if model_name == "gpt-3.5-turbo":
        if not api_key:
            raise ValueError("API key da OpenAI é necessária para usar o GPT-3.5")
        return ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0.2)
    
    elif model_name == "gpt-4":
        if not api_key:
            raise ValueError("API key da OpenAI é necessária para usar o GPT-4")
        return ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0.2)
    
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado")

def initialize_session_state():
    """Inicializa as variáveis na sessão do Streamlit."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False 