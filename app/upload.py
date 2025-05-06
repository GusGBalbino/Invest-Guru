import streamlit as st
import os
from pathlib import Path
import sys
import logging

# Adiciona o diretório do projeto ao PATH para importações relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import save_uploaded_file
from ingest.ingest_pdf import PDFProcessor

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_section():
    """Componente de upload de PDFs para a interface Streamlit."""
    st.header("📤 Upload de Documentos")
    
    # Configurações do processador de PDF
    with st.expander("⚙️ Configurações", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            embedding_model = st.selectbox(
                "Modelo de Embedding",
                options=["huggingface", "openai"],
                index=0,
                key="upload_embedding_model"
            )
            
            # Se OpenAI for selecionado, habilitar campo de API Key
            openai_api_key = None
            if embedding_model == "openai":
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Chave de API da OpenAI para embeddings"
                )
        
        with col2:
            chunk_size = st.number_input(
                "Tamanho do Chunk",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Tamanho de cada chunk em caracteres"
            )
            
            chunk_overlap = st.number_input(
                "Sobreposição do Chunk",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Quantidade de sobreposição entre chunks"
            )
    
    # Upload de PDF
    uploaded_file = st.file_uploader(
        "Selecione um arquivo PDF para adicionar à base de conhecimento",
        type="pdf",
        help="Apenas arquivos PDF são aceitos"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Arquivo selecionado: **{uploaded_file.name}**")
        
        with col2:
            process_btn = st.button("Processar PDF", type="primary", use_container_width=True)
        
        if process_btn:
            with st.spinner("Processando o arquivo PDF..."):
                try:
                    # Salvar o arquivo enviado
                    file_path = save_uploaded_file(uploaded_file)
                    
                    if file_path:
                        # Inicializar o processador com as configurações selecionadas
                        processor = PDFProcessor(
                            embedding_model_type=embedding_model,
                            openai_api_key=openai_api_key,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        # Processar o PDF
                        num_chunks = processor.process_pdf(file_path)
                        
                        if num_chunks > 0:
                            st.success(f"✅ Arquivo processado com sucesso! {num_chunks} chunks adicionados.")
                            st.session_state.document_processed = True
                        else:
                            st.warning("⚠️ Nenhum conteúdo foi extraído do PDF.")
                    else:
                        st.error("❌ Falha ao salvar o arquivo.")
                
                except Exception as e:
                    logger.error(f"Erro ao processar o PDF: {e}")
                    st.error(f"❌ Erro ao processar o PDF: {str(e)}")

def document_management_section():
    """Componente para gerenciar documentos carregados."""
    st.header("📚 Documentos Carregados")
    
    # Cria uma instância do processador para obter a lista de documentos
    try:
        embedding_model = "huggingface"  # Padrão para não precisar de API key
        processor = PDFProcessor(embedding_model_type=embedding_model)
        
        # Obter lista de documentos carregados
        loaded_docs = processor.get_loaded_sources()
        
        if not loaded_docs:
            st.info("Nenhum documento carregado ainda. Faça upload de PDFs na seção acima.")
        else:
            # Exibir a tabela de documentos
            st.dataframe(
                loaded_docs,
                column_config={
                    "name": "Nome do Arquivo",
                    "module": "Módulo",
                    "pages": "Páginas",
                    "chunk_count": "Chunks"
                },
                use_container_width=True
            )
            
            # Seleção de documento para remover
            selected_doc = st.selectbox(
                "Selecione um documento para remover",
                options=[doc["name"] for doc in loaded_docs],
                key="document_to_remove"
            )
            
            if st.button("🗑️ Remover Documento", type="secondary"):
                with st.spinner(f"Removendo {selected_doc}..."):
                    removed_count = processor.delete_by_source(selected_doc)
                    if removed_count > 0:
                        st.success(f"✅ Documento {selected_doc} removido com sucesso!")
                        st.rerun()  # Recarregar a página para atualizar a lista
                    else:
                        st.error(f"❌ Não foi possível remover o documento {selected_doc}.")
    
    except Exception as e:
        logger.error(f"Erro ao gerenciar documentos: {e}")
        st.error(f"❌ Erro ao carregar documentos: {str(e)}")
        
    # Botão para processar todos os PDFs no diretório
    if st.button("🔄 Reindexar Todos os PDFs"):
        with st.spinner("Reindexando todos os documentos..."):
            try:
                # Verificar se o diretório de PDFs existe
                pdf_dir = Path("data/pdfs")
                
                if not pdf_dir.exists() or not any(pdf_dir.glob("*.pdf")):
                    st.warning("Nenhum PDF encontrado na pasta data/pdfs.")
                else:
                    # Recriar o vectorstore do zero
                    import shutil
                    index_dir = Path("data/index")
                    if index_dir.exists():
                        shutil.rmtree(index_dir)
                    
                    # Inicializar o processador e processar o diretório
                    processor = PDFProcessor(embedding_model_type=embedding_model)
                    results = processor.process_directory(str(pdf_dir))
                    
                    if results:
                        total_chunks = sum(results.values())
                        st.success(f"✅ Reindexação concluída! {len(results)} PDFs processados com {total_chunks} chunks.")
                        st.rerun()  # Recarregar a página para atualizar a lista
                    else:
                        st.warning("⚠️ Nenhum documento foi processado.")
            
            except Exception as e:
                logger.error(f"Erro na reindexação: {e}")
                st.error(f"❌ Erro ao reindexar documentos: {str(e)}") 