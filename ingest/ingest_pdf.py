import os
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(
        self, 
        embedding_model_type: str = "openai",
        openai_api_key: Optional[str] = None,
        hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "data/index",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configurar embeddings
        if embedding_model_type.lower() == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key é necessária para embeddings da OpenAI")
            self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            logger.info("Usando embeddings da OpenAI")
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=hf_model_name)
            logger.info(f"Usando embeddings do HuggingFace: {hf_model_name}")
        
        # Inicializar text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # Inicializar ou carregar o vectorstore
        if os.path.exists(self.persist_directory):
            self.db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
            logger.info(f"Vectorstore carregado de {self.persist_directory}")
        else:
            self.db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
            logger.info(f"Novo vectorstore criado em {self.persist_directory}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extrai texto de um PDF com metadados de página e módulo."""
        logger.info(f"Extraindo texto de: {pdf_path}")
        
        # Determinar número do módulo a partir do nome do arquivo
        filename = os.path.basename(pdf_path)
        # Procura por um número após a palavra "Módulo" ou "MÓDULO"
        module_match = re.search(r'[Mm][Óó][Dd][Uu][Ll][Oo]\s*(\d+)', filename)
        module_number = module_match.group(1) if module_match else "Desconhecido"
        
        document = fitz.open(pdf_path)
        pages_text = []
        
        for page_num, page in enumerate(document):
            text = page.get_text()
            if text.strip():  # Se a página tem texto
                pages_text.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "module": f"Módulo {module_number}"
                    }
                })
        
        document.close()
        logger.info(f"Extraídas {len(pages_text)} páginas com texto de {filename}")
        return pages_text
    
    def chunk_texts(self, pages_text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Divide o texto em chunks com metadados preservados."""
        all_chunks = []
        
        for page in pages_text:
            content = page["content"]
            metadata = page["metadata"]
            
            chunks = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata]
            )
            
            all_chunks.extend([{
                "content": chunk.page_content,
                "metadata": chunk.metadata
            } for chunk in chunks])
        
        logger.info(f"Texto dividido em {len(all_chunks)} chunks")
        return all_chunks
    
    def add_to_vectorstore(self, chunks: List[Dict[str, Any]]) -> int:
        """Adiciona chunks ao vectorstore."""
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        ids = self.db.add_texts(
            texts=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Adicionados {len(ids)} chunks ao vectorstore")
        return len(ids)
    
    def process_pdf(self, pdf_path: str) -> int:
        """Processa um PDF do início ao fim, retorna número de chunks adicionados."""
        pages_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_texts(pages_text)
        num_added = self.add_to_vectorstore(chunks)
        return num_added
    
    def process_directory(self, directory_path: str) -> Dict[str, int]:
        """Processa todos os PDFs em um diretório."""
        results = {}
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                logger.info(f"Processando {filename}...")
                num_chunks = self.process_pdf(pdf_path)
                results[filename] = num_chunks
        
        return results
    
    def delete_by_source(self, source_name: str) -> int:
        """Remove documentos do vectorstore baseado no nome do arquivo."""
        try:
            # Consultar os IDs dos documentos com o filtro do nome do arquivo
            docs_to_delete = self.db.get(
                where={"source": source_name}
            )
            
            if docs_to_delete and hasattr(docs_to_delete, 'ids') and docs_to_delete.ids:
                # Remover documentos pelo ID
                self.db.delete(ids=docs_to_delete.ids)
                
                logger.info(f"Removidos {len(docs_to_delete.ids)} documentos de {source_name}")
                return len(docs_to_delete.ids)
            else:
                logger.warning(f"Nenhum documento encontrado para {source_name}")
                return 0
        except Exception as e:
            logger.error(f"Erro ao remover documentos: {e}")
            return 0
    
    def get_loaded_sources(self) -> List[Dict[str, Any]]:
        """Retorna uma lista com informações sobre os arquivos carregados."""
        try:
            all_metadatas = self.db.get()["metadatas"]
            sources = {}
            
            for metadata in all_metadatas:
                if "source" in metadata and "module" in metadata:
                    source_name = metadata["source"]
                    
                    if source_name not in sources:
                        sources[source_name] = {
                            "name": source_name,
                            "module": metadata["module"],
                            "pages": set([metadata.get("page", "N/A")]),
                            "chunk_count": 1
                        }
                    else:
                        sources[source_name]["pages"].add(metadata.get("page", "N/A"))
                        sources[source_name]["chunk_count"] += 1
            
            # Converter para lista e formatar para exibição
            result = []
            for source_name, info in sources.items():
                result.append({
                    "name": info["name"],
                    "module": info["module"],
                    "pages": len(info["pages"]),
                    "chunk_count": info["chunk_count"]
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Erro ao obter fontes carregadas: {e}")
            return []

if __name__ == "__main__":
    # Exemplo de uso
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Você pode usar embeddings da OpenAI ou do HuggingFace
    # Para OpenAI:
    # processor = PDFProcessor(embedding_model_type="openai", openai_api_key=openai_api_key)
    
    # Para HuggingFace:
    processor = PDFProcessor(embedding_model_type="huggingface")
    
    # Processar todos os PDFs no diretório
    results = processor.process_directory("data/pdfs")
    print(f"Resultados: {results}")
    
    # Listar fontes carregadas
    sources = processor.get_loaded_sources()
    print(f"Fontes carregadas: {sources}") 