import streamlit as st
import os
import sys
import logging
from pathlib import Path
import time

# Adiciona o diretório do projeto ao PATH para importações relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import get_llm_model, format_sources
from ingest.ingest_pdf import PDFProcessor
from dotenv import load_dotenv

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain


from langchain_core.prompts import PromptTemplate


# Carregar variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def setup_qa_chain(api_key, model_name="gpt-3.5-turbo", embedding_model="huggingface"):
    """Configura a cadeia de QA com base nos parâmetros selecionados."""
    try:
        index_path = Path("data/index")
        if not index_path.exists():
            return None, "O índice de documentos não existe. Carregue documentos primeiro."

        processor = PDFProcessor(
            embedding_model_type=embedding_model,
            openai_api_key=api_key if embedding_model == "openai" else None
        )

        retriever = processor.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        llm = get_llm_model(model_name, api_key)

        # Atual: o prompt se chama `question_prompt` na nova versão
        question_prompt = PromptTemplate.from_template("""
        Você está atuando como um reformulador de perguntas em um sistema de chat sobre investimentos.

        Dado o histórico da conversa e a pergunta atual, reformule a pergunta de forma **independente**, clara e objetiva, garantindo que ela **possa ser compreendida fora do contexto**.

        Não altere o significado, apenas reescreva para que fique completa e autocontida.

        HISTÓRICO DO CHAT:
        {chat_history}

        PERGUNTA ATUAL:
        {input}

        PERGUNTA REFORMULADA:
        """)

        retriever_with_history = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=question_prompt
        )

        qa_prompt = PromptTemplate.from_template("""
        Você é um assistente especializado em finanças pessoais e investimentos. 
        Seu objetivo é ajudar o usuário a compreender conceitos com base nos documentos fornecidos.

        ⚠️ Regras importantes:
        - **Nunca** invente informações.
        - **Nunca** dê sugestões de investimento, recomendação de compra ou previsão de mercado.
        - Use **somente** as informações disponíveis nos documentos.
        - Caso a resposta não esteja clara nos documentos, diga honestamente que **não há dados suficientes para responder**.
        - Seja didático, claro e objetivo. Assuma que a pessoa é leiga no assunto.
        - Sempre que possível, use analogias simples e organize a resposta em etapas ou tópicos.
        - Ao explicar algo, seja detalhista em como realizar aquilo, com exemplos.

        DOCUMENTOS FORNECIDOS:
        {context}

        PERGUNTA DO USUÁRIO:
        {input}

        RESPOSTA DETALHADA:
        """)

        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=qa_prompt
        )

        qa_chain = create_retrieval_chain(
            retriever=retriever_with_history,
            combine_docs_chain=combine_docs_chain
        )

        return qa_chain, None

    except Exception as e:
        logger.error(f"Erro ao configurar a cadeia de QA: {e}")
        return None, f"Erro ao configurar o sistema: {str(e)}"


def chat_section():
    """Componente de chat para a interface Streamlit."""
    st.header("💬 Pergunte ao Guru")

    
    # Configurações do modelo de linguagem
    with st.expander("⚙️ Configurações do Agente", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            llm_model = st.selectbox(
                "Modelo LLM",
                options=["gpt-3.5-turbo", "gpt-4o"],
                index=0,
                key="chat_llm_model"
            )
            
            # OpenAI API Key
            api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                help="Chave de API da OpenAI para o modelo de linguagem"
            )
        
        with col2:
            embedding_model = st.selectbox(
                "Modelo de Embedding",
                options=["huggingface", "openai"],
                index=0,
                key="chat_embedding_model"
            )
    
    # Verificar se já temos documentos carregados
    index_path = Path("data/index")
    if not index_path.exists() or not any(os.listdir(index_path)) if index_path.exists() else True:
        st.warning("⚠️ Nenhum documento carregado. Por favor, adicione PDFs na aba de Upload primeiro.")
        return
    
    # Inicializar a cadeia de QA
    qa_chain, error = setup_qa_chain(api_key, llm_model, embedding_model)
    
    if error:
        st.error(f"❌ {error}")
        return
    
    # Inicializa o histórico se ainda não existir
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Campo de entrada do usuário
    user_query = st.chat_input("Pergunte ao guru...")

    # Exibir mensagens anteriores (Mais recentes em cima)
    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Se for uma resposta do assistente, mostrar as fontes
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("🔍 Ver Fontes"):
                    st.markdown(message["sources"])
    
    if user_query:
        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Mostrar a mensagem do usuário na interface
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Preparar resposta do assistente
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("Buscando informações..."):
                    # Executar a consulta na cadeia de QA
                    chat_history = [(msg["content"], res["content"]) 
                                    for msg, res in zip(st.session_state.messages[::2], st.session_state.messages[1::2]) 
                                    if msg["role"] == "user" and res["role"] == "assistant"]
                    
                    start_time = time.time()
                    response = qa_chain.invoke({"input": user_query, "chat_history": chat_history})
                    end_time = time.time()

                    print(f'RESPONSE: {response}')
                    
                    answer = response["answer"]
                    source_documents = response["context"]
                    
                    # Formatar as fontes para exibição
                    if not source_documents:
                        sources_text = "_Nenhuma fonte foi usada._"
                    else:
                        sources_text = format_sources(source_documents)
                    
                    # Exibir a resposta
                    message_placeholder.markdown(answer)
                    
                    # Exibir as fontes em um expander
                    with st.expander("🔍 Ver Fontes"):
                        st.markdown(sources_text)
                        st.caption(f"Tempo de resposta: {end_time - start_time:.2f} segundos")
                    
                    # Adicionar resposta ao histórico
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_text
                    })
            
            except Exception as e:
                logger.error(f"Erro durante a consulta: {e}")
                message_placeholder.error(f"❌ Erro durante a consulta: {str(e)}")
                # Adicionar mensagem de erro ao histórico
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}",
                    "sources": "Nenhuma fonte disponível devido ao erro."
                })

def clear_chat_history():
    """Limpa o histórico de chat."""
    st.session_state.messages = []
    st.session_state.chat_history = [] 