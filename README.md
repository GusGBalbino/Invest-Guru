# 📊 RAG Local com PDFs de Investimentos

Sistema de perguntas e respostas baseado em Retrieval-Augmented Generation (RAG) para PDFs sobre investimentos.

## 🚀 Funcionalidades

- Responde perguntas sobre investimentos com base em PDFs
- Upload dinâmico de novos PDFs 
- Visualização e gerenciamento dos documentos carregados
- Contexto das respostas com referência aos documentos

## 📋 Requisitos

- Python 3.9+
- Pacotes listados em `requirements.txt`
- Chave de API da OpenAI (se usando o modelo remoto)

## 🔧 Instalação

1. Clone o repositório
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
- Crie um arquivo `.env` na raiz do projeto
- Adicione sua chave de API da OpenAI (se necessário):
  ```
  OPENAI_API_KEY=sua_chave_aqui
  ```

## 🏃‍♂️ Executando o sistema

```bash
streamlit run app/main.py
```

## 🗂️ Estrutura do projeto

```
rag_investimentos/
│
├── app/                       # Código da aplicação Streamlit
│   ├── chat.py                # Lógica do chat
│   ├── upload.py              # Upload de PDFs
│   ├── utils.py               # Funções auxiliares
│   └── main.py                # Streamlit App
│
├── data/
│   ├── pdfs/                  # PDFs carregados
│   └── index/                 # Vectorstore persistido (ex: chroma DB)
│
├── ingest/
│   └── ingest_pdf.py          # Pipeline de ingestão e indexação
│
├── requirements.txt
└── README.md
``` 