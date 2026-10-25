"""
Módulo de RAG (Retrieval-Augmented Generation) para Voice of Customer Intelligence.
Responsável por carregar o banco vetorial ChromaDB, aplicar filtros de metadados,
estruturar os prompts de grounding e gerar respostas executivas via Google Gemini.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())

# Diretório padrão da base vetorial ChromaDB
BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_PERSIST_DIR = BASE_DIR / "fase_4" / "models" / "chroma_db"
COLLECTION_NAME = "olist_customer_reviews"


def obter_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Inicializa o modelo de embeddings denso do Google Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não encontrada no arquivo .env!")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


def obter_llm(temperatura: float = 0.1) -> ChatGoogleGenerativeAI:
    """Inicializa o modelo de linguagem Gemini com baixa temperatura para evitar alucinações."""
    api_key = os.getenv("GOOGLE_API_KEY")
    modelo = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
    
    return ChatGoogleGenerativeAI(
        model=modelo,
        temperature=temperatura,
        google_api_key=api_key
    )


def carregar_vectorstore(diretorio_persist: Optional[Path] = None) -> Chroma:
    """Carrega a base vetorial ChromaDB persistida localmente."""
    caminho = diretorio_persist or CHROMA_PERSIST_DIR
    embeddings = obter_embeddings()
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(caminho)
    )


def formatar_documentos_contexto(docs: List[Document]) -> str:
    """Formata os documentos recuperados incluindo os metadados chave para o prompt."""
    if not docs:
        return "Nenhuma avaliação encontrada com os critérios informados."
    
    textos_formatados = []
    for doc in docs:
        review_id = doc.metadata.get("review_id", "N/A")
        nota = doc.metadata.get("review_score", "N/A")
        cat = doc.metadata.get("product_category_name", "N/A")
        atraso = doc.metadata.get("dias_atraso", 0.0)
        atrasou = "Sim" if doc.metadata.get("entregue_com_atraso") else "Não"
        
        texto_bloco = (
            f"[Review ID: {review_id} | Nota: {nota} estrelas | Categoria: {cat} | Entregue com atraso: {atrasou} ({atraso:.0f} dias)]\n"
            f"Avaliação: {doc.page_content.strip()}"
        )
        textos_formatados.append(texto_bloco)
        
    return "\n\n".join(textos_formatados)


def criar_prompt_rag() -> ChatPromptTemplate:
    """Retorna o template de prompt com grounding estrito e citação obrigatória."""
    template = """Você é um especialista em Inteligência de Voz do Cliente (Voice of Customer Intelligence) analisando o e-commerce Olist.
Sua missão é responder à dúvida com base ESTRITAMENTE nas avaliações reais de clientes fornecidas no contexto.

Diretrizes obrigatórias:
1. Fundamente-se APENAS nas avaliações reais listadas no contexto. Não presuma, deduza sem suporte ou invente dados.
2. Sempre cite os identificadores das avaliações [Review ID: ...] que comprovam e sustentam cada ponto levantado.
3. Se o contexto estiver vazio, insuficiente ou a pergunta for fora do contexto de e-commerce e satisfação do cliente, responda estritamente:
   "Não foram encontradas avaliações suficientes na base de conhecimento sobre este assunto."

Contexto com avaliações de clientes:
{context}

Pergunta do Usuário:
{question}

Resposta Executiva Fundamentada (com evidências):"""

    return ChatPromptTemplate.from_template(template)


def consultar_voz_do_cliente(
    pergunta: str,
    nota_maxima: Optional[int] = None,
    nota_minima: Optional[int] = None,
    apenas_com_atraso: Optional[bool] = None,
    categoria: Optional[str] = None,
    k: int = 5,
    vectorstore: Optional[Chroma] = None
) -> Tuple[str, List[Document]]:
    """
    Executa o pipeline RAG completo:
    1. Aplica filtros prévios de metadados no ChromaDB (pre-filtering);
    2. Recupera as k evidências mais similares;
    3. Gera a resposta executiva fundamentada com Gemini;
    4. Retorna a resposta textual e a lista de documentos para visualização detalhada.
    """
    if vectorstore is None:
        vectorstore = carregar_vectorstore()
        
    condicoes: List[Dict[str, Any]] = []
    
    if nota_maxima is not None:
        condicoes.append({"review_score": {"$lte": int(nota_maxima)}})
    if nota_minima is not None:
        condicoes.append({"review_score": {"$gte": int(nota_minima)}})
    if apenas_com_atraso is not None:
        condicoes.append({"entregue_com_atraso": {"$eq": bool(apenas_com_atraso)}})
    if categoria and categoria.lower() != "todas":
        condicoes.append({"product_category_name": {"$eq": str(categoria)}})
        
    # Construção do filtro na sintaxe nativa do ChromaDB
    filtro_chroma = None
    if len(condicoes) == 1:
        filtro_chroma = condicoes[0]
    elif len(condicoes) > 1:
        filtro_chroma = {"$and": condicoes}
        
    search_kwargs = {"k": k}
    if filtro_chroma:
        search_kwargs["filter"] = filtro_chroma
        
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    # 1. Recuperar evidências
    documentos_recuperados = retriever.invoke(pergunta)
    
    if not documentos_recuperados:
        return (
            "Não foram encontradas avaliações na base de conhecimento correspondentes aos critérios e filtros selecionados.",
            []
        )
        
    # 2. Executar a cadeia RAG com LCEL
    contexto_formatado = formatar_documentos_contexto(documentos_recuperados)
    prompt = criar_prompt_rag()
    llm = obter_llm()
    
    chain = prompt | llm | StrOutputParser()
    resposta = chain.invoke({
        "context": contexto_formatado,
        "question": pergunta
    })
    
    return resposta, documentos_recuperados
