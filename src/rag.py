"""
Módulo de RAG (Retrieval-Augmented Generation) para Voice of Customer Intelligence.
Responsável por carregar o banco vetorial ChromaDB, modelo de classificação Scikit-Learn,
aplicar filtros de metadados, calcular scores de similaridade e gerar respostas via Gemini.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
from dotenv import load_dotenv, find_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())

# Diretórios padrão
BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_PERSIST_DIR = BASE_DIR / "fase_4" / "models" / "chroma_db"
CLASSIFICADOR_PATH = BASE_DIR / "fase_4" / "models" / "classificador_sentimento.pkl"
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
    """Inicializa o modelo de linguagem Gemini Flash-Lite com baixa latência."""
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


def carregar_classificador():
    """Carrega o pipeline treinado de TF-IDF + LogisticRegression do Scikit-Learn."""
    if not CLASSIFICADOR_PATH.exists():
        return None
    return joblib.load(CLASSIFICADOR_PATH)


def classificar_sentimento(texto: str) -> Dict[str, Any]:
    """
    Classifica um comentário novo em 'positiva' ou 'negativa'
    utilizando o modelo treinado em 01_EDA.ipynb.
    """
    modelo = carregar_classificador()
    if modelo is None:
        return {"classe": "indisponível", "probabilidade": 0.0, "erro": "Modelo não encontrado."}
    
    predicao = modelo.predict([texto])[0]
    probabilidades = modelo.predict_proba([texto])[0]
    classes = list(modelo.classes_)
    idx_classe = classes.index(predicao)
    prob_confianca = float(probabilidades[idx_classe]) * 100
    
    return {
        "classe": str(predicao).lower(),
        "probabilidade": round(prob_confianca, 1),
        "classes": {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, probabilidades)}
    }


def gerar_resposta_sac(comentario_cliente: str, sentimento: str) -> str:
    """
    Gera uma sugestão de resposta empática e profissional para a equipe de atendimento (SAC),
    adaptando o tom conforme o sentimento identificado.
    """
    llm = obter_llm(temperatura=0.3)
    
    prompt = ChatPromptTemplate.from_template("""Você é um líder de Atendimento e Experiência do Cliente (Customer Experience) do e-commerce Olist.
Um cliente acabou de enviar a seguinte avaliação classificada como '{sentimento}':

Mensagem do Cliente:
"{comentario}"

Sua tarefa é redigir uma resposta oficial de SAC para esse cliente:
1. Comece de forma empática e respeitosa, chamando-o pelo tom acolhedor da marca.
2. Se a avaliação for negativa: peça desculpas pelo transtorno, demonstre compreensão e proponha uma resolução imediata (abertura de protocolo, verificação com transportadora ou canal de reembolso).
3. Se a avaliação for positiva: agradeça a preferência e reforce o compromisso com a excelência.
4. Mantenha um tom profissional, direto e humano (sem parecer uma resposta robótica).

Resposta Sugerida de SAC:""")
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"comentario": comentario_cliente, "sentimento": sentimento})


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
        sim_score = doc.metadata.get("similarity_score", 0.0)
        
        texto_bloco = (
            f"[Review ID: {review_id} | Nota: {nota} estrelas | Categoria: {cat} | Atrasou: {atrasou} ({atraso:.0f} dias) | Relevância: {sim_score:.1f}%]\n"
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
    Executa o pipeline RAG completo com cálculo de similaridade e filtros contextuais.
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
        
    filtro_chroma = None
    if len(condicoes) == 1:
        filtro_chroma = condicoes[0]
    elif len(condicoes) > 1:
        filtro_chroma = {"$and": condicoes}
        
    # 1. Recuperação com pontuação de similaridade (distância L2 / Cosseno)
    try:
        docs_and_scores = vectorstore.similarity_search_with_score(
            query=pergunta,
            k=k,
            filter=filtro_chroma
        )
    except Exception:
        # Fallback se o filtro não encontrar documentos
        docs_and_scores = []
        
    if not docs_and_scores:
        return (
            "Não foram encontradas avaliações na base de conhecimento correspondentes aos critérios e filtros selecionados.",
            []
        )
        
    # Normalizar distância em score de similaridade (0% a 100%)
    documentos_recuperados = []
    for doc, dist in docs_and_scores:
        # No ChromaDB, quanto menor a distância, maior a similaridade
        score_sim = max(0.0, min(100.0, (1.0 - (float(dist) / 1.5)) * 100))
        doc.metadata["similarity_score"] = round(score_sim, 1)
        doc.metadata["distancia_raw"] = round(float(dist), 4)
        documentos_recuperados.append(doc)
        
    # 2. Executar cadeia RAG com LCEL
    contexto_formatado = formatar_documentos_contexto(documentos_recuperados)
    prompt = criar_prompt_rag()
    llm = obter_llm()
    
    chain = prompt | llm | StrOutputParser()
    resposta = chain.invoke({
        "context": contexto_formatado,
        "question": pergunta
    })
    
    return resposta, documentos_recuperados
