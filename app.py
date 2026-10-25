"""
Streamlit Web App: Voice of Customer Intelligence com RAG & Simulador de SAC
FIAP - Pós-Tech em IA e Ciência de Dados (Tech Challenge - Fase 4)
"""

import sys
from pathlib import Path
from datetime import datetime
import streamlit as st

# Configuração de caminhos do projeto
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag import (
    carregar_vectorstore, 
    consultar_voz_do_cliente, 
    classificar_sentimento, 
    gerar_resposta_sac
)

# 1. Configurações da Página Streamlit
st.set_page_config(
    page_title="Voice of Customer Intelligence | Olist VoC",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 12px;
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


# 2. Carregamento dos Modelos com Cache de Recurso
@st.cache_resource(show_spinner="Carregando Base Vetorial ChromaDB...")
def obter_vectorstore():
    return carregar_vectorstore()

try:
    vectorstore = obter_vectorstore()
except Exception as e:
    vectorstore = None
    st.error(f"Erro ao carregar banco vetorial: {e}")


# 3. Barra Lateral: Filtros e Informações
with st.sidebar:
    st.title("🛍️ VoC Intelligence")
    st.caption("**FIAP Pós-Tech AI Scientist** | Tech Challenge - Fase 4")
    st.markdown("---")
    
    st.subheader("⚙️ Filtros Contextuais (RAG)")
    
    tipo_filtro_nota = st.selectbox(
        "Filtrar por Nota de Satisfação:",
        [
            "Todas as notas (1 a 5)",
            "Apenas Insatisfeitos (Notas 1 e 2)",
            "Apenas Neutros (Nota 3)",
            "Apenas Satisfeitos (Notas 4 e 5)",
            "Apenas Nota Máxima (Nota 5)",
            "Personalizado"
        ],
        index=0
    )
    
    nota_min = None
    nota_max = None
    if tipo_filtro_nota == "Apenas Insatisfeitos (Notas 1 e 2)":
        nota_max = 2
    elif tipo_filtro_nota == "Apenas Neutros (Nota 3)":
        nota_min = 3
        nota_max = 3
    elif tipo_filtro_nota == "Apenas Satisfeitos (Notas 4 e 5)":
        nota_min = 4
    elif tipo_filtro_nota == "Apenas Nota Máxima (Nota 5)":
        nota_min = 5
    elif tipo_filtro_nota == "Personalizado":
        faixa = st.slider("Selecione a faixa de notas:", 1, 5, (1, 5))
        nota_min = faixa[0]
        nota_max = faixa[1]
        
    filtro_logistica = st.selectbox(
        "Filtro de Logística / Prazo:",
        ["Todos os pedidos", "Apenas pedidos com atraso", "Apenas pedidos no prazo"],
        index=0
    )
    apenas_com_atraso = None
    if filtro_logistica == "Apenas pedidos com atraso":
        apenas_com_atraso = True
    elif filtro_logistica == "Apenas pedidos no prazo":
        apenas_com_atraso = False
        
    categorias_populares = [
        "Todas",
        "cama_mesa_banho",
        "beleza_saude",
        "esporte_lazer",
        "moveis_decoracao",
        "informatica_acessorios",
        "utilidades_domesticas",
        "relogios_presentes",
        "telefonia",
        "automotivo",
        "brinquedos",
        "ferramentas_jardim"
    ]
    categoria_selecionada = st.selectbox("Categoria de Produto:", categorias_populares, index=0)
    
    k_evidencias = st.slider("Qtd. de Avaliações Recuperadas (Top-K):", min_value=3, max_value=10, value=5)
    
    st.markdown("---")
    st.subheader("💡 Perguntas Rápidas")
    
    perguntas_sugeridas = [
        "Quais os principais motivos de insatisfação dos clientes?",
        "O que os clientes relatam sobre problemas com a entrega?",
        "O que os consumidores mais elogiam na experiência de compra?",
        "Qual é a previsão do tempo para amanhã? (Anti-Alucinação)"
    ]
    
    pergunta_clicada = None
    for p in perguntas_sugeridas:
        if st.button(p, use_container_width=True):
            pergunta_clicada = p


# 4. Estrutura Principal em Abas
aba_rag, aba_sac = st.tabs([
    "💬 Voice of Customer RAG", 
    "🧪 Simulador de SAC & Classificador de Sentimento"
])


# ==============================================================================
# ABA 1: CHAT RAG E EXPLORAÇÃO COM EVIDÊNCIAS
# ==============================================================================
with aba_rag:
    st.title("🎯 Voice of Customer Intelligence")
    st.markdown("""
    Explore dores, elogios e percepções reais de clientes com **IA Generativa fundamentada em evidências auditáveis**.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Base de Dados", "37k+ Reviews")
    with col2:
        st.metric("🧠 Modelo LLM", "Gemini 3.5 Flash-Lite")
    with col3:
        st.metric("🎯 Filtro Ativo", tipo_filtro_nota.split('(')[0])
    with col4:
        st.metric("🚚 Logística", "Com Atraso" if apenas_com_atraso else "Geral")

    st.markdown("---")

    # Histórico de mensagens
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "evidencias" in msg and msg["evidencias"]:
                with st.expander(f"🔍 Ver {len(msg['evidencias'])} Evidências e Scores de Similaridade"):
                    for idx, doc in enumerate(msg["evidencias"], 1):
                        meta = doc.metadata
                        estrelas = "⭐" * int(meta.get("review_score", 0))
                        atraso_str = f"Sim ({meta.get('dias_atraso', 0):.0f} dias)" if meta.get("entregue_com_atraso") else "Não"
                        score_sim = meta.get("similarity_score", 0.0)
                        
                        st.markdown(f"**Evidência #{idx}** | Review ID: `{meta.get('review_id')}` | Nota: {meta.get('review_score')} {estrelas}")
                        st.caption(f"Relevância Semântica: **{score_sim}%** | Categoria: `{meta.get('product_category_name')}` | Atrasou: `{atraso_str}`")
                        st.progress(score_sim / 100.0)
                        st.info(f"\"{doc.page_content}\"")
                        st.markdown("---")

    # Input do Usuário
    user_input = st.chat_input("Faça uma pergunta sobre as avaliações dos clientes...")
    prompt_final = user_input or pergunta_clicada

    if prompt_final:
        st.session_state.messages.append({"role": "user", "content": prompt_final})
        with st.chat_message("user"):
            st.markdown(prompt_final)
            
        with st.chat_message("assistant"):
            with st.spinner("Buscando evidências e gerando análise executiva..."):
                try:
                    resposta, docs = consultar_voz_do_cliente(
                        pergunta=prompt_final,
                        nota_maxima=nota_max,
                        nota_minima=nota_min,
                        apenas_com_atraso=apenas_com_atraso,
                        categoria=categoria_selecionada,
                        k=k_evidencias,
                        vectorstore=vectorstore
                    )
                    
                    st.markdown(resposta)
                    
                    # Exibir Evidências com Relevância
                    if docs:
                        with st.expander(f"🔍 Ver {len(docs)} Evidências e Scores de Similaridade"):
                            for idx, doc in enumerate(docs, 1):
                                meta = doc.metadata
                                estrelas = "⭐" * int(meta.get("review_score", 0))
                                atraso_str = f"Sim ({meta.get('dias_atraso', 0):.0f} dias)" if meta.get("entregue_com_atraso") else "Não"
                                score_sim = meta.get("similarity_score", 0.0)
                                
                                st.markdown(f"**Evidência #{idx}** | Review ID: `{meta.get('review_id')}` | Nota: {meta.get('review_score')} {estrelas}")
                                st.caption(f"Relevância Semântica: **{score_sim}%** | Categoria: `{meta.get('product_category_name')}` | Atrasou: `{atraso_str}`")
                                st.progress(score_sim / 100.0)
                                st.info(f"\"{doc.page_content}\"")
                                st.markdown("---")
                                
                    # Botão para Exportar Relatório Executivo
                    relatorio_md = f"""# Relatório Executivo - Voice of Customer Intelligence
**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  
**Pergunta Analisada:** {prompt_final}  
**Filtros Utilizados:** Nota (min={nota_min}, max={nota_max}) | Atraso={apenas_com_atraso} | Categoria={categoria_selecionada}

---

## Síntese Executiva
{resposta}

---

## Evidências Auditadas ({len(docs)} avaliações recuperadas)
"""
                    for idx, doc in enumerate(docs, 1):
                        meta = doc.metadata
                        relatorio_md += f"""
### Evidência #{idx}
- **Review ID:** {meta.get('review_id')}
- **Nota:** {meta.get('review_score')} estrelas
- **Categoria:** {meta.get('product_category_name')}
- **Relevância Semântica:** {meta.get('similarity_score', 0)}%
- **Comentário do Cliente:** "{doc.page_content}"
"""

                    st.download_button(
                        label="📥 Baixar Relatório Executivo (Markdown)",
                        data=relatorio_md,
                        file_name=f"relatorio_voc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resposta,
                        "evidencias": docs
                    })
                    
                except Exception as e:
                    st.error(f"Erro ao processar consulta: {e}")


# ==============================================================================
# ABA 2: SIMULADOR DE SAC E CLASSIFICADOR EM TEMPO REAL
# ==============================================================================
with aba_sac:
    st.title("🧪 Simulador de SAC & Triagem de Sentimentos")
    st.markdown("""
    Esta funcionalidade integra o **modelo Scikit-Learn (TF-IDF + Logistic Regression)** treinado no notebook de EDA 
    com o **Google Gemini** para classificar novas mensagens e redigir respostas humanizadas de atendimento ao cliente.
    """)
    
    st.markdown("---")
    
    st.subheader("1. Testar Avaliação de Cliente")
    
    exemplos_sac = [
        "Comprei há mais de 20 dias, o prazo venceu semana passada e até agora nem sinal do produto! Quero meu dinheiro de volta.",
        "Recebi super rápido, antes do prazo previsto! Muito bem embalado e funcionando perfeitamente. Recomendo muito!",
        "O produto até funciona, mas a cor veio diferente do anúncio e a embalagem estava um pouco amassada."
    ]
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    texto_exemplo = ""
    with col_ex1:
        if st.button("Exemplo: Reclamação Grave"):
            texto_exemplo = exemplos_sac[0]
    with col_ex2:
        if st.button("Exemplo: Elogio de Entrega"):
            texto_exemplo = exemplos_sac[1]
    with col_ex3:
        if st.button("Exemplo: Divergência"):
            texto_exemplo = exemplos_sac[2]
            
    texto_analise = st.text_area(
        "Cole ou digite a mensagem do cliente:", 
        value=texto_exemplo,
        height=120,
        placeholder="Digite o comentário do cliente aqui..."
    )
    
    if st.button("🔍 Analisar Sentimento & Gerar Resposta SAC", type="primary"):
        if not texto_analise.strip():
            st.warning("Por favor, insira um texto para análise.")
        else:
            with st.spinner("Classificando com Scikit-Learn e gerando resposta com Gemini..."):
                resultado = classificar_sentimento(texto_analise)
                classe = resultado["classe"]
                confianca = resultado["probabilidade"]
                
                # Exibir Resultado da Classificação
                st.markdown("### 📊 Resultado da Classificação (Scikit-Learn)")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    if classe == "positiva":
                        st.success(f"🟢 **Sentimento: POSITIVO**\n\nConfiança: **{confianca}%**")
                    else:
                        st.error(f"🔴 **Sentimento: NEGATIVO**\n\nConfiança: **{confianca}%**")
                
                with col_res2:
                    st.caption("Distribuição de Probabilidade do Modelo:")
                    for c_name, c_prob in resultado["classes"].items():
                        st.write(f"**{c_name.capitalize()}:** {c_prob}%")
                        st.progress(c_prob / 100.0)
                        
                st.markdown("---")
                
                # Geração da Resposta com Gemini
                st.markdown("### ✨ Sugestão de Resposta de SAC (Google Gemini)")
                resposta_sac = gerar_resposta_sac(texto_analise, classe)
                st.text_area("Resposta Pronta para Envio:", value=resposta_sac, height=200)
                
                st.success("✅ Resposta gerada com sucesso! Você pode copiar e enviar diretamente ao cliente.")
