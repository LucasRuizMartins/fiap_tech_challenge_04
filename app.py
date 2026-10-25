"""
Streamlit Web App: Voice of Customer Intelligence com RAG
FIAP - Pós-Tech em IA e Ciência de Dados (Tech Challenge - Fase 4)
"""

import sys
from pathlib import Path
import streamlit as st

# Configuração de caminhos do projeto
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag import carregar_vectorstore, consultar_voz_do_cliente

# 1. Configurações da Página Streamlit
st.set_page_config(
    page_title="Voice of Customer RAG | Olist Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada para dar acabamento premium
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4f46e5;
        margin-bottom: 10px;
    }
    .evidence-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# 2. Carregar o Vector Store com Cache de Recurso (uma única vez)
@st.cache_resource(show_spinner="Carregando Base Vetorial ChromaDB...")
def obter_vectorstore():
    return carregar_vectorstore()

try:
    vectorstore = obter_vectorstore()
    db_status = True
except Exception as e:
    vectorstore = None
    db_status = False
    st.error(f"Erro ao carregar banco vetorial: {e}")


# 3. Barra Lateral: Informações do Projeto e Filtros Contextuais
with st.sidebar:
    st.title("🛍️ VoC Intelligence")
    st.caption("**FIAP Pós-Tech AI Scientist** | Tech Challenge - Fase 4")
    st.markdown("---")
    
    st.subheader("⚙️ Filtros Contextuais de Metadados")
   
    
    # Filtro de Sentimento / Nota
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
        
    # Filtro de Logística
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
        
    # Filtro por Categoria de Produto
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
    
    # Quantidade de Evidências Top-K
    k_evidencias = st.slider("Qtd. de Avaliações Recuperadas (k):", min_value=3, max_value=10, value=5)
    
    st.markdown("---")
    st.subheader("💡 Perguntas Rápidas (Oficiais)")
    
    perguntas_sugeridas = [
        "Quais os principais motivos de insatisfação dos clientes?",
        "O que os clientes relatam sobre problemas com a entrega?",
        "O que os consumidores mais elogiam na experiência de compra?",
        "Qual é a previsão do tempo para amanhã? (Teste de Alucinação)"
    ]
    
    pergunta_clicada = None
    for p in perguntas_sugeridas:
        if st.button(p, use_container_width=True):
            pergunta_clicada = p


# 4. Painel Principal
st.title("🎯 Voice of Customer Intelligence com RAG")
st.markdown("""
Explore o sentimento, dores e percepções de clientes do e-commerce Olist através de **IA Generativa fundamentada em evidências reais**.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📦 Base de Conhecimento", "37k+ Reviews (Silver)")
with col2:
    st.metric("🧠 Modelo LLM", "Gemini 3.5 Flash-Lite (~1s)")
with col3:
    st.metric("🛡️ Rastreabilidade", "100% Citações de IDs")

st.markdown("---")

# Inicialização do histórico de mensagens na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "evidencias" in msg and msg["evidencias"]:
            with st.expander(f"🔍 Ver {len(msg['evidencias'])} Evidências Utilizadas"):
                for idx, doc in enumerate(msg["evidencias"], 1):
                    meta = doc.metadata
                    estrelas = "⭐" * int(meta.get("review_score", 0))
                    atraso_str = f"Sim ({meta.get('dias_atraso', 0):.0f} dias)" if meta.get("entregue_com_atraso") else "Não"
                    
                    st.markdown(f"""
                    **Evidência #{idx}** | Review ID: `{meta.get('review_id')}` | Nota: {meta.get('review_score')} {estrelas}  
                    *Categoria:* `{meta.get('product_category_name')}` | *Atrasou:* `{atraso_str}`  
                    > {doc.page_content}
                    ---
                    """)

# Entrada do Usuário (via chat input ou botão lateral clicado)
user_input = st.chat_input("Faça uma pergunta sobre as avaliações dos clientes...")
prompt_final = user_input or pergunta_clicada

if prompt_final:
    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt_final})
    with st.chat_message("user"):
        st.markdown(prompt_final)
        
    # Gerar a resposta com o RAG
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
                
                # Exibir as evidências recuperadas em painel expansível
                if docs:
                    with st.expander(f"🔍 Ver {len(docs)} Evidências Utilizadas"):
                        for idx, doc in enumerate(docs, 1):
                            meta = doc.metadata
                            estrelas = "⭐" * int(meta.get("review_score", 0))
                            atraso_str = f"Sim ({meta.get('dias_atraso', 0):.0f} dias)" if meta.get("entregue_com_atraso") else "Não"
                            
                            st.markdown(f"""
                            **Evidência #{idx}** | Review ID: `{meta.get('review_id')}` | Nota: {meta.get('review_score')} {estrelas}  
                            *Categoria:* `{meta.get('product_category_name')}` | *Atrasou:* `{atraso_str}`  
                            > {doc.page_content}
                            ---
                            """)
                
                # Salvar resposta no histórico da sessão
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resposta,
                    "evidencias": docs
                })
                
            except Exception as e:
                st.error(f"Erro ao processar consulta: {e}")
