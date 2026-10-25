# 🎯 Tech Challenge - Fase 4: Voice of Customer Intelligence com RAG
**FIAP - Pós-Tech em Inteligência Artificial e Ciência de Dados**

Para acessar a documentação completa, diagrama da arquitetura, justificativas técnicas e guia de reprodução, consulte o [README.md principal na raiz do projeto](../../README.md).

---

## 🚀 Resumo Rápido da Execução dos Notebooks

1. **`notebooks/00_s3.ipynb`**: Download do Kaggle e ingestão das camadas Raw e Bronze no AWS S3.
2. **`notebooks/01_EDA.ipynb`**: Limpeza dos textos, cálculo de métricas de entrega/atraso e gravação da base `reviews_enriquecidos.parquet` na camada Silver do S3.
3. **`notebooks/01_rag.ipynb`**: Indexação no ChromaDB, modelo de embeddings `models/gemini-embedding-001`, busca contextual com metadados (notas, categorias e atraso) e geração de respostas fundamentadas via `gemini-3.5-flash-lite`.
