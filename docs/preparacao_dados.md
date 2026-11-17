# Preparação da base de conhecimento

Código: [`src/voc_rag/ingestion/`](../src/voc_rag/ingestion/) · Execução: `python scripts/build_knowledge_base.py` · Exploração: [`notebooks/01_eda_reviews.ipynb`](../notebooks/01_eda_reviews.ipynb) e [`02_preparacao_base.ipynb`](../notebooks/02_preparacao_base.ipynb)

## 1. Seleção dos dados

| Tabela | Uso |
|---|---|
| `olist_order_reviews_dataset` | **Fonte principal**: título, comentário, nota e data da avaliação |
| `olist_orders_dataset` | Status do pedido, data da compra, datas de entrega real e estimada |
| `olist_order_items_dataset` + `olist_products_dataset` + `product_category_name_translation` | Categoria(s) dos produtos, nº de itens, valor |
| `olist_customers_dataset` | UF do cliente |
| `olist_geolocation`, `olist_sellers`, `olist_order_payments` | **Não usadas**: não acrescentam contexto relevante às perguntas de VoC propostas |

A unidade de conhecimento é a **avaliação**. As demais tabelas entram apenas como **metadados** (left join por `order_id`), sem alterar o número de avaliações.

## 2. Funil de seleção

| Etapa | Avaliações |
|---|---:|
| Total no dataset | 99.224 |
| `review_id` duplicado removido | −814 |
| Sem título e sem comentário (ausentes) | −56.040 |
| Texto sem conteúdo informativo (`.`, `10`, `!!!`) | −419 |
| **Documentos na base de conhecimento** | **41.951** |

Dos 41.951 documentos: 30.591 têm só mensagem, 9.741 título + mensagem e 1.619 só título.

## 3. Limpeza e preparação dos textos

Regras (função `normalize_text`):
- normalização Unicode NFC;
- remoção de caracteres de controle e de quebras de linha `\r\n` (frequentes no dataset);
- colapso de espaços e de pontuação repetida (`!!!!!!` → `!!!`);
- **acentos e caixa são preservados**: o texto é exibido como evidência ao usuário e o modelo de embeddings (E5 multilíngue) foi treinado com texto natural. A normalização sem acento/minúscula é aplicada apenas internamente no BM25 e nas regras de detecção de intenção.

Não removemos stopwords nem aplicamos stemming no texto embedado: modelos de embedding contextuais perdem informação com isso (por exemplo, "**não** recebi" depende da negação).

**Título + mensagem:** combinados em um único texto (`"Péssimo. Não chegou"`). O título costuma ser um resumo valioso ("Recomendo", "Produto falsificado"). Se a mensagem já começa com o título, não duplicamos.

## 4. Tratamento de valores ausentes

| Campo | Situação | Tratamento |
|---|---|---|
| `review_comment_message` / `title` | ~59% sem texto | Avaliações sem nenhum dos dois **saem da base de conhecimento**: não há o que recuperar. Se só um existe, ele é usado. |
| `order_delivered_customer_date` | ausente em pedidos não entregues/cancelados | `delivery_status` = `nao_entregue` ou `cancelado` (derivado do `order_status`) e `delivery_delay_days` = NaN |
| `product_category_name` | ausente em alguns produtos | `"sem_categoria"` (pt) / `"unknown"` (en) |
| `order_purchase_timestamp` | - | se ausente, o período (`year_month`) usa a data da avaliação |
| `customer_state`, `order_status` | raros | `"NA"` / `"desconhecido"` |

## 5. Duplicidades

- **`review_id` repetido (814):** a mesma avaliação aparece ligada a pedidos diferentes (o cliente avaliou uma compra dividida em vários pedidos). Mantemos uma ocorrência para não duplicar evidências.
- **Textos idênticos de reviews diferentes** ("Muito bom", "Recebi o produto"): **permanecem no índice**, porque são opiniões reais de clientes distintos e contam para a frequência. Na **seleção de evidências** eles são colapsados em uma só, com o contador `+N avaliações com texto idêntico`. Assim o contexto ganha diversidade sem perder a noção de volume.

## 6. Organização dos documentos

`data/processed/documents.parquet`: **1 documento = 1 avaliação**, sem chunking (mediana de 53 caracteres e máximo de ~210).

| Coluna | Descrição |
|---|---|
| `doc_id`, `review_id`, `order_id` | identificadores (rastreabilidade até o dado original) |
| `text`, `text_source` | texto limpo e sua origem (título/mensagem) |
| `review_score`, `sentiment` | nota 1–5 e faixa (negativa 1–2, neutra 3, positiva 4–5) |
| `review_date`, `purchase_date`, `year_month` | datas e período |
| `order_status`, `delivery_status`, `delivery_delay_days` | situação da entrega (no prazo / atrasado / não entregue / cancelado) |
| `main_category`, `product_categories(_en)` | categoria predominante e todas as categorias do pedido |
| `customer_state`, `n_items`, `order_value` | contexto do pedido |

**Somente o `text` é embedado.** Os metadados não são concatenados ao texto: isso "contaminaria" a similaridade (todas as avaliações nota 1 ficariam parecidas entre si por causa do token "nota 1"). Eles são usados como **filtros exatos** antes da busca (ver [decisoes_tecnicas.md](decisoes_tecnicas.md#metadados)).

## 7. Viés a considerar na interpretação

Clientes insatisfeitos comentam muito mais: cerca de 77% das notas 1 têm texto, contra cerca de 36% das notas 5. Na base de conhecimento, as avaliações negativas (1–2) são 26% dos documentos, contra 15% no dataset total. Por isso o sistema descreve frequências **em relação à amostra recuperada** e não extrapola percentuais para a base de clientes.
