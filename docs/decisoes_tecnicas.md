# Decisões técnicas

Cada decisão traz o contexto, a escolha, a justificativa e as alternativas consideradas.

## 1. Unidade de documento: 1 avaliação, sem chunking

- **Contexto:** comentários com mediana de 53 caracteres e máximo de ~210.
- **Escolha:** cada avaliação é um documento indivisível.
- **Por quê:** não há o que dividir; a avaliação inteira preserva o sentido e dá **rastreabilidade 1:1** (evidência → `review_id`).
- **Alternativa descartada:** agrupar avaliações (por pedido ou produto) em documentos maiores. Isso misturaria opiniões de clientes diferentes e tornaria a citação ambígua.

## 2. Modelo de embeddings: `intfloat/multilingual-e5-small`

- **Escolha:** modelo local multilíngue (384 dimensões, ~118M parâmetros) com os prefixos `query:` e `passage:`.
- **Por quê:**
  - bom desempenho em recuperação em português (família E5 em benchmarks MTEB/MIRACL);
  - foi treinado para **busca assimétrica** (pergunta curta → passagem), exatamente o caso de uso;
  - roda em CPU: indexar ~42 mil textos curtos leva alguns minutos, uma única vez;
  - sem custo por chamada e sem dependência de rede: qualquer avaliador reproduz o resultado.
- **Alternativas:** `multilingual-e5-base/large` (melhor qualidade, de 3 a 10 vezes mais lento; basta trocar `EMBEDDING_MODEL` no `.env` e reindexar); embeddings via API (custo, chave e dados enviados a terceiros); `paraphrase-multilingual-MiniLM` (simétrico, menos adequado a pergunta → passagem).

## 3. Armazenamento e indexação: busca exata em NumPy

- **Escolha:** matriz `float32` normalizada (`embeddings.npy`) alinhada a `documents.parquet`; a similaridade é o produto interno (= cosseno).
- **Por quê:**
  - ~42 mil × 384 ≈ 64 MB: uma consulta exata leva poucos milissegundos;
  - **recall exato**, sem a perda dos índices aproximados (HNSW/IVF);
  - pré-filtragem por metadados exata e trivial (máscara booleana);
  - zero infraestrutura e formato aberto.
- **Quando mudar:** a partir de milhões de vetores, ou com atualização contínua. A interface `VectorStore.search(query_vec, k, mask)` isola a troca para FAISS, Qdrant, Chroma ou pgvector.
- `manifest.json` registra o modelo usado. Carregar um índice com modelo diferente do configurado gera erro, para evitar consultas com vetores incompatíveis.

## 4. Recuperação híbrida: densa + BM25 com RRF

- **Por quê híbrida:** avaliações são curtas e cheias de termos decisivos ("**não** recebi", "quebrado", "cartucho"). Embeddings capturam paráfrases ("demorou uma eternidade" ≈ "atraso"), mas em frases curtas podem dar peso demais ao tom geral. O BM25 garante a correspondência exata desses termos.
- **Por quê RRF:** os scores de cosseno e de BM25 têm escalas incompatíveis. O *Reciprocal Rank Fusion* (`Σ 1/(60 + rank)`) combina apenas as posições, sem calibração, e é robusto.
- **Expansão de consulta (só no BM25):** quando um tema é detectado (ex.: entrega), adicionamos sinônimos ("prazo chegou recebi atraso correios"). O texto embedado continua sendo a pergunta original, porque o E5 lida bem com linguagem natural.
- O BM25 pode ser desligado (`USE_BM25=false`) para comparação; ver `notebooks/03_experimentos_embeddings.ipynb`.

## 5. Transformação da pergunta por regras (sem LLM)

- **Escolha:** regex sobre a pergunta normalizada para extrair filtros (nota, sentimento, categoria, período, entrega atrasada) e temas, mais a **reescrita para o núcleo** da pergunta.
- **Núcleo da pergunta:** perguntas de negócio têm um enquadramento analítico ("Quais padrões podem ser identificados nas avaliações de clientes…") que não se parece com nenhuma avaliação de cliente e dilui a similaridade. Uma lista de palavras de enquadramento é removida, e o restante vira o texto embedado: "O que os clientes relatam sobre problemas relacionados à entrega?" → `problemas entrega`. Na calibração, isso elevou a similaridade das perguntas válidas com as melhores avaliações (de ~0,86–0,90 para ~0,89–0,96). É uma versão determinística da ideia de aproximar a consulta do formato dos documentos (como no HyDE, mas sem chamar o LLM).
- **Por quê:** determinístico, instantâneo, sem custo, **explicável** (cada transformação aparece no trace) e testável (`tests/test_retriever.py`).
- **Decisão de domínio:** perguntas com intenção negativa ("problemas", "reclamações") filtram as notas 1–3, e "insatisfeitos" filtra as notas 1–2. Nas notas 4–5, reclamações são raras e trariam ruído; a nota 3 é mantida porque traz críticas pontuais ("produto bom, mas atrasou").
- **Riscos tratados:** palavras ambíguas foram restringidas ("presentes" não ativa a categoria `relogios_presentes`; "valorizam" não ativa o tema preço; "não recomendo" é negativo). O usuário pode desligar a extração automática.
- **Alternativa:** extração estruturada com o LLM (mais flexível, mas acrescenta uma chamada, latência e custo por pergunta).

## 6. Metadados como pré-filtro, não no texto embedado {#metadados}

- **Problema:** nota, situação da entrega e período **não estão no texto** do comentário. Perguntar "problemas em avaliações nota 1" a um índice puramente semântico recupera textos que *parecem* negativos, sem garantia da nota.
- **Escolha:** filtrar antes da busca, de forma exata.
- **Por que não concatenar no texto** ("nota: 1. categoria: telefonia. texto…"): todos os documentos com o mesmo metadado ficariam artificialmente próximos, e a similaridade passaria a refletir metadados, não a opinião do cliente.
- **Contribuição medida:** com filtro, 100% das evidências respeitam o recorte (`filter_compliance` em `run_eval.py`) e todas as vagas do top-k vão para avaliações do recorte pedido.
- **`delivery_status`** é derivado das datas real e estimada (no prazo, atrasado, não entregue, cancelado). Isso permite perguntar sobre "pedidos atrasados" de forma objetiva, independentemente de o cliente ter mencionado o atraso.

## 7. Seleção: relevância mínima, deduplicação, MMR e top-k = 15

- **Relevância mínima:** candidatos abaixo de `MIN_RELEVANCE` não entram no contexto, mesmo que estejam no top-k da fusão. Melhor um contexto menor que um contexto com ruído.
- **Deduplicação na seleção, não no índice:** textos idênticos de clientes diferentes ("Muito bom") são opiniões reais e contam para o volume. No contexto, viram uma evidência com `+N avaliações com texto idêntico`, para que o modelo veja diversidade e ainda saiba a recorrência.
- **Textos muito curtos no fim da fila:** em perguntas genéricas, os vizinhos mais próximos de "insatisfeitos" são avaliações como "insatisfeito" e "insatisfeita". Elas confirmam o tema, mas não explicam o **porquê**. Textos com menos de 3 palavras só entram se faltarem evidências mais informativas.
- **MMR (Maximal Marginal Relevance, λ = 0,7):** escolhe iterativamente a avaliação que maximiza `λ·relevância − (1−λ)·similaridade com as já escolhidas`. Sem isso, "problemas relatados" trazia 15 variações de "atraso na entrega". Com o MMR, o contexto cobre defeito, atraso, item faltando, produto errado etc. Na avaliação, a precisão@15 subiu de 0,87 para 0,92: as evidências ficaram mais diversas **e** mais pertinentes ao tema.
- **Top-k = 15:** equilibra cobertura de padrões e tamanho de contexto (~1.500 tokens de evidências). Ajustável na interface (5–30).
- **Reranker (cross-encoder) opcional:** melhora a ordenação, mas acrescenta ~1–3 s em CPU por consulta; vem desligado por padrão.

## 8. Contexto com rótulos e resumo da amostra

- Cada evidência é rotulada `[E1]..[En]` com nota, período, categoria e entrega. Os rótulos permitem **citação verificável**; os metadados permitem análises como "as queixas de atraso vêm de pedidos realmente atrasados".
- O **resumo estatístico** (notas, entrega, categorias da amostra) evita que o modelo estime frequências "de cabeça".

## 9. Prompt orientado a grounding

Regras principais: usar só as evidências; citar `[En]` em cada afirmação; expressar frequência **relativa à amostra** (não extrapolar para a base); organizar por padrões; declarar o que não foi possível concluir; responder `SEM_EVIDENCIAS_SUFICIENTES` quando não houver base.

## 10. Guardrails em duas camadas

### Calibração: por que um limiar de similaridade não basta

O primeiro desenho era o clássico: "se a melhor similaridade for menor que X, a pergunta está fora do escopo". A calibração com as perguntas de avaliação mostrou que **isso não funciona com o E5**:

| Pergunta | Esperado | Similaridade com a melhor avaliação |
|---|---|---:|
| Me passe uma receita de bolo de chocolate. | recusar | **0,903** |
| Quais avaliações sustentam … prazo de entrega? | responder | 0,903 |
| O que os clientes acham de pagar com criptomoedas? | recusar | 0,876 |
| Quais reclamações aparecem sobre celulares em 2018? | responder | **0,856** |
| Qual é a capital da Austrália? | recusar | 0,794 |

O E5 produz cossenos altos para qualquer par de textos (anisotropia), e perguntas analíticas ("quais padrões podem ser identificados…") não se parecem com nenhuma avaliação individual. As distribuições se sobrepõem, então nenhum limiar separa as classes. Testamos também o **score do cross-encoder** e o **z-score do top-10 em relação à média do corpus**: nenhum dos dois separou.

### Desenho final: sinais complementares

**Pré-geração** (determinística, não gasta tokens), avaliada nesta ordem:

| # | Sinal | Regra | Status | Captura |
|---|---|---|---|---|
| 1 | Recorte vazio | filtros retornam 0 documentos | `insufficient_evidence` | "seguros e serviços em 2016" |
| 2 | **Classificador de escopo** | margem < 0 **e** suporte lexical < 3 | `out_of_scope` | "capital da Austrália", "receita de bolo", "taxa Selic" |
| 3 | **Vocabulário** | algum termo-chave aparece em menos de 3 avaliações | `insufficient_evidence` | "criptomoedas", "Curitiba" |
| 4 | **Suporte lexical** | menos de 3 avaliações (no recorte) contêm 60% dos termos-chave | `insufficient_evidence` | "entrega por drone", "modo escuro do aplicativo" |
| 5 | Suporte semântico | menos de 3 candidatos com similaridade ≥ 0,85 (não se aplica a perguntas sem tema sobre um recorte filtrado) | `insufficient_evidence` | - |

- **Classificador de escopo** ([scope.py](../src/voc_rag/generation/scope.py)): compara a pergunta com 25 perguntas-exemplo do domínio e 25 de fora dele (média dos 3 vizinhos mais próximos de cada grupo). Comparar pergunta com pergunta é uma tarefa simétrica, que o E5 separa bem: todas as perguntas fora do escopo tiveram margem negativa. Os exemplos **não** incluem perguntas dos conjuntos de avaliação.
- **Por que "margem < 0 **e** sem suporte lexical"**: frases afirmativas como "Recebi tudo certinho" ficam com margem levemente negativa (−0,015), mas aparecem em centenas de avaliações. O suporte lexical evita a recusa indevida.
- **Termos-chave** = o núcleo da pergunta sem palavras de intenção ("problemas", "elogios") e sem as palavras já convertidas em filtro (categoria). Perguntas sem termos-chave ("Quais são os principais problemas?") pulam os sinais 3 e 4.
- O stemming é por prefixo de 5 letras ("atrasada", "atraso" e "atrasou" viram `atras`): simples e suficiente para essa checagem.

**Pós-geração:** o marcador de recusa do LLM vira `insufficient_evidence`; citações a rótulos inexistentes são removidas, com aviso; uma resposta sem nenhuma citação gera aviso de baixa rastreabilidade.

### Resultado e limitações

| | Conjunto de calibração (19) | Hold-out (22) |
|---|---:|---:|
| Decisão correta | 19/19 | 19/22 na 1ª execução → 21/22 após duas correções genéricas |

As correções feitas a partir do hold-out foram: "não" passou a ser palavra de intenção (antes, era considerado um "conceito desconhecido"), e perguntas sem tema sobre um recorte filtrado passaram a dispensar o corte semântico. O erro restante ("realidade **virtual** no site" passa porque "loja **virtual**" é comum) mostra o limite do sinal lexical. Esses casos dependem da segunda barreira, o LLM. As listas de palavras e os exemplos de escopo são específicos deste domínio e precisam ser revistos se o domínio ou o modelo de embedding mudarem.

## 11. LLM: Google Gemini via API

A geração usa o **Google Gemini** (`gemini-3.5-flash-lite`) pelo SDK oficial `google-genai`, com a chamada isolada em [`LLMClient._call_api`](../src/voc_rag/generation/llm.py). Trocar de provedor exige mudar só esse método.

**Por que um modelo hospedado, e não um local:**

- a obediência a instruções de grounding e citação e a capacidade de recusar são os critérios mais pesados da avaliação, e modelos de fronteira são claramente superiores nisso;
- o custo por pergunta é baixo, porque o contexto tem só cerca de 15 avaliações curtas.

**Por que o `gemini-3.5-flash-lite`:**

- **Estabilidade:** versões mais novas (`3.7-flash`, `3.8-flash`) retornaram sobrecarga (503) ou timeout nos testes.
- **Cota:** no plano gratuito, o `gemini-3.5-flash` permite só 20 respostas por dia, e a avaliação com LLM (26 respostas) mais os exemplos (8) já passam disso. O `flash-lite` tem cota separada e maior.
- **Qualidade suficiente para a tarefa:** na avaliação, 100% das respostas citaram evidências válidas (média de 13 a 14 de 15) e o modelo recusou corretamente a única pergunta sem suporte que passou pelos guardrails pré-geração (ver seção 12).
- **Latência:** respostas típicas em poucos segundos.

O `gemini-3.5-flash` continua disponível via `LLM_MODEL` e produz respostas um pouco mais longas. Com ele, vale usar `LLM_THINKING_BUDGET=0`: por padrão, esse modelo "pensa" antes de responder, e medimos cerca de 2.200 tokens de raciocínio interno para 400 de resposta. Desligar o raciocínio reduziu a latência de ~16 s para ~4 s, com as citações preservadas. A tarefa não exige raciocínio longo: as evidências já chegam selecionadas, rotuladas e com o resumo da amostra, e o prompt define a estrutura da resposta. O `flash-lite` não tem esse recurso, por isso usa `LLM_THINKING_BUDGET=-1` (não enviar a configuração).

**Robustez:**

- até 3 tentativas automáticas em erros temporários (429 e 5xx), com limite de 60 s por tentativa, para a interface não ficar presa;
- se a chamada falhar mesmo assim (chave inválida, modelo indisponível, timeout), o pipeline **não quebra**: cai no modo somente recuperação, mostra as evidências e explica a falha;
- modelos que não aceitam a configuração de raciocínio são detectados, e a chamada é repetida sem ela.

Sem chave configurada, o sistema também opera em **modo somente recuperação**, o que permite avaliar e demonstrar toda a parte de recuperação sem custo.

## 12. Avaliação sem rotulagem manual

- **Decisão (responder × recusar):** perguntas no escopo, fora do escopo e sem evidência com o resultado esperado conhecido.
- **Precisão das evidências (proxy):** um regex de tema por pergunta mede a fração das evidências que tratam do assunto. É uma aproximação: não substitui a avaliação humana, mas detecta regressões.
- **Filtros:** se a extração acerta o filtro esperado e se as evidências o respeitam.
- **Com LLM:** a taxa de respostas com citações válidas, a média de evidências citadas, a decisão final (incluindo as recusas feitas pelo próprio LLM) e o número de chamadas que falharam. Este último evita que uma falha da API (ex.: cota esgotada) apareça como "resposta correta" no modo somente recuperação.
- **Resultado com o `gemini-3.5-flash-lite`:** 100% de respostas com citações válidas nos dois conjuntos. No hold-out, o LLM recusou corretamente "realidade virtual no site", o único caso que escapava dos guardrails pré-geração, e recusou duas perguntas válidas cuja recuperação não trouxe avaliações sobre o assunto. Nesses dois casos, a recusa foi fiel às evidências recebidas. Um próximo passo é medir a *faithfulness* de cada afirmação com um LLM juiz.
