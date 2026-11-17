# Exemplos de perguntas e respostas

Saídas **reais** do pipeline sobre a base completa (41.951 avaliações), geradas por `python scripts/generate_examples.py`.

> As respostas foram geradas pelo Google Gemini (`gemini-3.5-flash-lite`). Para cada pergunta aparecem as etapas do pipeline (filtros extraídos, núcleo da pergunta, decisão do guardrail), a resposta gerada e as primeiras evidências do contexto. As marcadas com ✅ foram citadas na resposta. Como o LLM não é determinístico, uma nova execução pode produzir um texto diferente, sobre as mesmas evidências.

<!-- gerado em 2026-11-16 -->

## Perguntas respondidas

### Investigação de reclamações

**Pergunta:** Quais são os principais problemas relatados pelos clientes em suas avaliações?

- **Filtros aplicados:** nota 1-3
- **Núcleo usado na busca semântica:** `problemas`
- **Guardrail:** 102 avaliações relevantes (melhor similaridade 0.890)
- **Recuperação:** 14.418 avaliações no recorte → 111 candidatas relevantes → 15 evidências distintas (5 textos idênticos colapsados, diversidade: mmr)

**Resposta gerada:**

> Com base na amostra recuperada de 15 avaliações, os principais problemas relatados pelos clientes envolvem defeitos ou mau funcionamento dos produtos, falhas e atrasos na entrega, erros no envio de pedidos, problemas com o atendimento pós-venda/contato, e problemas gerais com os Correios.
> 
> Os temas identificados, organizados do mais para o menos frequente na amostra, são:
> 
> * **Defeitos, falhas de funcionamento ou problemas de ativação no produto:** Presente em 5 das 15 avaliações analisadas, englobando produtos que vieram com defeito [E1][E6], que não funcionaram após ligar na tomada [E8], que são barulhentos [E10], ou que apresentam dificuldade para ativar ou fazer funcionar no celular [E15].
> * **Problemas com a entrega (atrasos, não recebimento e problemas com os Correios):** Citado em 5 das 15 avaliações, incluindo relatos de problemas com os Correios [E3][E4], atrasos ou demora na chegada [E5][E9][E10], e problema em uma das compras realizadas [E11].
> * **Dificuldade de contato e suporte pós-venda:** Relatado em 3 das 15 avaliações, abrangendo dificuldades para trocar produtos [E2], falta de conseguirem contato por e-mail ou telefone para reclamações [E11], e grande dificuldade em contactar o fornecedor para resolver problemas [E14].
> * **Erros no pedido (recebimento de item incorreto ou incompleto):** Presente em 3 das 15 avaliações, com menções a pedidos que vieram diferentes do solicitado [E2], recebimento de cor errada (pedida vermelha e entregue preta) [E13], e itens faltantes na entrega (compra de 4 cofres e recebimento de apenas 2) [E12].
> * **Insatisfação genérica:** 1 avaliação menciona de forma ampla a necessidade de resolver um problema sem especificar o detalhe do ocorrido [E7].
> 
> Com base estritamente nas evidências fornecidas, não foi possível concluir detalhes adicionais sobre o volume total da base de clientes além desta amostra específica.

| id | nota | entrega | categoria | avaliação |
|---|---|---|---|---|
| E1 ✅ | 2★ | entregue no prazo | audio | O produto está com problemas (defeito). |
| E2 ✅ | 1★ | entregue no prazo | automotivo | . Estou chateado com o site. Motivo em que o meu pedido veio diferente do que pedi. E estou com dificuldade para trocar. |
| E3 ✅ | 1★ | entregue no prazo | telefonia | NAO CHEGOU POR PROBLEMAS NOS CORREIOS |
| E4 ✅ | 1★ | entregue com atraso | esporte_lazer | Meu problema foi com os correios. |
| E5 ✅ | 3★ | entregue com atraso | relogios_presentes | RECOMENDO. UNICO PROBLEMA E A DEMORA |
| E6 ✅ | 2★ | entregue no prazo | brinquedos | Veio com defeitos |
| E7 ✅ | 1★ | entregue com atraso | informatica_acessorios | Preciso resolver meu problema |
| E8 ✅ | 3★ | entregue no prazo | ferramentas_jardim | problema. porém não funcionou, quando ligou na tomada esquentou e não funcionou mais. |
| … | | | | *mais 7 evidências no contexto* |

### Tema específico

**Pergunta:** O que os clientes relatam sobre problemas relacionados à entrega?

- **Filtros aplicados:** nota 1-3
- **Núcleo usado na busca semântica:** `problemas entrega`
- **Guardrail:** 105 avaliações relevantes (melhor similaridade 0.919)
- **Recuperação:** 14.418 avaliações no recorte → 105 candidatas relevantes → 15 evidências distintas (27 textos idênticos colapsados, diversidade: mmr)

**Resposta gerada:**

> Com base na amostra recuperada de 15 avaliações, os clientes relatam problemas relacionados à entrega divididos nos seguintes padrões, do mais para o menos frequente:
> 
> 1. **Atrasos e demora na entrega:** Os clientes mencionam atrasos generalizados, demora ou problemas com o prazo [E1][E4][E7][E9][E13][E15].
> 2. **Erros no conteúdo ou quantidade dos itens entregues:** Relatos de que o produto veio errado, incompleto, faltando itens, ou que foi entregue apenas uma parte do pedido [E2][E5][E6][E11][E12].
> 3. **Produto não entregue:** Casos em que o produto não chegou ao destino, houve imprevistos informados pelos Correios, ou o cliente aguarda o produto que ainda não chegou [E3][E4][E8].
> 4. **Insatisfação com a transportadora:** Reclamação direcionada especificamente à falta de qualidade dos Correios [E10].
> 5. **Produto avariado/defeituoso na entrega:** Relato de item (cuba) que chegou quebrado [E14].
> 
> O que não foi possível concluir a partir da amostra: os motivos exatos para os atrasos sistêmicos e quais foram os impactos operacionais decorrentes dos cancelamentos ou solicitações de estorno além do que foi explicitamente relatado pelos clientes nas evidências citadas.

| id | nota | entrega | categoria | avaliação |
|---|---|---|---|---|
| E1 ✅ | 1★ | entregue com atraso | utilidades_domesticas | Atraso na entrega *(+7 idênticas)* |
| E2 ✅ | 3★ | entregue no prazo | moveis_decoracao | Foi entregue apenas 01 tampa e a mesma encontra-se com defeito. Foi aberto uma solicitação para troca e aguardo a entrega da outra. |
| E3 ✅ | 1★ | não entregue | moveis_quarto | Ja é as egunda vez que tenho problemas com entregas |
| E4 ✅ | 1★ | entregue com atraso | esporte_lazer | Problema com a entrega. O produto não foi entregue. Em 19/04 os Correios informaram que houve um imprevisto na entrega. Solicitei o estorno da compra e estou aguardando as devidas providências. |
| E5 ✅ | 1★ | entregue no prazo | beleza_saude | erro na entrega. a entrega nao chegou correta completa |
| E6 ✅ | 1★ | entregue no prazo | informatica_acessorios | Produto entregue errado |
| E7 ✅ | 1★ | entregue com atraso | informatica_acessorios | TERRIVEL ESTE PROBLEMA DA ENTREGA |
| E8 ✅ | 2★ | cancelado | sem_categoria | Problemas com entrega. Não chegou ainda estou na esperando ainda |
| … | | | | *mais 7 evidências no contexto* |

### Experiências positivas

**Pergunta:** Quais aspectos da experiência de compra são mais elogiados pelos clientes?

- **Filtros aplicados:** nota 4-5
- **Núcleo usado na busca semântica:** `elogiados`
- **Guardrail:** 84 avaliações relevantes (melhor similaridade 0.898)
- **Recuperação:** 27.533 avaliações no recorte → 120 candidatas relevantes → 15 evidências distintas (46 textos idênticos colapsados, diversidade: mmr)

**Resposta gerada:**

> Com base na amostra analisada, os principais aspectos elogiados pelos clientes concentram-se na satisfação geral com a experiência de compra, na qualidade dos produtos, na rapidez do atendimento e na eficiência da entrega.
> 
> Os padrões observados, organizados do mais para o menos frequente na amostra, são:
> 
> 1. **Satisfação geral com a experiência e superação de expectativas:** Clientes destacam que a experiência de compra foi ótima, satisfatória ou que superou expectativas em vários aspectos [E4][E5][E7][E8][E9][E10][E11][E13][E14][E15].
> 2. **Qualidade dos produtos e embalagem:** Há elogios específicos à qualidade dos itens adquiridos [E8][E10] e ao cuidado com as embalagens ("muito bem embalados") [E1]. *(Nota: uma avaliação com menção positiva geral também aponta pontos de melhoria pontuais na vedação do produto para evitar vazamentos [E9]).*
> 3. **Atendimento rápido:** A agilidade e a excelência no atendimento são explicitamente elogiadas por clientes [E10][E11].
> 4. **Entrega no prazo:** Todas as 15 avaliações da amostra registraram que o produto foi entregue no prazo [E1 a E15], sendo destacado por cliente como um ponto positivo junto à qualidade [E8].

| id | nota | entrega | categoria | avaliação |
|---|---|---|---|---|
| E1 ✅ | 5★ | entregue no prazo | livros_interesse_geral | Muito bem embalados |
| E2 | 5★ | entregue no prazo | moveis_cozinha_area_de_servico_jantar_e_jardim | Estao de parabéns *(+1 idênticas)* |
| E3 | 5★ | entregue no prazo | portateis_casa_forno_e_cafe | 5 estrela. Elogiável |
| E4 ✅ | 5★ | entregue no prazo | brinquedos | Ótimo. Superou as expectativas em todos os aspectos. Recomendo. |
| E5 ✅ | 5★ | entregue no prazo | automotivo | Ótimo em TDS os aspectos |
| E6 | 5★ | entregue no prazo | climatizacao | satisfeito com a compra |
| E7 ✅ | 5★ | entregue no prazo | cama_mesa_banho | Amei a experiência. |
| E8 ✅ | 5★ | entregue no prazo | informatica_acessorios | Minha experiência com essa loja tem sido satisfatória. O produto é de qualidade, e entrega no prazo. Recomendo d+ os dois! |
| … | | | | *mais 7 evidências no contexto* |

### Investigação de insatisfação

**Pergunta:** Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?

- **Filtros aplicados:** nota 1-2
- **Núcleo usado na busca semântica:** `insatisfeitos`
- **Guardrail:** 65 avaliações relevantes (melhor similaridade 0.965)
- **Recuperação:** 10.836 avaliações no recorte → 75 candidatas relevantes → 15 evidências distintas (18 textos idênticos colapsados, diversidade: mmr)

**Resposta gerada:**

> Na amostra analisada de 15 avaliações de clientes insatisfeitos (notas 1 e 2), os principais padrões identificados envolvem problemas de qualidade e funcionamento dos produtos, falhas e atrasos na entrega, além de dificuldades no atendimento.
> 
> Os padrões organizados do mais para o menos frequente são:
> 
> * **Problemas de qualidade, defeitos e acabamento nos produtos:** É o tema mais recorrente, presente em 6 das 15 avaliações. Os clientes relatam itens que vieram com defeitos [E2][E6][E9][E11], produto mal acabado [E5], produto com defeito/não funciona [E8][E10], além de um caso de produto fora dos padrões nacionais que não comporta documentos e cartões [E13].
> * **Problemas com entregas (atrasos, não recebimento e cancelamentos):** Aparece em 6 das 15 avaliações. Os clientes mencionam insatisfação por pedidos não entregues ou não recebidos [E1][E4][E7][E15], atrasos na entrega [E3][E15], frustração com a necessidade de buscar o pedido nos Correios sem aviso prévio sobre o local de retirada [E14], e pedidos cancelados [E7][E12].
> * **Insatisfação com parcerias e marcas:** Identificado em 1 avaliação em que o cliente relata frustração por ter comprado pensando ser com Lojas Americanas e critica as parcerias da empresa [E12].
> * **Dificuldades no atendimento:** Citado em 1 avaliação, onde o cliente reclama da impossibilidade de falar com a central de atendimento por telefone e pede uma solução urgente [E15].

| id | nota | entrega | categoria | avaliação |
|---|---|---|---|---|
| E1 ✅ | 1★ | não entregue | climatizacao | Por enquanto insatisfeita. |
| E2 ✅ | 2★ | entregue no prazo | brinquedos | Veio com defeitos |
| E3 ✅ | 1★ | entregue com atraso | beleza_saude | Atraso na entrega *(+2 idênticas)* |
| E4 ✅ | 1★ | não entregue | bebes | insatisfeita. Preciso saber a data que será entregue |
| E5 ✅ | 2★ | entregue no prazo | eletroportateis | Produto mal acabado |
| E6 ✅ | 2★ | entregue no prazo | beleza_saude | produto com defeito *(+5 idênticas)* |
| E7 ✅ | 1★ | cancelado | utilidades_domesticas | PRODUTO NÃO RECEBIDO |
| E8 ✅ | 1★ | entregue no prazo | utilidades_domesticas | Péssima meu produtdefeito |
| … | | | | *mais 7 evidências no contexto* |

### Exploração baseada em evidências

**Pergunta:** Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?

- **Filtros aplicados:** nota 1-3
- **Núcleo usado na busca semântica:** `problemas prazo entrega`
- **Guardrail:** 110 avaliações relevantes (melhor similaridade 0.922)
- **Recuperação:** 14.418 avaliações no recorte → 110 candidatas relevantes → 15 evidências distintas (13 textos idênticos colapsados, diversidade: mmr)

**Resposta gerada:**

> Com base na amostra recuperada de 15 avaliações, existem múltiplos relatos que sustentam problemas relacionados ao prazo de entrega, seja por atrasos efetivos ou por prazos considerados longos e demorados pelos clientes.
> 
> Os padrões identificados na amostra são os seguintes:
> 
> * **Atrasos e descumprimento do prazo estipulado:** Clientes relatam que o prazo não foi respeitado, que o produto chegou após o prazo ou que o material não foi entregue no prazo [E2][E3][E5][E6][E7][E8][E9][E11][E12].
> * **Prazos considerados longos ou demorados:** Clientes apontam a lentidão do serviço com reclamações sobre prazos demorados ou muito longos, mesmo quando recebidos dentro do prazo estipulado [E1][E10][E14].
> 
> O que não foi possível concluir: a amostra também contém avaliações que mencionam entrega no prazo ou qualidade do produto [E4][E13][E15], mas as evidências acima são as que sustentam especificamente as reclamações sobre os prazos.

| id | nota | entrega | categoria | avaliação |
|---|---|---|---|---|
| E1 ✅ | 3★ | entregue no prazo | artes | Um prazo de entrega demorado |
| E2 ✅ | 2★ | entregue no prazo | informatica_acessorios | O problema foi que chegou após o prazo |
| E3 ✅ | 2★ | entregue com atraso | esporte_lazer | Atraso na entrega *(+7 idênticas)* |
| E4 ✅ | 1★ | entregue no prazo | telefonia | PRODUTO ENTREGUE NO PRAZO COM QUALIDADE |
| E5 ✅ | 1★ | entregue com atraso | esporte_lazer | O prazo de entrega enorme de 16 dias úteis não foi respeitado. No site dá como dificuldade de entrega mas não fala qual. Preciso q me entregue o mais rápido possível ou me devolvam o dinheiro!!! |
| E6 ✅ | 1★ | entregue com atraso | relogios_presentes | Ñ compriu seu prazo de entrega |
| E7 ✅ | 1★ | entregue com atraso | brinquedos | Não atende o prazo de entrega |
| E8 ✅ | 1★ | entregue com atraso | esporte_lazer | Atraso na entrega. Ainda não recebi o produto é já se passou o prazo solicitado. |
| … | | | | *mais 7 evidências no contexto* |

### Metadados: categoria + sentimento

**Pergunta:** O que os clientes mais elogiam nos produtos de cama, mesa e banho?

- **Filtros aplicados:** nota 4-5; categorias: cama_mesa_banho
- **Núcleo usado na busca semântica:** `elogiam cama mesa banho`
- **Guardrail:** 82 avaliações relevantes (melhor similaridade 0.903)
- **Recuperação:** 2.738 avaliações no recorte → 101 candidatas relevantes → 15 evidências distintas (3 textos idênticos colapsados, diversidade: mmr)

**Resposta gerada:**

> Com base na amostra de 15 avaliações recuperadas, os clientes elogiam principalmente a estética e a beleza dos produtos, além da qualidade geral, praticidade e o atendimento/serviço de entrega. 
> 
> Abaixo estão os temas organizados do mais para o menos frequente na amostra:
> 
> 1. **Beleza e estética dos produtos (design e decoração):** Em 7 das 15 avaliações, os clientes destacam que os itens são bonitos, elegantes ou deixam o ambiente agradável [E2][E3][E5][E8][E9][E10][E13]. Especificamente, destacam-se elogios à beleza de colchas, saias para cama box, tapetes e cortinas.
> 2. **Qualidade geral e recomendação:** Em 6 das 15 avaliações, os clientes afirmam que o produto é "bom", "excelente" ou recomendam a compra [E2][E4][E7][E9][E13][E15].
> 3. **Praticidade e características específicas:** Em 2 das 15 avaliações, há menções específicas à facilidade de uso, como a saia para cama box ser fácil de colocar, além de elogios ao tecido [E13][E14].
> 4. **Agilidade na entrega e atendimento:** Em 2 das 15 avaliações, os clientes elogiam a rapidez da entrega e o bom atendimento da loja [E9][E11][E12].
> 
> O que não foi possível concluir: Não há detalhes aprofundados sobre a durabilidade a longo prazo dos materiais ou sobre aspectos negativos específicos dos produtos, visto que a amostra é composta exclusivamente por notas 4 e 5.

| id | nota | entrega | categoria | avaliação |
|---|---|---|---|---|
| E1 | 4★ | entregue no prazo | cama_mesa_banho | Saia para cama box king |
| E2 ✅ | 4★ | entregue no prazo | cama_mesa_banho | RECOMENDO. A colcha é muito bonita e destaca bem na cama. |
| E3 ✅ | 5★ | entregue no prazo | cama_mesa_banho | BOM E BONITO |
| E4 ✅ | 5★ | entregue no prazo | cama_mesa_banho | Jogo de cama. Excelente produto. Recomendo a todos |
| E5 ✅ | 5★ | entregue no prazo | cama_mesa_banho | Muito lindo a saia ficou lindissima em minha cama |
| E6 | 5★ | entregue no prazo | cama_mesa_banho | Amei o meu coberdrom |
| E7 ✅ | 4★ | entregue no prazo | cama_mesa_banho | Boa loja terceirizada |
| E8 ✅ | 4★ | entregue no prazo | cama_mesa_banho | Produto bonito. Achei um pouco pequeno para a cama. |
| … | | | | *mais 7 evidências no contexto* |

### Metadados: entrega atrasada (derivada das datas)

**Pergunta:** Quais temas aparecem nas avaliações de pedidos com entrega atrasada?

- **Filtros aplicados:** entrega: atrasado
- **Núcleo usado na busca semântica:** `entrega atrasada`
- **Guardrail:** 97 avaliações relevantes (melhor similaridade 0.950)
- **Recuperação:** 3.809 avaliações no recorte → 97 candidatas relevantes → 15 evidências distintas (23 textos idênticos colapsados, diversidade: mmr)

**Resposta gerada:**

> Nas 15 avaliações analisadas sobre pedidos com entrega atrasada, o tema central é o atraso ou demora no cumprimento do prazo, acompanhado em casos específicos por falta de retorno da empresa, suspeita de greve dos correios e produtos ainda não recebidos. 
> 
> A organização dos temas identificados na amostra, do mais para o menos frequente, é a seguinte:
> 
> 1. **Atraso direto na entrega / descumprimento de prazo:** É o tema predominante na amostra, mencionado de forma direta pelos clientes através de relatos de atraso, demora ou não atendimento ao prazo estabelecido [E1][E2][E3][E4][E5][E6][E7][E8][E10][E12][E13][E15].
> 2. **Espera prolongada e ausência de comunicação:** Em menções específicas, clientes relatam estar aguardando a entrega e apontam falta de retorno por parte da empresa [E8][E9][E11].
> 3. **Suspeita de causa externa:** Em uma das avaliações, o atraso é associado à dúvida sobre uma possível greve dos correios [E14].
> 
> Não foi possível concluir detalhes sobre os processos logísticos internos, causas raiz operacionais além da menção isolada a correios, ou dados quantitativos sobre a base inteira de clientes, visto que as informações limitam-se aos textos e metadados das 15 avaliações recuperadas.

| id | nota | entrega | categoria | avaliação |
|---|---|---|---|---|
| E1 ✅ | 1★ | entregue com atraso | cama_mesa_banho | Produto com entrega atrasada |
| E2 ✅ | 3★ | entregue com atraso | esporte_lazer | Tive atraso na entrega |
| E3 ✅ | 1★ | entregue com atraso | utilidades_domesticas | A entrega esta atrasada. |
| E4 ✅ | 1★ | entregue com atraso | moveis_escritorio | entrega muito atrasada. |
| E5 ✅ | 2★ | entregue com atraso | sem_categoria | Entrega em atraso |
| E6 ✅ | 1★ | entregue com atraso | automotivo | Entrega super atrasada |
| E7 ✅ | 1★ | entregue com atraso | esporte_lazer | Atrasou na entrega |
| E8 ✅ | 1★ | entregue com atraso | moveis_escritorio | Aguardando a entrega. Entrega em atraso. |
| … | | | | *mais 7 evidências no contexto* |

### Metadados: categoria + período

**Pergunta:** Quais reclamações aparecem sobre celulares em 2018?

- **Filtros aplicados:** nota 1-3; categorias: telefonia; período: 2018-01 a 2018-12
- **Núcleo usado na busca semântica:** `reclamações celulares`
- **Guardrail:** 63 avaliações relevantes (melhor similaridade 0.889)
- **Recuperação:** 369 avaliações no recorte → 63 candidatas relevantes → 15 evidências distintas (1 textos idênticos colapsados, diversidade: mmr)

**Resposta gerada:**

> Na amostra analisada de 15 avaliações de celulares em 2018, as reclamações concentram-se em problemas de divergência/incorreção de produtos, falhas na entrega e logística, defeitos e falsificações, problemas com acessórios e falhas no atendimento/pós-venda.
> 
> Os principais padrões identificados na amostra são:
> 
> 1. **Problemas com o produto enviado (divergências, incorreções, defeitos e falsificação):** 
>    - Recebimento de produto incorreto ou diferente do anunciado ocorre em 4 das 15 avaliações, incluindo o envio de película de silicone em vez de vidro [E15], capa maior que o celular [E12], produto incorreto de forma geral [E1] e falta de itens do pedido (pagou por 2 kits e recebeu apenas 1) [E2].
>    - Há relato de produto com defeito [E14] e de recebimento de uma falsificação do Moto G5 Plus (com lacre rompido, sem manual ou garantia, selos da Anatel falsos, IMEI clonado e design incorreto) [E4].
>    - Problemas de compatibilidade ou tamanho inadequado são citados em 2 avaliações, como película menor que a tela [E3] e um telefone no qual nenhum chip cabe, nem com adaptador [E5].
>    - Película que veio quebrada e com avarias [E8].
>    - Película mal colocada que ficou com bolha grande [E10].
> 
> 2. **Problemas de entrega, atrasos e cancelamentos:**
>    - Problemas logísticos e com os Correios aparecem em 3 avaliações, citando que o produto não chegou por problemas nos Correios [E6], não recebimento de película devido a problemas no pacote no meio do caminho [E9], e atraso na entrega com menção a falta de respeito [E7].
>    - Falhas graves no envio por parte do fornecedor/loja são relatadas em 1 avaliação, que aponta demora no envio, não envio do celular original, proposta de troca aceita mas não cumprida [E11].
>    - Falta de estorno após o cancelamento do pedido é citada em 1 avaliação [E13].
> 
> 3. **Falhas no atendimento e pós-venda:**
>    - Ausência de retorno ou solução por parte da empresa após reclamações e solicitações de reembolso/suporte é mencionada em 3 avaliações [E8][E9][E13].

| id | nota | entrega | categoria | avaliação |
|---|---|---|---|---|
| E1 ✅ | 1★ | entregue no prazo | telefonia | reclamação. produto incorreto |
| E2 ✅ | 1★ | entregue no prazo | telefonia | Fiz um pedido com 2kits para dois celulares. Paguei pelos dois e só recebi um. |
| E3 ✅ | 1★ | entregue no prazo | telefonia | Película menor do que a tela do celular |
| E4 ✅ | 1★ | entregue no prazo | telefonia | Veio uma falsificaçao do Moto G5 plus, com o lacre da caixa rompido, sem manual, sem termo de garantia da Motorola, selos da anatel falsos, IMEI clonado, design do celular todo errado. Não recomendo! |
| E5 ✅ | 3★ | entregue no prazo | telefonia | Telefone celular. Nenhum chip cabe, nem com adaptador!Estou buscando solução mais prática.Caso não consiga, terei que trocar o produto.Recebi dia 17/8. |
| E6 ✅ | 1★ | entregue no prazo | telefonia | NAO CHEGOU POR PROBLEMAS NOS CORREIOS |
| E7 ✅ | 2★ | entregue com atraso | telefonia | Ruim. Falta de respeito com cliente |
| E8 ✅ | 1★ | entregue no prazo | telefonia | PRODUTO CHEGOU C AVARIAS. Até o momento ninguém retornou minha solicitação de reembolso pois a película veio quebrada !!! |
| … | | | | *mais 7 evidências no contexto* |

## Perguntas recusadas (fora do escopo ou sem evidências)

| Pergunta | Status | Motivo (trace) |
|---|---|---|
| Qual é a capital da Austrália? | `out_of_scope` | pergunta mais próxima de exemplos fora do domínio (margem -0.050) e sem suporte lexical na base |
| Me passe uma receita de bolo de chocolate. | `out_of_scope` | pergunta mais próxima de exemplos fora do domínio (margem -0.016) e sem suporte lexical na base |
| O que os clientes acham de pagar com criptomoedas na loja? | `insufficient_evidence` | termo(s) praticamente ausente(s) das avaliações: criptomoedas |
| O que os clientes acham da entrega por drone? | `insufficient_evidence` | apenas 0 avaliação(ões) mencionam os termos ['entrega', 'drone'] no recorte (mínimo: 3) |
| O que os clientes reclamam sobre seguros e serviços em 2016? | `insufficient_evidence` | nenhuma avaliação atende aos filtros aplicados |

Nenhuma dessas perguntas chega ao LLM: a recusa é decidida antes da geração, sem custo de tokens.
