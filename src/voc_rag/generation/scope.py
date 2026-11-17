"""Classificador de escopo por exemplos (nearest prototypes).

Por que não um limiar de similaridade com as avaliações?
O E5 gera similaridades cosseno altas para QUALQUER texto (anisotropia): na
calibração, "Me passe uma receita de bolo" teve similaridade 0,90 com a melhor
avaliação, MAIOR que várias perguntas legítimas (~0,86). Um limiar absoluto
não separa as classes.

Solução: comparar a pergunta com dois conjuntos de perguntas-exemplo: dentro do
escopo (experiência de compra) e fora dele (conhecimentos gerais, código, finanças,
receitas...). Comparar pergunta com pergunta é uma tarefa simétrica, que o E5
separa bem. A margem = média dos 3 exemplos mais próximos (escopo) − idem (fora).

Os exemplos NÃO incluem as perguntas do conjunto de avaliação (evaluation/questions.yaml),
para não "vazar" o teste.
"""

from __future__ import annotations

import numpy as np

from voc_rag.indexing.embeddings import Embedder

IN_SCOPE_EXAMPLES = [
    "O que os compradores dizem sobre o frete?",
    "Os pedidos chegam no prazo?",
    "Há queixas sobre produtos com defeito?",
    "O que acham da embalagem dos produtos?",
    "Os clientes recomendam a loja?",
    "Como é a experiência de troca e devolução?",
    "Reclamações sobre cobrança ou pagamento",
    "Os produtos correspondem às fotos do site?",
    "O que dizem sobre o tamanho das roupas?",
    "Quais são as queixas sobre pedidos incompletos?",
    "O que os clientes acham do custo-benefício?",
    "Opiniões sobre a nota fiscal",
    "Os clientes receberam o rastreamento do pedido?",
    "Comentários sobre a qualidade do tecido",
    "O que falam dos vendedores parceiros?",
    "Clientes satisfeitos com a compra",
    "Avaliações sobre móveis que chegaram danificados",
    "Comentários negativos sobre eletrônicos",
    "O que mais incomoda os compradores?",
    "Pontos positivos mencionados nas avaliações",
    "Problemas com produtos falsificados",
    "Quanto tempo demora a entrega segundo os clientes?",
    "Elogios ao atendimento",
    "Avaliações de perfumes",
    "Reclamações de produtos que não funcionam",
]

OUT_OF_SCOPE_EXAMPLES = [
    "Qual é a previsão do tempo para amanhã?",
    "Quem descobriu o Brasil?",
    "Me conte uma piada",
    "Traduza esta frase para o inglês",
    "Qual a fórmula de Bhaskara?",
    "Como fazer um bolo de cenoura?",
    "Qual o preço da ação da Petrobras hoje?",
    "Escreva um poema sobre o mar",
    "Quem é o presidente dos Estados Unidos?",
    "Como instalar o Windows?",
    "Qual a distância da Terra à Lua?",
    "Explique a teoria da relatividade",
    "Crie um código em JavaScript",
    "Quais os sintomas da gripe?",
    "Qual o melhor time de futebol?",
    "Como funciona a blockchain?",
    "Resuma o livro Dom Casmurro",
    "Qual a população da China?",
    "Me ajude com meu currículo",
    "Qual a capital do Canadá?",
    "Como aprender inglês rápido?",
    "O que é machine learning?",
    "Qual a receita de lasanha?",
    "Como investir em criptomoedas?",
    "Quais filmes estão em cartaz?",
]


class ScopeClassifier:
    def __init__(self, embedder: Embedder, top_n: int = 3):
        self.embedder = embedder
        self.top_n = top_n
        self._in = np.stack([embedder.embed_query(q) for q in IN_SCOPE_EXAMPLES])
        self._out = np.stack([embedder.embed_query(q) for q in OUT_OF_SCOPE_EXAMPLES])

    def margin(self, question: str) -> float:
        """> 0: mais parecida com perguntas do domínio; < 0: com perguntas fora do domínio."""
        v = self.embedder.embed_query(question)
        s_in = np.sort(self._in @ v)[-self.top_n :].mean()
        s_out = np.sort(self._out @ v)[-self.top_n :].mean()
        return float(s_in - s_out)
