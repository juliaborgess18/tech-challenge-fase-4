"""Transformação da pergunta do usuário.

Etapas (todas determinísticas e explicáveis, sem chamada a LLM):
1. Normalização (espaços, caixa/acentos para matching de regras).
2. Extração de filtros de metadados a partir da linguagem natural:
   - nota explícita ("nota 1", "notas 4 e 5", "nota baixa/alta")
   - sentimento implícito ("insatisfeitos", "reclamações" -> notas baixas;
     "elogios", "satisfeitos" -> notas altas)
   - categoria de produto (nomes do Olist + sinônimos comuns)
   - período (anos 2016-2018)
   - entrega atrasada ("atraso", "atrasou")
3. Reescrita para o "núcleo" da pergunta: remove o enquadramento analítico
   ("Quais padrões podem ser identificados nas avaliações de clientes...") que não
   se parece com nenhuma avaliação e dilui a similaridade. Ex.:
   "O que os clientes relatam sobre problemas relacionados à entrega?" -> "problemas entrega".
   O núcleo é o texto embedado (com fallback para a pergunta original).
4. Termos-chave: o núcleo sem palavras de intenção ("problemas", "elogios") e sem
   as palavras já convertidas em filtro (categoria). São usados pelos guardrails
   para checar se o assunto existe na base ("criptomoedas" não aparece em nenhuma avaliação).
5. Expansão leve de consulta para o BM25 com sinônimos do tema detectado
   (ex.: "entrega" -> "prazo chegou recebi atraso correios").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

from voc_rag.retrieval.filters import MetadataFilter


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text.lower())).strip()


# --------------------------------------------------------------------- temas
TOPIC_PATTERNS: dict[str, tuple[str, str]] = {
    # tema: (regex sobre texto normalizado, termos de expansão para o BM25)
    "entrega": (
        r"entreg|prazo|atras|chegou|chegar|nao recebi|recebi|correio|transportad|frete|demor",
        "entrega prazo chegou recebi atraso atrasou correios demorou",
    ),
    "qualidade": (
        r"qualidade|defeit|quebrad|estragad|danific|falsific|original|material",
        "qualidade defeito quebrado danificado material original",
    ),
    "atendimento": (
        r"atendimento|suporte|contato|resposta|vendedor|loja|sac",
        "atendimento contato resposta vendedor loja",
    ),
    "produto_errado": (
        r"errad|diferente|trocad|nao corresponde|faltando|incomplet",
        "errado diferente trocado faltando incompleto",
    ),
    "preco": (r"preco|car[oa]s?|barat|custo|valor", "preço barato caro custo benefício"),
}

NEGATIVE_PATTERN = r"insatisf|reclama|problema|negativ|ruim|pessim|critic|frustr|decepcion|queixa|nao recomend|nao gost"
POSITIVE_PATTERN = r"elogi|positiv|satisf|valoriz|gost|recomend|bem avaliad|otim|excelent"
LATE_PATTERN = r"atras|fora do prazo|demor"

# ---------------------------------------------------------------- categorias
CATEGORY_ALIASES: dict[str, list[str]] = {
    "cama_mesa_banho": [r"cama mesa", r"cama, mesa", r"lencol", r"toalha", r"enxoval"],
    "beleza_saude": [r"beleza", r"saude", r"cosmetic", r"maquiagem"],
    "perfumaria": [r"perfum"],
    "esporte_lazer": [r"esporte", r"lazer", r"fitness"],
    "informatica_acessorios": [r"informatica", r"computador", r"notebook", r"mouse", r"teclado"],
    "moveis_decoracao": [r"movei", r"decoracao", r"\bmovel\b"],
    "relogios_presentes": [r"relogio"],
    "utilidades_domesticas": [r"utilidades domesticas", r"cozinha", r"panela"],
    "telefonia": [r"telefon", r"celular", r"smartphone", r"capinha", r"fone"],
    "automotivo": [r"automotiv", r"\bcarro\b", r"veicul"],
    "brinquedos": [r"brinquedo"],
    "bebes": [r"\bbebe"],
    "eletronicos": [r"eletronic"],
    "ferramentas_jardim": [r"ferramenta", r"jardim"],
    "eletrodomesticos": [r"eletrodomestic"],
    "livros_interesse_geral": [r"\blivro"],
    "pet_shop": [r"\bpet\b", r"cachorro", r"\bgato"],
    "fashion_bolsas_e_acessorios": [r"\bbolsa", r"\bmoda\b"],
    "cool_stuff": [r"cool stuff"],
}


# Enquadramento analítico/meta da pergunta (sem acento, minúsculas), removido do núcleo
META_WORDS = set(
    """a o as os ao aos e de do da dos das em no na nos nas um uma uns umas que se me te
    qual quais quem como onde quando porque por pelo pela pelos pelas para pra com sem sobre entre
    sao ser ter tem foi estao esta ha existe existem pode podem possivel
    principais principal mais menos maior menor muito muitos muitas frequente frequentes frequencia
    cliente clientes consumidor consumidores comprador compradores usuario usuarios pessoas
    avaliacao avaliacoes avaliam avaliaram comentario comentarios review reviews opiniao opinioes
    relatam relatados relatadas relatado relatos relata dizem falam acham pensam mencionam mencionados
    aparecem aparece identificados identificadas identificar sustentam sustenta conclusao
    padrao padroes recorrente recorrentes tema temas tipo tipos aspecto aspectos
    relacionado relacionados relacionada relacionadas respeito experiencia
    nota notas baixa baixas alta altas pedido pedidos produto produtos loja compra compras
    passe diga mostre liste explique nao sim ja tambem
    motiva motivam motivo motivos razao razoes causa causas leva levam explica explicam
    seu sua seus suas meu minha meus minhas nosso nossa nossos nossas dele dela deles delas
    isso isto esse essa esses essas este esta estes estas aquele aquela eles elas voce voces""".split()
)
# Palavras de intenção/sentimento: ficam no núcleo (orientam a busca) mas não são termos-chave
INTENT_PREFIX = re.compile(
    r"^(problem|reclam|elogi|insatisf|satisf|gost|positiv|negativ|ruim|bom|boa|otim|queix|critic|"
    r"pessim|excelen|recomend|valoriz|incomod|frustr|decepc)"
)


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    embedding_text: str
    bm25_text: str
    filters: MetadataFilter
    core_text: str = ""
    key_terms: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sentiment_hint: str | None = None  # "negativa" | "positiva" | None
    notes: list[str] = field(default_factory=list)  # explicação das transformações (para o trace)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["filters"] = self.filters.to_dict()
        return data


def _extract_score_filter(q: str, notes: list[str]) -> tuple[int | None, int | None]:
    # "nota 1", "notas 1 e 2", "nota 4 ou 5", "nota 1 a 3"
    m = re.search(r"notas? (?:de )?([1-5])(?:\s*(?:e|ou|a|-|ate)\s*([1-5]))?", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        notes.append(f"nota explícita detectada: {min(a, b)}-{max(a, b)}")
        return min(a, b), max(a, b)
    if re.search(r"nota(s)? (baixa|ruim|pessima)|baixa(s)? nota|avaliac\w+ baixa", q):
        notes.append("'nota baixa' -> notas 1-2")
        return 1, 2
    if re.search(r"nota(s)? (alta|boa|maxima)|alta(s)? nota|avaliac\w+ alta", q):
        notes.append("'nota alta' -> notas 4-5")
        return 4, 5
    return None, None


def _extract_categories(q: str, known_categories: set[str] | None, notes: list[str]) -> list[str]:
    found: list[str] = []
    for cat, patterns in CATEGORY_ALIASES.items():
        if any(re.search(p, q) for p in patterns):
            found.append(cat)
    # nomes oficiais escritos por extenso ("cama mesa banho", "relogios presentes")
    for cat in known_categories or set():
        phrase = cat.replace("_", " ")
        if len(phrase) > 4 and phrase in q and cat not in found:
            found.append(cat)
    if known_categories:
        found = [c for c in found if c in known_categories]
    if found:
        notes.append("categoria(s) detectada(s): " + ", ".join(found))
    return found


def _extract_period(q: str, notes: list[str]) -> tuple[str | None, str | None]:
    years = sorted(set(re.findall(r"\b(201[6-8])\b", q)))
    if not years:
        return None, None
    notes.append("período detectado: " + "-".join(years))
    return f"{years[0]}-01", f"{years[-1]}-12"


def extract_core(question: str, categories: list[str] | None = None) -> tuple[str, list[str]]:
    """Retorna (núcleo da pergunta, termos-chave normalizados)."""
    category_tokens = {t for c in categories or [] for t in c.split("_")}
    core_words, key_terms = [], []
    for word in re.findall(r"[^\W\d_]+", question, flags=re.UNICODE):
        norm = strip_accents(word.lower())
        if norm in META_WORDS or len(norm) < 3:
            continue
        core_words.append(word.lower())
        if not INTENT_PREFIX.match(norm) and norm not in category_tokens:
            key_terms.append(norm)
    return " ".join(core_words), key_terms


def process_query(
    question: str,
    known_categories: set[str] | None = None,
    auto_filters: bool = True,
) -> ProcessedQuery:
    original = re.sub(r"\s+", " ", question).strip()
    q = normalize_for_matching(original)
    notes: list[str] = []

    topics = [name for name, (pattern, _) in TOPIC_PATTERNS.items() if re.search(pattern, q)]
    expansion = " ".join(TOPIC_PATTERNS[t][1] for t in topics)

    sentiment_hint = None
    if re.search(NEGATIVE_PATTERN, q):
        sentiment_hint = "negativa"
    elif re.search(POSITIVE_PATTERN, q):
        sentiment_hint = "positiva"

    flt = MetadataFilter()
    if auto_filters:
        flt.min_score, flt.max_score = _extract_score_filter(q, notes)
        if flt.min_score is None and sentiment_hint == "negativa":
            # reclamações e problemas aparecem quase só em notas 1-3
            flt.min_score, flt.max_score = 1, 3
            notes.append("intenção negativa (problemas/insatisfação) -> notas 1-3")
        elif flt.min_score is None and sentiment_hint == "positiva":
            flt.min_score, flt.max_score = 4, 5
            notes.append("intenção positiva (elogios/satisfação) -> notas 4-5")
        if re.search(r"insatisf", q) and flt.max_score == 3 and flt.min_score == 1:
            flt.max_score = 2
            notes.append("'insatisfeitos' -> restringe a notas 1-2")
        flt.categories = _extract_categories(q, known_categories, notes)
        flt.period_start, flt.period_end = _extract_period(q, notes)
        if re.search(r"entrega(s)? atrasad|pedidos? atrasad|entregues? com atraso|entrega(s)? fora do prazo", q):
            flt.delivery_status = ["atrasado"]
            notes.append("'entrega atrasada' -> pedidos entregues após a data estimada")

    if topics:
        notes.append("temas: " + ", ".join(topics))

    core_text, key_terms = extract_core(original, flt.categories)
    if core_text and core_text != original:
        notes.append(f"núcleo da pergunta usado na busca semântica: '{core_text}'")

    return ProcessedQuery(
        original=original,
        normalized=q,
        embedding_text=core_text or original,
        core_text=core_text,
        key_terms=key_terms,
        bm25_text=f"{original} {expansion}".strip(),
        filters=flt,
        topics=topics,
        sentiment_hint=sentiment_hint,
        notes=notes,
    )
