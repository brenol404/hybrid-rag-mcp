"""Compressor de contexto estilo-Caveman: remove palavras de função previsíveis.

Ideia (inspirada no wilpel/caveman-compression): a "gramática" carrega pouca
informação factual. Ao podar só palavras de função (artigos, conectivos,
auxiliares e enchimentos) dos chunks ANTES de montar o prompt, reduzimos o
input de tokens sem tocar em números, nomes próprios ou termos técnicos.

Níveis:
  0 = desligado (None);
  1 = leve  — remove conectivos e enchimentos;
  2 = agressivo — também remove determinantes e verbos auxiliares.
Negação (não/not/nunca/…) e números nunca são removidos.
"""

from __future__ import annotations

from collections.abc import Callable

# (categoria, palavras) — PT e EN. Nunca incluir negação.
_DETERMINERS = {
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "este",
    "esta",
    "estes",
    "estas",
    "esse",
    "essa",
    "esses",
    "essas",
    "aquele",
    "aquela",
    "aqueles",
    "aquelas",
    "cada",
    "the",
    "an",
    "this",
    "that",
    "these",
    "those",
    "each",
}
_CONNECTIVES = {
    "e",
    "ou",
    "mas",
    "porém",
    "contudo",
    "entretanto",
    "todavia",
    "portanto",
    "porque",
    "pois",
    "que",
    "como",
    "quando",
    "enquanto",
    "ainda",
    "também",
    "logo",
    "assim",
    "and",
    "or",
    "but",
    "yet",
    "however",
    "therefore",
    "because",
    "while",
    "when",
    "then",
    "also",
    "moreover",
    "so",
}
_AUXILIARIES = {
    "é",
    "são",
    "foi",
    "era",
    "eram",
    "ser",
    "estar",
    "estava",
    "está",
    "estão",
    "sendo",
    "tem",
    "têm",
    "ter",
    "há",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
}
_FILLERS = {
    "muito",
    "muitos",
    "muitas",
    "bastante",
    "essencialmente",
    "realmente",
    "basicamente",
    "apenas",
    "quase",
    "junto",
    "além",
    "todos",
    "todas",
    "very",
    "quite",
    "really",
    "basically",
    "essentially",
    "just",
}

_NEGATIONS = {
    "não",
    "nao",
    "nunca",
    "jamais",
    "nem",
    "sem",
    "nada",
    "not",
    "never",
    "no",
    "nothing",
    "without",
}


def make_compressor(level: int) -> Callable[[str], str] | None:
    """Constrói o compressor do nível pedido; nível 0 retorna None (desligado)."""
    if level <= 0:
        return None
    stops: set[str] = set()
    if level >= 1:
        stops |= _CONNECTIVES | _FILLERS
    if level >= 2:
        stops |= _DETERMINERS | _AUXILIARIES
    if not stops:
        return None
    return lambda text: _compress(text, stops)


def _compress(text: str, stops: set[str]) -> str:
    if not text:
        return text
    kept: list[str] = []
    for token in text.split():
        if token.lower() in stops:
            continue
        kept.append(token)
    # Segurança: nunca devolver vazio — se tudo for função, mantém o original.
    if not kept:
        return text
    collapsed = " ".join(kept)
    # Mantém quebras de linha significativas (separação entre trechos) razoáveis.
    return collapsed.replace(" .", ".").replace(" ,", ",").replace(" :", ":")


def token_savings(original: str, compressed: str) -> float:
    """Economia aproximada de tokens (baseada em whitespace) em 0..1."""
    before = len(original.split())
    after = len(compressed.split())
    if before == 0:
        return 0.0
    return round((before - after) / before, 4)
