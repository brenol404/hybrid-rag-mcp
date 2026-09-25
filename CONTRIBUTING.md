# Contribuindo

Setup (Python 3.11+, Ollama de pé):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ollama pull bge-m3 && ollama pull qwen3:8b
```

## Antes de abrir PR

```bash
ruff check . && ruff format --check .  # lint
mypy src                                # tipos (strict)
pytest -q --cov --cov-fail-under=75     # testes + cobertura
```

O CI (`test`) roda os três + smoke stdio/HTTP; o job `eval` exige Ollama real
e barra `recall@1 < 0.8`. Medições pesadas (`grid_search`, `judge`) rodam
localmente, não no CI.

## Convenções

- Mensagens de commit em PT, no imperativo, com o porquê no corpo.
- Decisões de arquitetura vão para `docs/adr/` (veja os existentes).
- Sem dependência nova no runtime sem discutir: o core é `pip install` puro.
- Métricas no README só com metodologia reproduzível junto.
