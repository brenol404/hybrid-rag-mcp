from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from .config import Settings, get_settings
from .rag.engine import RAGEngine

logger = logging.getLogger("hybrid_rag_mcp.ui")

_HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hybrid RAG MCP - Dashboard Visual</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 50: '#f5f3ff', 500: '#8b5cf6', 600: '#7c3aed', 700: '#6d28d9' }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col antialiased">
  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/60 sticky top-0 z-50 backdrop-blur-md">
    <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="h-9 w-9 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center font-bold text-white shadow-lg shadow-brand-500/20">
          ⚡
        </div>
        <div>
          <h1 class="text-base font-semibold leading-none flex items-center gap-2">
            Hybrid RAG MCP
            <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Live Inspector</span>
          </h1>
          <p class="text-xs text-slate-400 mt-1">Qdrant Vector + BM25 Lexical via RRF Fusion</p>
        </div>
      </div>
      <!-- Tab Navigation -->
      <nav class="flex space-x-1 bg-slate-800/80 p-1 rounded-xl border border-slate-700/50">
        <button onclick="switchTab('search')" id="tab-search" class="tab-btn px-4 py-1.5 rounded-lg text-xs font-medium bg-brand-600 text-white shadow">
          🔍 Busca Híbrida
        </button>
        <button onclick="switchTab('ask')" id="tab-ask" class="tab-btn px-4 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition">
          💬 Chat Agent
        </button>
        <button onclick="switchTab('chunks')" id="tab-chunks" class="tab-btn px-4 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition">
          📚 Chunks & Ingestão
        </button>
        <button onclick="switchTab('stats')" id="tab-stats" class="tab-btn px-4 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition">
          📊 Métricas & Saúde
        </button>
      </nav>
    </div>
  </header>

  <!-- Main Content Area -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 space-y-6">
    <!-- TAB 1: SEARCH INSPECTOR -->
    <section id="view-search" class="space-y-6">
      <!-- Search Form -->
      <div class="glass rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex flex-col md:flex-row gap-3">
          <div class="relative flex-1">
            <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">🔍</span>
            <input type="text" id="searchInput" placeholder="Digite uma pergunta técnica ou palavras-chave (ex: 'arquitetura RAG' ou 'embeddings')..." 
                   class="w-full bg-slate-900 border border-slate-700 rounded-xl pl-10 pr-4 py-2.5 text-sm focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition">
          </div>
          <button onclick="executeSearch()" id="btnSearch" class="bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white font-medium px-6 py-2.5 rounded-xl text-sm transition shadow-lg shadow-brand-600/30 flex items-center justify-center gap-2">
            <span>Inspecionar Busca</span>
          </button>
        </div>

        <!-- Controls / Weights -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-slate-800 text-xs text-slate-400">
          <div class="flex items-center gap-3">
            <label for="topKInput" class="whitespace-nowrap font-medium text-slate-300">Top K:</label>
            <input type="number" id="topKInput" min="1" max="50" value="5" class="w-16 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200">
          </div>
          <div class="flex items-center gap-3">
            <label class="whitespace-nowrap font-medium text-slate-300">Peso Vetorial (Dense):</label>
            <input type="range" id="weightVector" min="0" max="2" step="0.1" value="1.0" class="w-24 accent-brand-500" oninput="document.getElementById('valWVector').innerText = this.value">
            <span id="valWVector" class="font-mono text-slate-200">1.0</span>
          </div>
          <div class="flex items-center gap-3">
            <label class="whitespace-nowrap font-medium text-slate-300">Peso Léxico (BM25):</label>
            <input type="range" id="weightLexical" min="0" max="2" step="0.1" value="1.0" class="w-24 accent-brand-500" oninput="document.getElementById('valWLexical').innerText = this.value">
            <span id="valWLexical" class="font-mono text-slate-200">1.0</span>
          </div>
        </div>
      </div>

      <!-- Results Grid (3 Columns Comparison) -->
      <div id="searchResults" class="grid grid-cols-1 lg:grid-cols-3 gap-6 hidden">
        <!-- Coluna 1: RRF Fusion (Final) -->
        <div class="space-y-3">
          <div class="flex items-center justify-between pb-2 border-b border-indigo-500/30">
            <h2 class="text-sm font-semibold text-indigo-400 flex items-center gap-2">
              <span class="h-2 w-2 rounded-full bg-indigo-400"></span>
              Fusão RRF (Resultado Final)
            </h2>
            <span id="countHybrid" class="text-xs font-mono text-slate-400">0 hits</span>
          </div>
          <div id="listHybrid" class="space-y-3"></div>
        </div>

        <!-- Coluna 2: Semantic (Qdrant) -->
        <div class="space-y-3">
          <div class="flex items-center justify-between pb-2 border-b border-violet-500/30">
            <h2 class="text-sm font-semibold text-violet-400 flex items-center gap-2">
              <span class="h-2 w-2 rounded-full bg-violet-400"></span>
              Vetorial Semântico (Qdrant)
            </h2>
            <span id="countVector" class="text-xs font-mono text-slate-400">0 hits</span>
          </div>
          <div id="listVector" class="space-y-3"></div>
        </div>

        <!-- Coluna 3: Lexical (BM25) -->
        <div class="space-y-3">
          <div class="flex items-center justify-between pb-2 border-b border-emerald-500/30">
            <h2 class="text-sm font-semibold text-emerald-400 flex items-center gap-2">
              <span class="h-2 w-2 rounded-full bg-emerald-400"></span>
              Léxico por Palavra-Chave (BM25)
            </h2>
            <span id="countLexical" class="text-xs font-mono text-slate-400">0 hits</span>
          </div>
          <div id="listLexical" class="space-y-3"></div>
        </div>
      </div>
    </section>

    <!-- TAB 2: CHAT AGENT -->
    <section id="view-ask" class="space-y-6 hidden">
      <div class="glass rounded-2xl p-6 shadow-xl space-y-4">
        <h2 class="text-base font-semibold text-slate-200">Pergunte ao Agente RAG</h2>
        <div class="flex gap-3">
          <input type="text" id="askInput" placeholder="Faça uma pergunta com base nos documentos..." 
                 class="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition">
          <button onclick="executeAsk()" id="btnAsk" class="bg-brand-600 hover:bg-brand-500 text-white font-medium px-6 py-2.5 rounded-xl text-sm transition shadow flex items-center gap-2">
            <span>Perguntar</span>
          </button>
        </div>
      </div>

      <!-- Agent Response Box -->
      <div id="askResultBox" class="glass rounded-2xl p-6 shadow-xl space-y-4 hidden">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-xs font-semibold px-2 py-0.5 rounded-full bg-brand-500/20 text-brand-300 border border-brand-500/30" id="askProviderBadge">FallbackLLM</span>
            <span class="text-xs font-mono text-slate-400" id="askElapsed">0.0s</span>
          </div>
          <span id="cacheHitBadge" class="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">Sem Cache</span>
        </div>
        <div class="prose prose-invert max-w-none text-sm text-slate-200 bg-slate-900/60 p-4 rounded-xl border border-slate-800/80 leading-relaxed whitespace-pre-wrap font-sans" id="askAnswer">
        </div>
        <!-- Trace Steps -->
        <div class="space-y-2 pt-2 border-t border-slate-800">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-400">Trilha de Execução (Agent Trace)</h3>
          <div id="askTraceList" class="space-y-1.5 text-xs font-mono"></div>
        </div>
        <!-- Sources -->
        <div class="space-y-2 pt-2 border-t border-slate-800">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-400">Fontes Utilizadas</h3>
          <div id="askSourcesList" class="flex flex-wrap gap-2 text-xs"></div>
        </div>
      </div>
    </section>

    <!-- TAB 3: CHUNKS EXPLORER & INGESTION -->
    <section id="view-chunks" class="space-y-6 hidden">
      <!-- Ingest Direct Text -->
      <div class="glass rounded-2xl p-6 shadow-xl space-y-4">
        <h2 class="text-sm font-semibold text-slate-200 flex items-center justify-between">
          <span>Adicionar Conteúdo Rápido ao Corpus</span>
          <span class="text-xs font-normal text-slate-400">Indexação instantânea no Qdrant e BM25</span>
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
          <input type="text" id="ingestDocName" placeholder="Nome do documento (ex: manual.md)" 
                 class="md:col-span-1 bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-brand-500">
          <textarea id="ingestContent" placeholder="Cole o texto ou markdown a ser indexado..." rows="3"
                    class="md:col-span-3 bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-brand-500"></textarea>
        </div>
        <div class="flex justify-end">
          <button onclick="executeIngest()" id="btnIngest" class="bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-5 py-2 rounded-xl text-sm transition shadow flex items-center gap-2">
            <span>Indexar Texto</span>
          </button>
        </div>
      </div>

      <!-- Chunks Table -->
      <div class="glass rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <span>Chunks Indexados</span>
            <span id="chunksBadge" class="text-xs font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded-full">0</span>
          </h2>
          <button onclick="loadChunks()" class="text-xs text-brand-400 hover:text-brand-300 font-medium">Recarregar</button>
        </div>
        <div id="chunksContainer" class="space-y-3 max-h-[550px] overflow-y-auto pr-1"></div>
      </div>
    </section>

    <!-- TAB 4: METRICS & SYSTEM HEALTH -->
    <section id="view-stats" class="space-y-6 hidden">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4" id="statsCards">
        <!-- Loaded via JS -->
      </div>
      <div class="glass rounded-2xl p-6 shadow-xl space-y-3">
        <h3 class="text-sm font-semibold text-slate-200">Métricas Detalhadas do Prometheus</h3>
        <pre id="prometheusRaw" class="bg-slate-900 p-4 rounded-xl text-xs font-mono text-slate-300 overflow-x-auto max-h-96"></pre>
      </div>
    </section>
  </main>

  <script>
    function switchTab(tab) {
      ['search', 'ask', 'chunks', 'stats'].forEach(t => {
        document.getElementById(`view-${t}`).classList.add('hidden');
        const b = document.getElementById(`tab-${t}`);
        b.className = 'tab-btn px-4 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition';
      });
      document.getElementById(`view-${tab}`).classList.remove('hidden');
      const activeBtn = document.getElementById(`tab-${tab}`);
      activeBtn.className = 'tab-btn px-4 py-1.5 rounded-lg text-xs font-medium bg-brand-600 text-white shadow';

      if (tab === 'chunks') loadChunks();
      if (tab === 'stats') loadStats();
    }

    async function executeSearch() {
      const query = document.getElementById('searchInput').value.trim();
      if (!query) return;
      const topK = parseInt(document.getElementById('topKInput').value) || 5;
      const wv = parseFloat(document.getElementById('weightVector').value) || 1.0;
      const wl = parseFloat(document.getElementById('weightLexical').value) || 1.0;

      const btn = document.getElementById('btnSearch');
      btn.innerText = 'Buscando...';
      btn.disabled = true;

      try {
        const resp = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k: topK, weights: [wv, wl] })
        });
        const data = await resp.json();
        renderSearchHits(data);
      } catch (err) {
        alert('Erro ao buscar: ' + err);
      } finally {
        btn.innerText = 'Inspecionar Busca';
        btn.disabled = false;
      }
    }

    function renderSearchHits(data) {
      document.getElementById('searchResults').classList.remove('hidden');

      const renderList = (elId, countId, hits, tagColor) => {
        const el = document.getElementById(elId);
        document.getElementById(countId).innerText = `${hits.length} hits`;
        if (hits.length === 0) {
          el.innerHTML = '<div class="text-xs text-slate-500 italic p-4 text-center">Nenhum resultado encontrado</div>';
          return;
        }
        el.innerHTML = hits.map((h, i) => `
          <div class="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2 hover:border-slate-700 transition">
            <div class="flex items-center justify-between text-xs">
              <span class="font-medium text-slate-200 truncate max-w-[180px]" title="${h.doc_name}">📄 ${h.doc_name}</span>
              <span class="font-mono text-[11px] px-2 py-0.5 rounded-md ${tagColor}">Score: ${h.score.toFixed(4)}</span>
            </div>
            <p class="text-xs text-slate-300 line-clamp-3 font-sans leading-relaxed">${escapeHtml(h.content)}</p>
            <div class="text-[10px] text-slate-500 font-mono">ID: ${h.chunk_id}</div>
          </div>
        `).join('');
      };

      renderList('listHybrid', 'countHybrid', data.hybrid || [], 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30');
      renderList('listVector', 'countVector', data.vector || [], 'bg-violet-500/20 text-violet-300 border border-violet-500/30');
      renderList('listLexical', 'countLexical', data.lexical || [], 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30');
    }

    async function executeAsk() {
      const question = document.getElementById('askInput').value.trim();
      if (!question) return;

      const btn = document.getElementById('btnAsk');
      btn.innerText = 'Pensando...';
      btn.disabled = true;

      try {
        const resp = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const res = await resp.json();
        
        document.getElementById('askResultBox').classList.remove('hidden');
        document.getElementById('askAnswer').innerText = res.answer;
        document.getElementById('askProviderBadge').innerText = `${res.provider} (${res.model})`;
        document.getElementById('askElapsed').innerText = `${(res.elapsed_ms / 1000).toFixed(2)}s`;
        
        const cacheEl = document.getElementById('cacheHitBadge');
        if (res.cache_hit) {
          cacheEl.className = 'text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
          cacheEl.innerText = '⚡ Cache Hit';
        } else {
          cacheEl.className = 'text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400';
          cacheEl.innerText = 'Cache Miss (Gerado)';
        }

        const traceEl = document.getElementById('askTraceList');
        traceEl.innerHTML = (res.trace || []).map(t => `
          <div class="flex items-center gap-2 text-slate-400">
            <span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 uppercase text-[10px]">${t.step}</span>
            <span>${escapeHtml(t.detail)}</span>
          </div>
        `).join('');

        const srcEl = document.getElementById('askSourcesList');
        srcEl.innerHTML = (res.sources || []).map(s => `
          <span class="px-2 py-1 rounded bg-slate-800/80 border border-slate-700/80 text-slate-300 font-mono">📄 ${s}</span>
        `).join('');

      } catch (err) {
        alert('Erro ao consultar agente: ' + err);
      } finally {
        btn.innerText = 'Perguntar';
        btn.disabled = false;
      }
    }

    async function executeIngest() {
      const docName = document.getElementById('ingestDocName').value.trim();
      const content = document.getElementById('ingestContent').value.trim();
      if (!docName || !content) {
        alert('Preencha o nome do documento e o conteúdo.');
        return;
      }

      const btn = document.getElementById('btnIngest');
      btn.innerText = 'Indexando...';
      btn.disabled = true;

      try {
        const resp = await fetch('/api/ingest_text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ doc_name: docName, content })
        });
        const data = await resp.json();
        alert(`Sucesso! Indexados ${data.chunks} novos chunks. Total no índice: ${data.indexed}`);
        document.getElementById('ingestDocName').value = '';
        document.getElementById('ingestContent').value = '';
        loadChunks();
      } catch (err) {
        alert('Erro ao indexar texto: ' + err);
      } finally {
        btn.innerText = 'Indexar Texto';
        btn.disabled = false;
      }
    }

    async function loadChunks() {
      const container = document.getElementById('chunksContainer');
      container.innerHTML = '<div class="text-xs text-slate-500 italic p-4 text-center">Carregando chunks...</div>';
      try {
        const resp = await fetch('/api/chunks');
        const chunks = await resp.json();
        document.getElementById('chunksBadge').innerText = chunks.length;

        if (chunks.length === 0) {
          container.innerHTML = '<div class="text-xs text-slate-500 italic p-4 text-center">Nenhum chunk indexado ainda. Use o formulário acima para adicionar conteúdo!</div>';
          return;
        }

        container.innerHTML = chunks.map(c => `
          <div class="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
            <div class="flex items-center justify-between text-xs">
              <span class="font-medium text-brand-300">📄 ${c.doc_name}</span>
              <span class="font-mono text-[10px] text-slate-500">${c.chunk_id}</span>
            </div>
            <p class="text-xs text-slate-300 font-sans leading-relaxed">${escapeHtml(c.content)}</p>
          </div>
        `).join('');
      } catch (err) {
        container.innerHTML = `<div class="text-xs text-rose-500 p-4 text-center">Erro: ${err}</div>`;
      }
    }

    async function loadStats() {
      try {
        const resp = await fetch('/api/stats');
        const stats = await resp.json();

        const cardsEl = document.getElementById('statsCards');
        cardsEl.innerHTML = `
          <div class="glass p-5 rounded-2xl space-y-1">
            <div class="text-xs font-medium text-slate-400">Total Chunks Vetoriais</div>
            <div class="text-2xl font-bold text-white font-mono">${stats.vector_chunks}</div>
            <div class="text-[11px] text-slate-500">Qdrant (${stats.qdrant_mode})</div>
          </div>
          <div class="glass p-5 rounded-2xl space-y-1">
            <div class="text-xs font-medium text-slate-400">Total Chunks Léxicos</div>
            <div class="text-2xl font-bold text-white font-mono">${stats.lexical_chunks}</div>
            <div class="text-[11px] text-slate-500">SQLite BM25 Index</div>
          </div>
          <div class="glass p-5 rounded-2xl space-y-1">
            <div class="text-xs font-medium text-slate-400">Embedder Ativo</div>
            <div class="text-base font-semibold text-slate-200 truncate" title="${stats.embed_model}">${stats.embed_model}</div>
            <div class="text-[11px] text-slate-500">Dimensão: ${stats.dim}</div>
          </div>
          <div class="glass p-5 rounded-2xl space-y-1">
            <div class="text-xs font-medium text-slate-400">Cache Semântico</div>
            <div class="text-2xl font-bold text-white font-mono">${stats.cache_enabled ? 'Ativo' : 'Desativado'}</div>
            <div class="text-[11px] text-slate-500">${stats.cache_entries} entradas</div>
          </div>
        `;

        document.getElementById('prometheusRaw').innerText = stats.prometheus || 'Nenhuma métrica coletada ainda.';
      } catch (err) {
        console.error('Erro ao carregar estatísticas:', err);
      }
    }

    function escapeHtml(text) {
      if (!text) return '';
      return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }
  </script>
</body>
</html>
"""


class UIApp:
    def __init__(self, engine: RAGEngine | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or RAGEngine(self.settings)

    def create_app(self) -> Starlette:
        routes = [
            Route("/", self.index_endpoint, methods=["GET"]),
            Route("/api/search", self.search_endpoint, methods=["POST"]),
            Route("/api/ask", self.ask_endpoint, methods=["POST"]),
            Route("/api/chunks", self.chunks_endpoint, methods=["GET"]),
            Route("/api/ingest_text", self.ingest_text_endpoint, methods=["POST"]),
            Route("/api/stats", self.stats_endpoint, methods=["GET"]),
            Route("/api/health", self.health_endpoint, methods=["GET"]),
        ]
        return Starlette(routes=routes)

    async def index_endpoint(self, request: Request) -> HTMLResponse:
        return HTMLResponse(_HTML_DASHBOARD)

    async def health_endpoint(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def search_endpoint(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Corpo JSON inválido"}, status_code=400)

        query = body.get("query", "").strip()
        if not query:
            return JSONResponse({"error": "Query vazia"}, status_code=400)

        top_k = int(body.get("top_k", self.settings.top_k))
        weights_list = body.get("weights", [1.0, 1.0])
        weights = (float(weights_list[0]), float(weights_list[1]))

        detailed = self.engine.search_detailed(query=query, top_k=top_k, weights=weights)

        def hit_to_dict(h: Any) -> dict[str, Any]:
            return {
                "chunk_id": h.chunk_id,
                "doc_name": h.doc_name,
                "content": h.content,
                "score": float(h.score),
                "strategy": h.strategy,
            }

        return JSONResponse(
            {
                "query": query,
                "hybrid": [hit_to_dict(h) for h in detailed["hybrid"]],
                "vector": [hit_to_dict(h) for h in detailed["vector"]],
                "lexical": [hit_to_dict(h) for h in detailed["lexical"]],
            }
        )

    async def ask_endpoint(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Corpo JSON inválido"}, status_code=400)

        question = body.get("question", "").strip()
        if not question:
            return JSONResponse({"error": "Pergunta vazia"}, status_code=400)

        top_k = int(body.get("top_k", self.settings.top_k))
        import time

        t0 = time.perf_counter()
        result = self.engine.ask(question, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return JSONResponse(
            {
                "answer": result.answer,
                "sources": [
                    s.doc_name if hasattr(s, "doc_name") else str(s) for s in result.sources
                ],
                "provider": result.provider,
                "model": result.model,
                "cache_hit": result.cache_hit,
                "trace": [
                    {
                        "step": t.step,
                        "detail": t.detail,
                        "ok": t.ok,
                    }
                    for t in result.trace
                ],
                "elapsed_ms": round(elapsed_ms, 1),
            }
        )

    async def chunks_endpoint(self, request: Request) -> JSONResponse:
        all_chunks = self.engine.all_chunks()
        doc_filter = request.query_params.get("doc", "").lower()
        if doc_filter:
            all_chunks = [c for c in all_chunks if doc_filter in c.doc_name.lower()]

        return JSONResponse(
            [
                {
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "content": c.content,
                    "index": c.index,
                }
                for c in all_chunks
            ]
        )

    async def ingest_text_endpoint(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Corpo JSON inválido"}, status_code=400)

        doc_name = body.get("doc_name", "").strip()
        content = body.get("content", "").strip()
        if not doc_name or not content:
            return JSONResponse({"error": "doc_name e content são obrigatórios"}, status_code=400)

        result = self.engine.ingest_text(doc_name, content)
        return JSONResponse(result)

    async def stats_endpoint(self, request: Request) -> JSONResponse:
        chunks = self.engine.all_chunks()
        cache_entries = 0
        if self.engine._cache is not None:
            cache_entries = len(self.engine._cache._entries)

        try:
            dim = self.engine._embedder.dim
        except Exception:
            dim = 1024

        return JSONResponse(
            {
                "vector_chunks": len(chunks),
                "lexical_chunks": len(chunks),
                "qdrant_mode": "servidor" if self.settings.qdrant_url else "local",
                "embed_model": self.settings.embed_model,
                "dim": dim,
                "cache_enabled": self.settings.cache_enabled,
                "cache_entries": cache_entries,
                "prometheus": self.engine.metrics.render_prometheus(),
            }
        )


def build_app(settings: Settings | None = None) -> Starlette:
    ui = UIApp(settings=settings)
    return ui.create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG MCP - Dashboard Visual")
    parser.add_argument("--host", default="127.0.0.1", help="Host HTTP (padrão: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Porta HTTP (padrão: 8501)")
    args = parser.parse_args()

    import uvicorn

    app = build_app()
    print(f"[hybrid-rag-ui] Dashboard Visual iniciado em: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
