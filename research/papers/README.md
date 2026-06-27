# Research papers (downloaded)

Core literature for Memgram's memory design. PDFs are gitignored (keep the repo
light); this index records what each is and why it matters.

| File | Paper | arXiv | Why it matters for Memgram |
|---|---|---|---|
| `MemGPT_2310.08560.pdf` | MemGPT: Towards LLMs as Operating Systems | 2310.08560 | Self-editing memory via function calls (`core_memory_replace` = contradiction); memory-pressure paging = synchronous summarizer; heartbeats = multi-hop retrieval |
| `Mem0_2504.19413.pdf` | Mem0: Production-Ready AI Agents with Scalable Long-Term Memory | 2504.19413 | **The ADD/UPDATE/DELETE/NOOP update phase** — the exact contradiction mechanism we lack; LOCOMO eval rigor (10-run averaging) |
| `GenerativeAgents_2304.03442.pdf` | Generative Agents: Interactive Simulacra of Human Behavior | 2304.03442 | Retrieval = recency × **importance** × relevance; reflection trees — basis for importance scoring at write |
| `LOCOMO_2402.17753.pdf` | Evaluating Very Long-Term Conversational Memory of LLM Agents | 2402.17753 | The benchmark (single/multi-hop/**temporal**/open-domain) to measure contradiction properly |
| `A-MEM_2502.12110.pdf` | A-MEM: Agentic Memory for LLM Agents | 2502.12110 | Zettelkasten-style dynamic linking/evolution of memories (future: memory organization) |
| `MemorySurvey_2512.13564.pdf` | Memory in the Age of AI Agents (survey, Dec 2025) | 2512.13564 | Recent taxonomy of the whole field — map Memgram against it |
| `LongTermConvMemBaseline_2511.17208.pdf` | A Simple Yet Strong Baseline for Long-Term Conversational Memory | 2511.17208 | Strong/simple baseline on long-term conv memory — sanity check for our approach |

Re-download: `curl -L -o <file> https://arxiv.org/pdf/<id>`.
Extract text: `python -c "from pypdf import PdfReader; print(PdfReader('f.pdf').pages[0].extract_text())"`.
