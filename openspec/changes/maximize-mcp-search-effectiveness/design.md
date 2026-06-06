## Context

The docsearch project has two search entry points that have diverged:
- **`api.py` `/search`**: Full two-stage retrieval with hard and boost label match modes, min-CE threshold filtering, sigmoid normalization, and low-relevance hints
- **`mcp_server.py` `search_docs`**: Subset of the above — hard-filter only, no boost mode, no min-CE threshold, and inline reimplementations of label parsing/filter building that duplicate `label_resolver.py`

Both entry points call the same Qdrant collection, same bi-encoder/cross-encoder models, and same embedding pipeline. The divergence is purely in the search orchestration layer.

Additional issues exist deeper in the pipeline:
- **Chunker** uses character counts (2000 default) while the bi-encoder (`multi-qa-mpnet-base-cos-v1`) has a 512-token limit (~1500 chars), causing silent truncation
- **Overlap** is only 100 chars (5%) — context at chunk boundaries is lost
- **Chunk payloads** lack structural metadata (page title, section headings, content type)
- **No source diversity** — a single URL can dominate top results
- **No context exploration** — LLMs can't fetch surrounding chunks or all chunks from a URL

## Goals / Non-Goals

**Goals:**
1. Single shared search implementation used by both API `/search` and MCP `search_docs`
2. MCP `search_docs` gains boost mode, min-CE threshold, and source diversity — parity with API
3. Token-aware chunking that respects the bi-encoder's 512-token sequence limit
4. Meaningful chunk overlap (15-20%) to preserve cross-boundary context
5. Structural metadata in chunk payloads: page title, section headings, content type
6. New MCP tools for context exploration: `get_chunks_for_url`, `get_adjacent_chunks`
7. Source diversity cap (max 2 chunks per URL) in search results

**Non-Goals:**
- Changing the embedding model or cross-encoder (left for a future change)
- Adding hybrid/sparse search (BM25) — infrastructure is ready but out of scope
- Query expansion, HyDE, or multi-query retrieval — LLM caller can do this
- Usage-based relevance feedback / learning loop
- Pagination for search results
- Changing the Qdrant collection schema beyond payload additions (no re-index needed)

## Decisions

### Decision 1: Merge search logic into `label_resolver.py` as `search_utils.py`

**Chosen**: Rename/refactor `label_resolver.py` → `search_utils.py` to hold ALL shared search functions: label parsing, filter building, boost blending, result normalization, diversity capping, hint generation.

**Alternatives considered**:
- Keep `label_resolver.py` and add a new `search_core.py` — creates confusion about which module owns what. Single module is clearer.
- Put shared logic in `api.py` and import from MCP — circular dependency risk since MCP imports `api` for models.

**Rationale**: A single, testable module with pure functions avoids the dual-implementation trap. Both API and MCP become thin wrappers that call the same functions.

### Decision 2: Token-aware chunking with tiktoken

**Chosen**: Use `tiktoken` with the `cl100k_base` encoding (same as multi-qa-mpnet's tokenizer) to count tokens and split text. Default chunk size: **400 tokens** (~1500 chars) with **80 token** overlap (20%).

**Alternatives considered**:
- Use SentenceTransformer's tokenizer directly — adds heavy import dependency to chunker; tiktoken is lightweight and fast.
- Keep character-based chunking but reduce size to 1500 — still approximate, paragraphs can exceed limit.
- Use `text_splitter` from langchain — adds a large dependency for one function.

**Rationale**: Token-accurate chunking guarantees no silent truncation. 400 tokens leaves headroom for the model's 512-token limit (special tokens, attention mask padding). 20% overlap preserves cross-boundary context.

### Decision 3: Structural metadata extraction during crawl

**Chosen**: Extract metadata from the crawled markdown BEFORE chunking:
- **Page title**: First `# ` heading in the document
- **Section heading**: The nearest `## ` or `### ` heading preceding each chunk
- **Content type**: Heuristic classifier based on URL patterns and content signals

Store as Qdrant payload fields alongside existing fields.

**Alternatives considered**:
- Post-hoc extraction during search — too expensive (would need to re-parse content on every query)
- Store in SQLite — adds join complexity; Qdrant payload is already fetched with results, zero extra latency
- Skip content type classification — LLMs benefit from knowing whether a chunk is API reference vs. conceptual docs; heuristic is cheap and "good enough"

**Rationale**: Inline extraction during crawl is a one-time cost. Qdrant payload is the natural home — it's returned with every search result without additional queries.

### Decision 4: Source diversity via post-retrieval capping

**Chosen**: After cross-encoder reranking, enforce max 2 chunks per URL before returning top-N results. Fill vacated slots from next-best results from other URLs.

**Alternatives considered**:
- Pre-filter at Qdrant level (group by URL) — Qdrant doesn't support `GROUP BY` in search; would require overscroll + post-processing anyway.
- Diversity reranking (MMR) — more sophisticated but adds latency (pairwise similarity computation). Simple capping achieves 80% of the value at near-zero cost.
- Weighted scoring penalty for same-URL chunks — more gradual but harder to tune; capping is predictable.

**Rationale**: Simple per-URL cap is predictable, testable, and efficient. It ensures the LLM sees breadth across sources.

### Decision 5: Context exploration tools

**Chosen**: Two new MCP tools:
- `get_chunks_for_url(url, limit?, offset?)` — scroll Qdrant for all chunks matching a URL, return with content and metadata. Supports pagination for large docs.
- `get_adjacent_chunks(url, chunk_index, window?)` — fetch chunk_index-1 and chunk_index+1 for surrounding context. Uses Qdrant scroll with URL + chunk_index filter.

**Alternatives considered**:
- Single `explore_context` tool with mode parameter — harder for LLM to discover (MCP tool descriptions are per-function); separate tools are self-documenting.
- Include adjacent chunk content inline in search results — doubles response size even when LLM doesn't need context; on-demand is more efficient.

**Rationale**: Separate, focused tools let the LLM decide when context is needed. They're simple Qdrant scroll operations — fast and cheap.

### Decision 6: No changes to embedding/cross-encoder models

**Chosen**: Keep `multi-qa-mpnet-base-cos-v1` and `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Rationale**: Model changes would require re-embedding the entire corpus and extensive evaluation. This is a separate change. The current models are "good enough" — the improvements here fix the pipeline around them.

## Risks / Trade-offs

- **[Risk] Token-aware chunking changes all chunk boundaries** → All existing chunks need re-ingestion after deploy. Mitigation: `trigger_crawl(mode="all")` handles this; old vectors are purged before new ones inserted.
- **[Risk] `tiktoken` adds a new dependency** → tiktoken is lightweight (~1MB), pure Python, and widely used. No system libraries needed.
- **[Risk] Source diversity may hide the best result** → If the 3rd-best chunk from URL-A is actually the best answer, capping at 2 could hurt. Mitigation: the cap is configurable and the boost mode allows off-label results to surface.
- **[Risk] Content type heuristic may misclassify** → Some docs defy URL-based classification. Mitigation: content type is advisory metadata, not a filter — LLMs see it but aren't forced by it.
- **[Trade-off] Enriched payloads increase storage** → Each point grows by ~200 bytes (title + heading + content_type). For 10k chunks, that's ~2MB. Negligible.

## Open Questions

1. **Content type taxonomy**: What types should the heuristic distinguish? Proposed: `api-reference`, `conceptual`, `tutorial`, `reference`, `changelog`. Confirm with real doc sources.
2. **Diversity cap value**: Is 2 per URL the right default? Could be 1 for small result sets (limit ≤ 5), 3 for larger ones. Start with 2 and iterate.
3. **Adjacent chunk window**: Default 1 (one before, one after)? Or 2 (two in each direction)? Start with 1.
