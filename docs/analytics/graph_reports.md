# ScriptBots Analytics: Lineage & Interaction Graph Reports

**Tracking Beads:** `bd-2z0.11.7`, `bd-2z0.11.11`, `bd-2z0.11.12`
**Applies to:** `scriptbots-analytics`, `scriptbots-storage`, `sb-analyze`, `fnx` ecosystem (`fnx-classes`, `fnx-algorithms`, `fnx-readwrite`).

---

## 1. Executive Summary

Evolutionary simulations in `ScriptBots` generate rich relational structures: directed acyclic kinship graphs (parent-child lineages) and directed interaction networks (combat, mating, food-sharing, and sensory encounters).

`scriptbots-analytics` integrates the `fnx` graph analysis suite (`fnx-classes`, `fnx-algorithms`, and `fnx-readwrite` version `0.2.0`) to provide high-throughput, topologically verified graph analytics reports and export tooling:

1. **`lineage-structure` (`LINEAGE_STRUCTURE_SCHEMA_ID_V1`)**:
   - Constructs a directed graph of agent ancestry from `PersistedAncestryBirth` records.
   - Verifies acyclicity (strongly connected components / SCCs must contain 0 non-trivial cycles).
   - Segregates lineages into independent founder dynasties via weakly connected components (WCCs).
   - Computes generation depth via DAG longest-path topological relaxation.
   - Evaluates in-degree (fecundity / parenthood) and out-degree distributions.
2. **`dynasty-communities` (`DYNASTY_COMMUNITIES_SCHEMA_ID_V1`)**:
   - Computes community structure over the undirected lineage projection using the Louvain modularity optimization algorithm.
   - Evaluates modularity ($Q$) according to the Newman-Girvan formulation.
   - Benchmarks community alignment against phylogenetic founder families via contingency matrix purity and the Adjusted Rand Index (ARI).
3. **`interaction-centrality` (`INTERACTION_CENTRALITY_SCHEMA_ID_V1`)**:
   - Evaluates social and combat interaction networks between agents.
   - Calculates directed in-degree, out-degree, betweenness centrality (via Brandes' shortest-path algorithm), and PageRank.
   - Reports canonical interaction selection, repeated-edge attributes and run-wide capture accounting.
4. **Graph Export & Interoperability (`sb-analyze export-graph`)**:
   - Exports lineage, dynasty, and interaction graphs to standard formats: GraphML (`.graphml`) and strict Edge-List (`.edgelist`).

---

## 2. Graph Construction & The String-Key Tax

### 2.1 Graph Models in `fnx-classes`

`fnx-classes` 0.2.0 uses `DiGraph` and `Graph` parameterized by string node identifiers (`&str` / `String`). Nodes and edges support arbitrary key-value attribute maps (`ValueMap`).

- **Lineage DAG (`DiGraph`)**:
  - Directed edge $(P, C)$ signifies parent $P \to \text{child } C$.
  - Node attributes: `generation` (u64), `birth_tick` (u64).
  - Bulk construction: Edges are populated using `DiGraph::extend_edges_unrecorded` to minimize graph modification audit overhead.
  - Isolated founders (founder agents that leave no offspring) are explicitly added to ensure 100% census representation matching the persistence database.
- **Dynasty Projection (`Graph`)**:
  - Undirected projection generated via `digraph.to_undirected()` preserving bidirectional topological connectivity for community detection.
- **Interaction Graph (`DiGraph`)**:
  - Directed edge $(S, T)$ signifies actor $S \to \text{target } T$.
  - Edge attributes: `weight` (f64, sum of selected magnitudes), `count` (integer, selected event multiplicity).
  - Combat damage and food-share energy retain their recorded magnitudes; their sum is not a normalized physical quantity.
  - Nodes are the actors and targets of selected events. Agents with no selected interactions are excluded.
  - Missing/nonfinite magnitudes and nonfinite aggregate weights are errors.

### 2.2 The String-Key Performance Tax

`ScriptBots` internal identifiers are compact integers (`AgentId`, `u64`). Because `fnx-classes` requires `String` node identifiers, conversion incurs heap allocation and hashing overhead:

- For large populations ($N \ge 100{,}000$), node formatting (`agent_id.to_string()`) and edge string tuples account for a measurable portion of graph construction time.
- To mitigate this tax:
  - Graph construction utilizes reserved string capacities and bulk insertion (`extend_edges_unrecorded`).
  - Algorithms that query node attributes minimize repeated lookups by indexing node indices sequentially.
  - Granular stage telemetry breaks out extraction, construction, and algorithmic analysis timings.

---

## 3. Algorithms & Metrics

### 3.1 Lineage Structure Analysis

| Metric / Stage | Method / Algorithm | Validation / Guarantee |
| :--- | :--- | :--- |
| **Acyclicity (SCC)** | `strongly_connected_components` | Strict assertion: no non-trivial SCC ($|c| \le 1$ for all components). Any cycle indicates simulation corruption. |
| **Founder Families (WCC)** | `weakly_connected_components` | Partitions the lineage forest into disconnected trees originating from distinct founders. |
| **Generation Depth** | Topological sort + Longest path relaxation | Finds maximum ancestor-to-descendant chain length in the DAG. |
| **Degree Distributions** | In-degree and out-degree histograms | Measures fecundity skew (number of offspring per parent). |

### 3.2 Dynasty Communities & Modularity

The `dynasty-communities` report characterizes sub-family clustering and speciation:

- **Louvain Modularity Optimization**:
  - Invokes `fnx_algorithms::louvain_communities_with_params(graph, resolution, seed, max_iter)`.
  - Configurable resolution parameter $\gamma \in (0, 2.0]$ (default: $1.0$) controls community scale.
  - PRNG seed ensures fully deterministic, reproducible partition assignments.
- **Newman-Girvan Modularity ($Q$)**:
  - Calculated via `fnx_algorithms::modularity_with_resolution(graph, communities, resolution)`.
  - $Q \approx 0$ indicates random connectivity; $Q > 0.4$ reveals strong modular isolation between clades.
- **Founder Alignment (ARI & Purity)**:
  - Computes the contingency matrix between founder WCC labels and Louvain community labels.
  - **Purity**: $\frac{1}{N} \sum_k \max_j |C_k \cap F_j|$.
  - **Adjusted Rand Index (ARI)**: Corrects for chance agreement between phylogenetic ancestry and graph modularity.

### 3.3 Interaction Centrality

The canonical interaction events currently record completed combat hits and food sharing.
All three centrality algorithms use the unweighted simple directed graph. The count and
weight attributes are preserved for export; they do not change these centrality scores.

- **Degree Centrality**:
  - Out-degree: Number of distinct selected targets.
  - In-degree: Number of distinct selected actors targeting this agent.
- **Betweenness Centrality**:
  - Directed betweenness computed via `fnx_algorithms::betweenness_centrality_directed(digraph)`.
  - Identifies bridge agents that connect otherwise isolated social clusters.
  - Above 1,000 nodes, defaults to 100 source nodes; `sample_k` explicitly requests a
    positive source count. Sources are ordered by BLAKE3(seed, node identity), making
    the declared seed effective. The report records the exact source UIDs and seed.
- **PageRank**:
  - Directed PageRank computed via `fnx_algorithms::pagerank_directed(digraph)`.
  - Measures steady-state influence within the interaction network.

---

## 4. Selection, Bounds and Capture Evidence

Reports and interaction exports call the same finished-run reader over `interactions`.
The writer projects these rows from typed replay interaction events under the same
run/tick/sequence identity. Empty canonical results do not trigger a replay fallback.

| Parameter | Contract |
| :--- | :--- |
| Neither tick bound | `recent_page`: newest `limit` events, returned in ascending `(tick, seq)` order. |
| `start_tick` and `end_tick` | `complete_window`: every persisted event in `[start_tick, end_tick)`. Both bounds are required and start must be less than end. |
| `limit` | Maximum selected rows, default and maximum 4,096. A complete window exceeding it fails. Zero produces an empty recent page, or accepts only an empty complete window. |
| `max_projected_bytes` | Bounds fixed-width `InteractionGraphEvent` values, including one overflow sentinel. Default derives from the Rust type's size and maximum row count. Narrative payloads and coordinates are not loaded. |
| `max_graph_work` | Conservative all-source betweenness node/edge visit budget, checked from the requested row cap before loading. Default is `6 × 4096²`. It is not a wall-clock or whole-process memory bound. |
| `deadline_ms`, `fallback` | Refused: the synchronous offline SQL reader has no hard deadline, and replay selection is a different population. |

The report's `input` identifies the run, selection, ordering, budgets, selected
event IDs, total canonical run rows, and whether a recent page omitted older rows.
SQL execution time and the engine's working memory remain unbounded; result bounds
must not be interpreted as a hard database execution budget.

Capture counters describe the **whole run**: observed, persisted, sampled-out and
truncated events. Their accounting identity and agreement with the canonical row
count are checked. Missing counters mean `unknown`; an empty selection alone never
proves there were no encounters. Run-wide omissions cannot be localized to a tick
sub-window from these counters. Capture completeness is separate from page truncation
and from the algorithms' unweighted semantics.

---

## 5. Graph Export & CLI Subcommands

### 5.1 CLI Command: `sb-analyze export-graph`

The analytics binary exposes direct graph export functionality:

```bash
# Export lineage DAG to GraphML format
sb-analyze ./simulation.db export-graph --graph lineage --format graphml --out ./lineage.graphml

# Export dynasty undirected projection to edge-list format
sb-analyze ./simulation.db export-graph --graph dynasty --format edgelist --out ./dynasty.edgelist

# Export interaction network to GraphML format
sb-analyze ./simulation.db export-graph --graph interaction --format graphml \
  --params start_tick=100 --params end_tick=200 --out ./interactions.graphml

# The report selects exactly the same complete window
sb-analyze ./simulation.db run interaction-centrality \
  --params start_tick=100 --params end_tick=200 --json ./interactions.json
```

For databases containing multiple runs, supply `--run-id <canonical-128-bit-hex-id>`.
Omitting it on an ambiguous database fails; run IDs do not alter agent identities.

### 5.2 Supported Formats

- **GraphML (`.graphml`)**: Full XML specification including node and edge attributes (`birth_tick`, `generation`, `weight`, `count`). Compatible with Gephi, Cytoscape, and NetworkX.
- **Strict Edge-List (`.edgelist`)**: fnx attributed syntax: `source target count=2;weight=5.5`. Both attributes survive parsing; this is not a two-column or weight-only edge list.

Interaction GraphML embeds JSON selection evidence in the graph attribute
`scriptbots_interaction_evidence`. Interaction edge lists carry the same JSON in
their first comment line, `# scriptbots.interaction-edgelist.v1 <JSON>`. Preserve
that header when exchanging files so the selected population remains identifiable.

---

## 6. Execution Telemetry & Performance Budgets

All graph analysis functions emit structured microsecond/millisecond timings in their report telemetry:

- `extraction_time_ms`: Time spent querying SQLite and decoding records.
- `graph_build_time_ms`: Time spent instantiating `fnx-classes` structures and populating nodes/edges.
- `analysis_time_ms`: Execution time for `fnx-algorithms` graph procedures (WCC, SCC, longest path, Louvain, betweenness, PageRank).
- `total_duration_ms`: Wall-clock end-to-end execution.

The 100,000-edge lineage smoke test records its actual stage timings and enforces
its existing 15-second budget. Earlier example timings here had no retained source
or host identity and are withdrawn. Performance claims require the pinned DSR lane.
