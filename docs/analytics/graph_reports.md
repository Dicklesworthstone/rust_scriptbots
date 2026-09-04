# ScriptBots Analytics: Lineage & Interaction Graph Reports

**Tracking Bead:** `bd-2z0.11.7`  
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
   - Documents the pairwise interaction persistence gap between full replay event logs and relational SQLite tables.
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
  - Edge attributes: `weight` (f64, cumulative interaction magnitude), `count` (u64, interaction frequency).

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

Interactions reflect competition, predation, mating attempts, and sensor collisions:

- **Degree Centrality**:
  - Out-degree: Initiated encounters (aggression/activity).
  - In-degree: Targeted encounters (vulnerability/attractiveness).
- **Betweenness Centrality**:
  - Directed betweenness computed via `fnx_algorithms::betweenness_centrality_directed(digraph)`.
  - Identifies bridge agents that connect otherwise isolated social clusters.
- **PageRank**:
  - Directed PageRank computed via `fnx_algorithms::pagerank_directed_with_params(digraph, alpha, max_iter, tol)`.
  - Measures steady-state influence within the interaction network.

---

## 4. Pairwise Persistence Gap Analysis

### 4.1 The Divergence

During simulation execution, agent interactions occur at high frequency (thousands per tick). Two recording paths exist in the architecture:

1. **Replay Event Stream (`replay_events` table)**:
   - Serialized stream of `ReplayEventKind::Interaction { tick, ordinal, kind, magnitude }`.
   - Captures actor ordinal, target ordinal, interaction category (combat, feeding, mating, collision), and scalar magnitude.
   - Complete, chronological, but requires linear parsing and decoding of compressed payload chunks.
2. **Relational Interaction Store (`interactions` table)**:
   - Planned normalized relational schema: `(tick, actor_uid, target_uid, interaction_kind, magnitude)`.
   - Currently omitted or sparse in standard storage workloads to conserve I/O bandwidth during high-tick runs.

### 4.2 Fallback Strategy in `scriptbots-analytics`

To provide robust reporting across both storage modes:
- `analyze_interaction_centrality` first queries the relational `interactions` table if present.
- When `interactions` contains zero records, it transparently falls back to extracting interaction events from the `replay_events` stream.
- When neither source contains pairwise interaction data, the report completes cleanly with `nodes: 0`, `edges: 0`, empty centralities, and notes the persistence gap in report diagnostics.

---

## 5. Graph Export & CLI Subcommands

### 5.1 CLI Command: `sb-analyze export-graph`

The analytics binary exposes direct graph export functionality:

```bash
# Export lineage DAG to GraphML format
sb-analyze export-graph --db ./simulation.db --kind lineage --format graphml --output ./lineage.graphml

# Export dynasty undirected projection to edge-list format
sb-analyze export-graph --db ./simulation.db --kind dynasty --format edgelist --output ./dynasty.edgelist

# Export interaction network to GraphML format
sb-analyze export-graph --db ./simulation.db --kind interaction --format graphml --output ./interactions.graphml
```

### 5.2 Supported Formats

- **GraphML (`.graphml`)**: Full XML specification including node and edge attributes (`birth_tick`, `generation`, `weight`, `count`). Compatible with Gephi, Cytoscape, and NetworkX.
- **Strict Edge-List (`.edgelist`)**: Plaintext space-delimited format (`source target [weight]`) generated via `fnx_readwrite::EdgeListEngine::strict()`.

---

## 6. Execution Telemetry & Performance Budgets

All graph analysis functions emit structured microsecond/millisecond timings in their report telemetry:

- `extraction_time_ms`: Time spent querying SQLite and decoding records.
- `graph_build_time_ms`: Time spent instantiating `fnx-classes` structures and populating nodes/edges.
- `analysis_time_ms`: Execution time for `fnx-algorithms` graph procedures (WCC, SCC, longest path, Louvain, betweenness, PageRank).
- `total_duration_ms`: Wall-clock end-to-end execution.

### Benchmark Smoke Results (100,000-Edge Synthetic Lineage DAG)
- Graph construction (100,000 edges, 100,001 nodes): ~450 ms.
- Weakly Connected Components: ~120 ms.
- Strongly Connected Components (Acyclicity proof): ~180 ms.
- Longest path generation chain: ~210 ms.
- Total processing time: $\approx 1.2$ seconds.
