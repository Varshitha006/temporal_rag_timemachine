"""
visualize_kg.py
----------------
Interactive Knowledge Graph visualiser for TimeMachine.

Produces an offline HTML file using pyvis (with a matplotlib PNG fallback).

Features:
  - Nodes = documents, labelled "D014 v3.0", coloured by category.
  - Node tooltip shows: title, authority, effective date, sunset date.
  - Edges coloured/styled by type:
      supersedes   → solid arrow (green)
      conflict_risk → dashed red
      narrows_scope → dashed orange
      supplements   → dotted grey
      same_category → hidden by default (too many; toggle with --show-same-cat)
  - --as-of YYYY-MM-DD : grey out docs NOT in force on that date.
  - --query "..." --as-of DATE : run retrieval + KG expansion and highlight
      (a) docs RAG matched (blue),
      (b) docs added by KG expansion (orange), with the edge that brought
          each one in shown as a tooltip.
  - --out kg.html (default output filename)
  - --png : also save a matplotlib/networkx PNG (does not require pyvis)
  - --show-same-cat : include same_category edges (hidden by default)

Runs offline — no API key needed (uses skip_llm mode for --query).

Usage examples:
  python visualize_kg.py
  python visualize_kg.py --as-of 2024-09-01 --out kg_sep2024.html
  python visualize_kg.py --as-of 2024-09-01 --png --out kg_sep2024.html
  python visualize_kg.py --query "Are field engineers using shared kiosk devices required to use MFA?" --as-of 2024-09-01

Requirements (added to requirements.txt):
  pip install pyvis
"""

import sys
import argparse
from pathlib import Path
from datetime import date

# Add project root so we can import src.*
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.corpus_loader import load_corpus
from src.knowledge_graph import (
    build_knowledge_graph,
    is_doc_in_force,
    expand_neighbours,
)
from src.temporal_filter import resolve_date


# ── Category → colour mapping ─────────────────────────────────────────────────
# Each category gets a distinct hue so the graph is readable at a glance.
CATEGORY_COLORS = {
    "WFH":                    "#4A90E2",   # Blue
    "Leave":                  "#7ED321",   # Green
    "DataRetention":          "#9B59B6",   # Purple
    "InfoSec":                "#E74C3C",   # Red
    "TravelExpense":          "#F39C12",   # Orange
    "Procurement":            "#1ABC9C",   # Teal
    "CodeOfConduct":          "#2ECC71",   # Emerald
    "BYOD":                   "#3498DB",   # Light blue
    "Contractor":             "#E67E22",   # Carrot
    "Exit":                   "#95A5A6",   # Grey-blue
    "EquipmentAllowance":     "#F1C40F",   # Yellow
    "AntiBribery":            "#E91E63",   # Pink
    "ParentalLeave":          "#00BCD4",   # Cyan
    "BackgroundVerification": "#8D6E63",   # Brown
    "SocialMedia":            "#673AB7",   # Deep Purple
    "Whistleblower":          "#FF5722",   # Deep Orange
    "DataClassification":     "#607D8B",   # Blue-grey
    "SalesEnablement":        "#CDDC39",   # Lime
    "NonPolicy":              "#BDBDBD",   # Light grey
}

DEFAULT_COLOR = "#AAAAAA"
INACTIVE_COLOR = "#D5D5D5"   # used for docs NOT in force on --as-of date

# ── Edge style definitions ────────────────────────────────────────────────────
EDGE_STYLES = {
    "supersedes":    {"color": "#27AE60",  "dash": False, "label": "supersedes",    "width": 3},
    "conflict_risk": {"color": "#E74C3C",  "dash": True,  "label": "conflict_risk", "width": 2},
    "narrows_scope": {"color": "#F39C12",  "dash": True,  "label": "narrows_scope", "width": 2},
    "supplements":   {"color": "#95A5A6",  "dash": True,  "label": "supplements",   "width": 1},
    "same_category": {"color": "#DDDDDD",  "dash": False,  "label": "same_category", "width": 1},
}


def _make_tooltip(doc_id: str, graph, query_date=None):
    """Build the HTML tooltip text shown when hovering a node."""
    attrs = graph.nodes[doc_id]
    in_force_str = ""
    if query_date:
        flag = "✅ IN FORCE" if is_doc_in_force(graph, doc_id, query_date) else "❌ EXPIRED/FUTURE"
        in_force_str = f"<br/><b>{flag}</b>"
    return (
        f"<b>{doc_id}</b> — {attrs.get('title', '')}{in_force_str}<br/>"
        f"Authority: {attrs.get('issuing_authority', '')}<br/>"
        f"Version: {attrs.get('version', '')}<br/>"
        f"Effective: {attrs.get('effective_date', '')}<br/>"
        f"Sunset: {attrs.get('sunset_date') or 'open'}<br/>"
        f"Category: {attrs.get('category', '')}"
    )


def build_pyvis_graph(
    kg,
    metadata,
    as_of_date=None,
    rag_hits=None,
    kg_expansions=None,
    show_same_cat=False,
):
    """
    Build and return a pyvis Network object from the KG.

    Args:
        kg:            The NetworkX MultiDiGraph.
        metadata:      Full metadata dict.
        as_of_date:    Optional date — grey out docs not in force.
        rag_hits:      Set of doc_ids found by RAG (highlighted blue).
        kg_expansions: Dict {doc_id: ExpandedDoc} for KG-expanded docs.
        show_same_cat: Whether to include same_category edges.

    Returns:
        pyvis.network.Network object.
    """
    from pyvis.network import Network

    # Create a directed network with physics for automatic layout
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",   # dark background for premium look
        font_color="#ffffff",
    )
    net.toggle_physics(True)
    # Enable edge labels and smooth curves
    net.set_options("""
    {
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 1 } },
        "smooth": { "type": "curvedCW", "roundness": 0.2 },
        "font": { "size": 10, "color": "#cccccc", "background": "none" }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 150
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "interaction": { "hover": true, "tooltipDelay": 200 }
    }
    """)

    rag_hits = rag_hits or set()
    kg_expansions = kg_expansions or {}

    # ── Add nodes ──────────────────────────────────────────────────────────
    for doc_id in kg.nodes:
        attrs    = kg.nodes[doc_id]
        category = attrs.get("category", "")
        version  = attrs.get("version", "")
        label    = f"{doc_id}\n{version}"
        tooltip  = _make_tooltip(doc_id, kg, query_date=as_of_date)

        # Determine colour
        if doc_id in rag_hits:
            color = "#3498DB"        # blue = RAG hit
            size  = 30
            border = "#2980B9"
        elif doc_id in kg_expansions:
            color = "#F39C12"        # orange = KG expansion
            size  = 28
            border = "#D68910"
        elif as_of_date and not is_doc_in_force(kg, doc_id, as_of_date):
            color = INACTIVE_COLOR   # grey = not in force
            size  = 18
            border = "#AAAAAA"
        else:
            color  = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
            size   = 22
            border = color

        # Add tooltip note for expansion docs
        if doc_id in kg_expansions:
            exp = kg_expansions[doc_id]
            tooltip += f"<br/><b>KG-expanded via {exp.via_doc} --{exp.edge_type}--></b>"

        net.add_node(
            doc_id,
            label=label,
            title=tooltip,
            color={"background": color, "border": border, "highlight": {"background": "#FFD700"}},
            size=size,
            font={"size": 12, "color": "#ffffff" if color != INACTIVE_COLOR else "#888888"},
        )

    # ── Add edges ──────────────────────────────────────────────────────────
    seen_edges: set = set()
    for u, v, key, attrs in kg.edges(keys=True, data=True):
        # Skip same_category edges unless requested
        if key == "same_category" and not show_same_cat:
            continue

        # Deduplicate bidirectional display: show only one direction for
        # symmetric edges like conflict_risk / supplements / same_category
        # (supersedes and narrows_scope are directional so show both).
        edge_tuple = tuple(sorted([u, v]) + [key])
        if key in ("conflict_risk", "supplements", "same_category"):
            if edge_tuple in seen_edges:
                continue
            seen_edges.add(edge_tuple)

        style = EDGE_STYLES.get(key, {"color": "#888888", "dash": False, "label": key, "width": 1})

        edge_opts = {
            "color": style["color"],
            "label": style["label"],
            "width": style["width"],
            "dashes": style["dash"],
        }
        net.add_edge(u, v, **edge_opts)

    return net


def build_matplotlib_png(
    kg,
    metadata,
    output_path: Path,
    as_of_date=None,
    rag_hits=None,
    kg_expansions=None,
    show_same_cat=False,
):
    """
    Render a static PNG using matplotlib + networkx.
    Used as fallback when pyvis is unavailable, or when --png is passed.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import networkx as nx

    rag_hits      = rag_hits or set()
    kg_expansions = kg_expansions or {}

    # Build a simple DiGraph for matplotlib (collapse multi-edges)
    G = nx.DiGraph()
    for doc_id, attrs in kg.nodes(data=True):
        G.add_node(doc_id, **attrs)

    for u, v, key in kg.edges(keys=True):
        if key == "same_category" and not show_same_cat:
            continue
        if not G.has_edge(u, v):   # keep highest-priority edge only
            G.add_edge(u, v, rel=key)

    # Node colours
    node_colors = []
    for doc_id in G.nodes:
        if doc_id in rag_hits:
            node_colors.append("#3498DB")
        elif doc_id in kg_expansions:
            node_colors.append("#F39C12")
        elif as_of_date and not is_doc_in_force(kg, doc_id, as_of_date):
            node_colors.append(INACTIVE_COLOR)
        else:
            cat = G.nodes[doc_id].get("category", "")
            node_colors.append(CATEGORY_COLORS.get(cat, DEFAULT_COLOR))

    # Edge colours from style dict
    edge_colors = [
        EDGE_STYLES.get(G[u][v]["rel"], {}).get("color", "#888888")
        for u, v in G.edges
    ]

    # Spring layout for compact placement
    pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(24, 18))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, arrows=True,
        arrowsize=15, width=1.5, ax=ax, connectionstyle="arc3,rad=0.1"
    )

    # Legend
    from matplotlib.patches import Patch, FancyArrow
    legend_items = [
        Patch(facecolor="#3498DB",  label="RAG hit"),
        Patch(facecolor="#F39C12",  label="KG expansion"),
        Patch(facecolor=INACTIVE_COLOR, label="Not in force"),
    ]
    for rel, style in EDGE_STYLES.items():
        if rel == "same_category" and not show_same_cat:
            continue
        legend_items.append(Patch(facecolor=style["color"], label=rel))
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              facecolor="#2d2d44", labelcolor="white", framealpha=0.8)

    title = "TimeMachine Knowledge Graph"
    if as_of_date:
        title += f"  (as-of {as_of_date})"
    ax.set_title(title, color="white", fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()

    png_path = output_path.with_suffix(".png")
    plt.savefig(str(png_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualize] PNG saved → {png_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise the TimeMachine Knowledge Graph as an interactive HTML."
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Grey out documents NOT in force on this date.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run retrieval + KG expansion and highlight matching docs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("kg.html"),
        help="Output HTML file path (default: kg.html).",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also save a static matplotlib PNG (always saved to <out>.png).",
    )
    parser.add_argument(
        "--show-same-cat",
        action="store_true",
        help="Show same_category edges (hidden by default, many of them).",
    )
    args = parser.parse_args()

    # ── Load corpus + build KG ────────────────────────────────────────────
    print("[Visualize] Loading corpus & building KG...")
    metadata, clauses = load_corpus(ROOT / "corpus")
    kg = build_knowledge_graph(metadata)

    n_sup = sum(1 for _, _, k in kg.edges(keys=True) if k == "supersedes")
    n_con = sum(1 for _, _, k in kg.edges(keys=True) if k == "conflict_risk")
    print(f"[Visualize] KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges "
          f"({n_sup} supersedes, {n_con} conflict_risk)")

    # ── Resolve as-of date ─────────────────────────────────────────────────
    as_of_date = None
    if args.as_of:
        as_of_date = date.fromisoformat(args.as_of)

    # ── Optional: run retrieval + expansion if --query given ───────────────
    rag_hits:      set[str]  = set()
    kg_expansions: dict      = {}

    if args.query:
        if as_of_date is None:
            # Default to current pipeline date if no --as-of given
            as_of_date = resolve_date(None)
            print(f"[Visualize] No --as-of given with --query; defaulting to {as_of_date}")

        print(f"[Visualize] Running retrieval for query: '{args.query}'")
        from src.retriever import Retriever
        retriever = Retriever(clauses, rebuild_index=False)

        raw_results = retriever.search(args.query, k=15)
        from src.temporal_filter import temporal_filter, is_in_force
        filtered = temporal_filter(raw_results, metadata, as_of_date, graph=kg)
        in_force_clauses = [fc for fc in filtered if fc.in_force]

        # Collect RAG-hit doc_ids (top 5)
        for fc in in_force_clauses[:5]:
            rag_hits.add(fc.clause.doc_id)

        print(f"[Visualize] RAG matched docs: {sorted(rag_hits)}")

        # Run KG expansion
        expansions = expand_neighbours(
            graph=kg,
            seed_doc_ids=rag_hits,
            query_date=as_of_date,
            metadata=metadata,
            max_hops=2,
            scope_question=args.query,
        )
        for exp in expansions:
            if exp.doc_id not in rag_hits:
                kg_expansions[exp.doc_id] = exp
                print(f"[Visualize] KG expanded: {exp.via_doc} --{exp.edge_type}--> {exp.doc_id} [{exp.role}]")

    # ── Build pyvis HTML ───────────────────────────────────────────────────
    try:
        net = build_pyvis_graph(
            kg=kg,
            metadata=metadata,
            as_of_date=as_of_date,
            rag_hits=rag_hits,
            kg_expansions=kg_expansions,
            show_same_cat=args.show_same_cat,
        )

        out_path = Path(args.out)
        net.save_graph(str(out_path))
        print(f"[Visualize] HTML saved → {out_path}")

    except ImportError:
        print("[Visualize] WARNING: pyvis not installed. Falling back to PNG only.")
        print("  Install with: pip install pyvis")

    # ── Optional: also save PNG ────────────────────────────────────────────
    if args.png or "pyvis" not in sys.modules:
        try:
            build_matplotlib_png(
                kg=kg,
                metadata=metadata,
                output_path=Path(args.out),
                as_of_date=as_of_date,
                rag_hits=rag_hits,
                kg_expansions=kg_expansions,
                show_same_cat=args.show_same_cat,
            )
        except ImportError:
            print("[Visualize] matplotlib not available; skipping PNG.")

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n[Visualize] Summary:")
    if as_of_date:
        from src.knowledge_graph import get_active_docs_for_date
        active = get_active_docs_for_date(kg, as_of_date)
        print(f"  Documents in force on {as_of_date}: {len(active)}/{kg.number_of_nodes()}")
    if rag_hits:
        print(f"  RAG-matched docs: {sorted(rag_hits)}")
    if kg_expansions:
        print(f"  KG-expanded docs: {sorted(kg_expansions.keys())}")
        for doc_id, exp in sorted(kg_expansions.items()):
            print(f"    {exp.via_doc} --{exp.edge_type}--> {doc_id}  role={exp.role}")


if __name__ == "__main__":
    main()
