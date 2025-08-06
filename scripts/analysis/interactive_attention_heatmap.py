#!/usr/bin/env python3
"""Generate an interactive Plotly heat-map of per-gene attention scores.

The script largely mirrors the original matplotlib implementation but writes a
*stand-alone* HTML file (JavaScript embedded, Plotly shipped via CDN).  You can
open the resulting file in any modern browser (*no Python server needed*) and
zoom, pan, or hover individual cells to inspect exact values.

Usage
-----
$ python scripts/interactive_attention_heatmap.py \
        --tsv attn.tsv \
        --json gene_symbol2ensg.json \
        --out heatmap.html

Arguments are identical to the previous script except that the default output
is now *heatmap.html* instead of a PNG.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", default="attn.tsv", help="Per-gene attention TSV from export_attention.py")
    p.add_argument("--json", default="gene_symbol2ensg.json", help="Priority gene mapping JSON")
    p.add_argument("--out", default="heatmap.html", help="Output HTML file (stand-alone)")
    p.add_argument("--random", type=int, default=100, help="Number of extra random genes to draw in addition to priority list")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    rng = random.Random(42)

    # ------------------------------------------------------------------
    # 1. Load per-gene attention scores (TSV produced by export_attention.py)
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(args.tsv, sep="\t")  # header present
        score_map = dict(zip(df["ensembl_id"], df["max_attention"]))
    except (ValueError, KeyError):
        # Fallback in case TSV has no header
        df = pd.read_csv(args.tsv, sep="\t", header=None, names=["ensembl_id", "rank", "max_attention"])
        score_map = dict(zip(df["ensembl_id"], df["max_attention"]))

    # ------------------------------------------------------------------
    # 2. Priority gene list (symbol → ENSG mapping) & random extras
    # ------------------------------------------------------------------
    sym2ensg = json.load(open(args.json))
    priority_ensg: List[str] = [sym2ensg[sym] for sym in sym2ensg if sym2ensg[sym] in score_map]

    # Draw additional random HVGs not already in the priority list
    remaining = [gid for gid in score_map.keys() if gid not in priority_ensg]
    random_extra = rng.sample(remaining, k=min(args.random, len(remaining)))

    ordered_genes = priority_ensg + random_extra
    scores = np.array([score_map[g] for g in ordered_genes], dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. Build symmetric matrix M[i,j] = (score_i + score_j)/2  (outer-average)
    # ------------------------------------------------------------------
    mat = (scores[:, None] + scores[None, :]) / 2.0  # shape (G, G)

    # ------------------------------------------------------------------
    # 4. Emit stand-alone HTML with embedded data & Plotly heat-map
    # ------------------------------------------------------------------
    import json as pyjson  # avoid confusion with sym2ensg json

    labels = list(sym2ensg.keys()) + [f"extra_{i}" for i in range(len(random_extra))]
    # Use JavaScript-friendly JSON
    labels_js = pyjson.dumps(labels)
    mat_js = pyjson.dumps(mat.tolist())

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <title>Interactive Gene Attention Heat-map</title>
    <script src=\"https://cdn.plot.ly/plotly-2.27.1.min.js\"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; }}
        header {{ padding: 1rem; background: #222; color: #fafafa; }}
        #heatmap {{ width: 100vw; height: 90vh; }}
    </style>
</head>
<body>
    <header>
        <h2>Interactive Gene Attention Heat-map</h2>
        <p>Hover cells for exact average max-attention weights, zoom/pan as needed.</p>
    </header>
    <div id=\"heatmap\"></div>
    <script>
        // Embedded data -------------------------------------------------------
        const labels = {labels_js};
        const z = {mat_js};

        // Build Plotly trace ---------------------------------------------------
        const data = [{{
            z: z,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: {{ title: 'avg max-attention' }},
        }}];

        const layout = {{
            autosize: true,
            margin: {{l: 120, r: 40, t: 20, b: 120}},
            xaxis: {{
                tickangle: 90,
                tickfont: {{size: 8}},
                automargin: true,
            }},
            yaxis: {{
                tickfont: {{size: 8}},
                automargin: true,
            }},
        }};

        // Render --------------------------------------------------------------
        Plotly.newPlot('heatmap', data, layout);
    </script>
</body>
</html>
"""
    Path(args.out).write_text(html)
    print(f"Interactive heat-map written to {args.out}\nOpen the file in any browser – no server required.")


if __name__ == "__main__":
    main()
