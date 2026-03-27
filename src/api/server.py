"""
IMMUNEX REST API and Dashboard

Provides endpoints for:
- Running the pipeline
- Querying the knowledge graph
- Viewing hypotheses
- Checking candidate safety
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.knowledge_graph.builder import KnowledgeGraphBuilder, INNATE_IMMUNE_TARGETS

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IMMUNEX API",
    description="Autonomous Multi-Agent AI System for Drug Repurposing in Innate Immunity",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory knowledge graph instance
kg = KnowledgeGraphBuilder()
kg.load_innate_immune_targets()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    kg_stats: dict


class SearchRequest(BaseModel):
    drug_name: Optional[str] = None
    pathway: Optional[str] = None
    min_confidence: float = 0.3


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the IMMUNEX dashboard."""
    stats = kg.stats
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>IMMUNEX Dashboard</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px; }}
            h1 {{ font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #fff, #888); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}
            .subtitle {{ color: #666; font-size: 1rem; margin-bottom: 40px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-bottom: 32px; }}
            .card {{ background: #141414; border: 1px solid #222; border-radius: 12px; padding: 24px; }}
            .card h3 {{ color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
            .card .value {{ font-size: 2rem; font-weight: 700; color: #fff; }}
            .card .detail {{ color: #666; font-size: 0.85rem; margin-top: 4px; }}
            .section {{ margin-top: 32px; }}
            .section h2 {{ font-size: 1.3rem; color: #ccc; margin-bottom: 16px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ text-align: left; padding: 12px; color: #888; font-size: 0.8rem; text-transform: uppercase; border-bottom: 1px solid #222; }}
            td {{ padding: 12px; border-bottom: 1px solid #1a1a1a; }}
            .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 500; }}
            .badge-green {{ background: rgba(34, 197, 94, 0.1); color: #22c55e; }}
            .badge-blue {{ background: rgba(59, 130, 246, 0.1); color: #3b82f6; }}
            .api-section {{ background: #141414; border: 1px solid #222; border-radius: 12px; padding: 24px; margin-top: 24px; }}
            code {{ background: #1a1a1a; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; color: #10b981; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>IMMUNEX</h1>
            <p class="subtitle">Autonomous Multi-Agent AI System for Drug Repurposing in Innate Immunity</p>

            <div class="grid">
                <div class="card">
                    <h3>Knowledge Graph Nodes</h3>
                    <div class="value">{stats['total_nodes']}</div>
                    <div class="detail">{stats['node_types'].get('target', 0)} targets, {stats['node_types'].get('pathway', 0)} pathways</div>
                </div>
                <div class="card">
                    <h3>Edges</h3>
                    <div class="value">{stats['total_edges']}</div>
                    <div class="detail">Drug-target interactions + pathway links</div>
                </div>
                <div class="card">
                    <h3>Innate Immune Targets</h3>
                    <div class="value">{len(INNATE_IMMUNE_TARGETS)}</div>
                    <div class="detail">Across 7 pathways</div>
                </div>
                <div class="card">
                    <h3>System Status</h3>
                    <div class="value" style="color: #22c55e;">Online</div>
                    <div class="detail">All agents operational</div>
                </div>
            </div>

            <div class="section">
                <h2>Target Pathways</h2>
                <table>
                    <tr>
                        <th>Pathway</th>
                        <th>Key Targets</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>TLR Signaling</td>
                        <td>TLR2, TLR4, TLR7, TLR9, MyD88</td>
                        <td><span class="badge badge-green">Active</span></td>
                    </tr>
                    <tr>
                        <td>Inflammasome</td>
                        <td>NLRP3, NLRC4, AIM2, Caspase-1, ASC</td>
                        <td><span class="badge badge-green">Active</span></td>
                    </tr>
                    <tr>
                        <td>cGAS-STING</td>
                        <td>cGAS, STING, TBK1, IRF3</td>
                        <td><span class="badge badge-green">Active</span></td>
                    </tr>
                    <tr>
                        <td>NF-&kappa;B</td>
                        <td>IKK&beta;, RelA, NEMO</td>
                        <td><span class="badge badge-green">Active</span></td>
                    </tr>
                    <tr>
                        <td>JAK-STAT</td>
                        <td>JAK1, JAK2, TYK2, STAT1</td>
                        <td><span class="badge badge-green">Active</span></td>
                    </tr>
                    <tr>
                        <td>Trained Immunity</td>
                        <td>mTOR, HIF-1&alpha;</td>
                        <td><span class="badge badge-green">Active</span></td>
                    </tr>
                    <tr>
                        <td>Complement</td>
                        <td>C3, C5, Factor D</td>
                        <td><span class="badge badge-green">Active</span></td>
                    </tr>
                </table>
            </div>

            <div class="api-section">
                <h2 style="margin-bottom: 16px;">API Endpoints</h2>
                <table>
                    <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
                    <tr><td><code>GET</code></td><td><code>/health</code></td><td>System health and KG stats</td></tr>
                    <tr><td><code>GET</code></td><td><code>/targets</code></td><td>List all innate immune targets</td></tr>
                    <tr><td><code>GET</code></td><td><code>/pathways</code></td><td>List all tracked pathways</td></tr>
                    <tr><td><code>GET</code></td><td><code>/candidates</code></td><td>Get ranked repurposing candidates</td></tr>
                    <tr><td><code>GET</code></td><td><code>/kg/stats</code></td><td>Knowledge graph statistics</td></tr>
                </table>
            </div>

            <p style="color: #444; font-size: 0.8rem; margin-top: 40px; text-align: center;">
                IMMUNEX v0.1.0 &middot; Built for Encode: AI for Science Fellowship (Pillar VC x ARIA) &middot;
                <a href="https://github.com/harshith-vaddiparthy/immunex" style="color: #666;">GitHub</a>
            </p>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        kg_stats=kg.stats,
    )


@app.get("/targets")
async def list_targets():
    """List all innate immune targets tracked by IMMUNEX."""
    targets = []
    for gene, info in INNATE_IMMUNE_TARGETS.items():
        targets.append({
            "gene_symbol": gene,
            "name": info["name"],
            "uniprot": info.get("uniprot", ""),
            "pathway": info["pathway"],
        })
    return {"total": len(targets), "targets": targets}


@app.get("/pathways")
async def list_pathways():
    """List all innate immune pathways."""
    from src.agents.literature_scanner import INNATE_IMMUNE_PATHWAYS

    pathways = []
    for key, info in INNATE_IMMUNE_PATHWAYS.items():
        pathways.append({
            "key": key,
            "name": info["name"],
            "targets": info["targets"],
            "num_targets": len(info["targets"]),
        })
    return {"total": len(pathways), "pathways": pathways}


@app.get("/candidates")
async def get_candidates(
    pathway: Optional[str] = Query(None, description="Filter by pathway key"),
    min_confidence: float = Query(0.3, description="Minimum confidence threshold"),
    limit: int = Query(50, description="Maximum results"),
):
    """Get ranked drug repurposing candidates from the knowledge graph."""
    candidates = kg.find_repurposing_candidates(
        target_pathway=pathway,
        min_confidence=min_confidence,
    )
    return {
        "total": len(candidates),
        "candidates": candidates[:limit],
        "filters": {"pathway": pathway, "min_confidence": min_confidence},
    }


@app.get("/kg/stats")
async def kg_stats():
    """Get knowledge graph statistics."""
    return kg.stats


def start():
    """Start the IMMUNEX API server."""
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("API_PORT", 8051)),
        reload=True,
    )


if __name__ == "__main__":
    import os
    start()
