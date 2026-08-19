"""HTML Canvas visualization for society simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


class Renderer:
    """Generates HTML canvas visualizations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_network(
        self,
        society_snapshot: dict,
        events: List[dict],
        filename: str = "network.html",
    ) -> Path:
        """Render social network visualization."""
        agents = society_snapshot.get("agents", {})
        proposals = society_snapshot.get("proposals", {})

        # Compute node positions (circular layout)
        n_agents = len(agents)
        positions = {}
        for i, agent_id in enumerate(agents.keys()):
            angle = 2 * np.pi * i / n_agents
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            positions[agent_id] = (float(x), float(y))

        # Build edges
        edges = []
        seen = set()
        for agent_id, agent_data in agents.items():
            for known_id in agent_data.get("known_agents", []):
                key = tuple(sorted([int(agent_id), int(known_id)]))
                if key not in seen:
                    seen.add(key)
                    if str(known_id) in agents:
                        edges.append({
                            "source": int(agent_id),
                            "target": int(known_id),
                            "trust": agent_data.get("trust", {}).get(str(known_id),
                                      agent_data.get("trust", {}).get(known_id, 0.5)),
                        })

        # Build proposal data
        proposal_data = []
        for pid, pdata in proposals.items():
            proposal_data.append({
                "id": int(pid),
                "content": pdata.get("content", "")[:50],
                "support": pdata.get("support_count", 0),
                "oppose": pdata.get("oppose_count", 0),
                "creator": pdata.get("creator", -1),
            })

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Society Network Visualization</title>
    <style>
        body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; }}
        #container {{ display: flex; height: 100vh; }}
        #canvas {{ flex: 1; }}
        #sidebar {{ width: 300px; padding: 20px; overflow-y: auto; background: #16213e; }}
        h2 {{ color: #e94560; margin-top: 0; }}
        .proposal {{ background: #0f3460; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .stats {{ color: #aaa; font-size: 12px; }}
    </style>
</head>
<body>
    <div id="container">
        <canvas id="canvas"></canvas>
        <div id="sidebar">
            <h2>Society Network</h2>
            <div class="stats">
                Agents: {n_agents} | Edges: {len(edges)} | Proposals: {len(proposal_data)}
            </div>
            <h3>Proposals</h3>
            <div id="proposals"></div>
        </div>
    </div>
    <script>
        const positions = {json.dumps(positions)};
        const edges = {json.dumps(edges)};
        const proposals = {json.dumps(proposal_data)};

        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        function resize() {{
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        }}
        resize();
        window.addEventListener('resize', resize);

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw edges
            for (const edge of edges) {{
                ctx.strokeStyle = `rgba(255,255,255,${{0.05 + edge.trust * 0.3}})`;
                ctx.lineWidth = 0.5 + edge.trust * 2;
                const sx = positions[edge.source][0] * canvas.width;
                const sy = positions[edge.source][1] * canvas.height;
                const tx = positions[edge.target][0] * canvas.width;
                const ty = positions[edge.target][1] * canvas.height;
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(tx, ty);
                ctx.stroke();
            }}

            // Draw nodes
            for (const [id, pos] of Object.entries(positions)) {{
                const x = pos[0] * canvas.width;
                const y = pos[1] * canvas.height;
                const degree = edges.filter(e => e.source == id || e.target == id).length;
                const r = 3 + Math.sqrt(degree) * 0.7;
                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fillStyle = '#e94560';
                ctx.fill();
            }}
        }}
        draw();

        // Show proposals
        const div = document.getElementById('proposals');
        for (const p of proposals) {{
            const el = document.createElement('div');
            el.className = 'proposal';
            el.innerHTML = `<b>#${{p.id}}</b>: ${{p.content}}<br>
                <span style="color:#4ecca3">Support: ${{p.support}}</span> |
                <span style="color:#e94560">Oppose: ${{p.oppose}}</span>`;
            div.appendChild(el);
        }}
    </script>
</body>
</html>"""
        path = self.output_dir / filename
        path.write_text(html)
        return path

    def render_timeline(
        self,
        events: List[dict],
        filename: str = "timeline.html",
    ) -> Path:
        """Render event timeline visualization."""
        # Keep task boundaries; round numbers restart for every task.
        rounds = {}
        for event in events:
            key = (event.get("task", 0), event.get("round", 0))
            if key not in rounds:
                rounds[key] = []
            rounds[key].append(event)

        timeline_data = []
        for task_id, round_id in sorted(rounds.keys()):
            type_counts = {}
            for e in rounds[(task_id, round_id)]:
                t = e.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            timeline_data.append({"task": task_id, "round": round_id, "counts": type_counts})

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Event Timeline</title>
    <style>
        body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; padding: 20px; }}
        .round {{ display: flex; align-items: center; margin: 2px 0; }}
        .round-num {{ width: 40px; color: #888; }}
        .bar {{ height: 16px; margin: 1px 0; border-radius: 2px; }}
        .propose {{ background: #e94560; }}
        .support {{ background: #4ecca3; }}
        .oppose {{ background: #ff6b6b; }}
        .contact {{ background: #3498db; }}
        .follow {{ background: #9b59b6; }}
        .other {{ background: #555; }}
    </style>
</head>
<body>
    <h2>Event Timeline</h2>
    <div id="timeline"></div>
    <script>
        const data = {json.dumps(timeline_data)};
        const container = document.getElementById('timeline');
        const types = ['propose', 'support', 'oppose', 'contact', 'follow', 'other'];

        for (const round of data) {{
            const div = document.createElement('div');
            div.className = 'round';
            div.innerHTML = `<span class="round-num" title="Task ${{round.task}}">${{round.task}}:${{round.round}}</span>`;

            for (const t of types) {{
                const count = round.counts[t] || 0;
                if (count > 0) {{
                    const bar = document.createElement('div');
                    bar.className = `bar ${{t}}`;
                    bar.style.width = (count * 8) + 'px';
                    bar.title = `${{t}}: ${{count}}`;
                    div.appendChild(bar);
                }}
            }}
            container.appendChild(div);
        }}
    </script>
</body>
</html>"""
        path = self.output_dir / filename
        path.write_text(html)
        return path

    def render_metrics(
        self,
        rounds_data: List[dict],
        filename: str = "metrics.html",
    ) -> Path:
        """Render metrics dashboard."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Metrics Dashboard</title>
    <style>
        body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; padding: 20px; }}
        .chart {{ margin: 20px 0; }}
        canvas {{ background: #16213e; border-radius: 5px; }}
        h2 {{ color: #e94560; }}
    </style>
</head>
<body>
    <h2>Society Metrics</h2>
    <div class="chart">
        <h3>Proposals & Trust Over Time</h3>
        <canvas id="chart" width="800" height="300"></canvas>
    </div>
    <script>
        const data = {json.dumps(rounds_data)};
        const canvas = document.getElementById('chart');
        const ctx = canvas.getContext('2d');

        function draw() {{
            ctx.clearRect(0, 0, 800, 300);
            const maxProposals = Math.max(...data.map(d => d.alive_proposals), 1);
            const maxTrust = Math.max(...data.map(d => d.average_trust), 1);

            // Draw proposals line
            ctx.strokeStyle = '#e94560';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < data.length; i++) {{
                const x = (i / (data.length - 1)) * 780 + 10;
                const y = 290 - (data[i].alive_proposals / maxProposals) * 270;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();

            // Draw trust line
            ctx.strokeStyle = '#4ecca3';
            ctx.beginPath();
            for (let i = 0; i < data.length; i++) {{
                const x = (i / (data.length - 1)) * 780 + 10;
                const y = 290 - (data[i].average_trust / maxTrust) * 270;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();
        }}
        draw();
    </script>
</body>
</html>"""
        path = self.output_dir / filename
        path.write_text(html)
        return path

    def render_proposals(
        self,
        society_snapshot: dict,
        events: Optional[List[dict]] = None,
        task_summaries: Optional[List[dict]] = None,
        filename: str = "proposals.html",
    ) -> Path:
        """Render proposal birth, mutation, merge, death, and adoption state."""
        proposals = list(society_snapshot.get("proposals", {}).values())
        rows = []
        for proposal in sorted(proposals, key=lambda item: item["id"]):
            lineage = f"parent={proposal.get('parent')} merged={proposal.get('merged_from', [])}"
            rows.append(
                "<tr>"
                f"<td>{proposal['id']}</td><td>{proposal.get('created_round', 0)}</td>"
                f"<td>{proposal.get('option_id')}</td><td>{proposal.get('creator')}</td>"
                f"<td>{lineage}</td><td>{proposal.get('support_count', 0)}</td>"
                f"<td>{'alive' if proposal.get('alive', True) else 'dead'}</td>"
                "</tr>"
            )
        history_rows = []
        for event in events or []:
            if event.get("type") in {"propose", "modify", "merge"}:
                history_rows.append(
                    "<tr>"
                    f"<td>{event.get('task')}</td><td>{event.get('round')}</td>"
                    f"<td>{event.get('type')}</td><td>{event.get('proposal_id', event.get('proposal_a'))}</td>"
                    f"<td>{event.get('new_proposal_id', '')}</td>"
                    "</tr>"
                )
        for summary in task_summaries or []:
            history_rows.append(
                f"<tr><td>{summary['task']}</td><td>{summary['decision_time']}</td>"
                f"<td>adopt</td><td>{summary['winner']}</td><td></td></tr>"
            )

        html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Proposal Evolution</title><style>
body { background:#1a1a2e; color:#eee; font-family:monospace; padding:20px; }
table { border-collapse:collapse; width:100%; } th,td { border:1px solid #405070; padding:7px; }
th { background:#0f3460; } tr:nth-child(even) { background:#16213e; }
</style></head><body><h2>Proposal Evolution: Final Task State</h2>
<table><thead><tr><th>ID</th><th>Birth</th><th>Option</th><th>Creator</th>
<th>Lineage</th><th>Support</th><th>State</th></tr></thead><tbody>""" + "".join(rows) + \
            "</tbody></table><h2>All Task Events</h2><table><thead><tr><th>Task</th>" \
            "<th>Round</th><th>Event</th><th>Source proposal</th><th>New proposal</th>" \
            "</tr></thead><tbody>" + "".join(history_rows) + "</tbody></table></body></html>"
        path = self.output_dir / filename
        path.write_text(html)
        return path
