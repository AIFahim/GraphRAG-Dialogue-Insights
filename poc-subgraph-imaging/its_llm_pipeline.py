#!/usr/bin/env python3
"""
Simple LLM-Based Pipeline
Everything controlled by LLM - no manual logic

Flow:
1. User Query → LLM generates Cypher (full graph structure)
2. Execute Cypher on Neo4j
3. LLM decides: What to visualize + Suggestions
4. Simple matplotlib draws graph
"""
import os
import json
from neo4j import GraphDatabase
from huggingface_hub import InferenceClient
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables only.")

# Config
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required: Set in .env or environment
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "seoss_extracted", "results")

# Schema for LLM
SCHEMA = """
SEOSS Issue Tracking System Schema:

NODES:
- Issue: {id, type, summary, status, priority}
- Person: {name}
- Component: {name}

RELATIONSHIPS:
- RELATES_TO: Issue relates to Issue
- DEPENDS_UPON: Issue depends on Issue
- DUPLICATES: Issue duplicates Issue
- ASSIGNED_TO: Issue assigned to Person
- AFFECTS: Issue affects Component

EXAMPLES:

1. Count bugs (NO relationships):
MATCH (i:Issue {type: 'Bug'})
RETURN count(i) as total_bugs

2. List bugs (NO relationships):
MATCH (i:Issue {type: 'Bug'})
RETURN i.id as issue_id, i.summary as summary, i.status as status
LIMIT 10

3. Related issues (WITH relationship):
MATCH (i:Issue {id: 'MRM-488'})-[r:RELATES_TO]-(related:Issue)
RETURN i.id as source, type(r) as rel, related.id as target LIMIT 10

4. Dependencies (WITH relationship):
MATCH (i1:Issue)-[r:DEPENDS_UPON]->(i2:Issue)
RETURN i1.id as source, type(r) as rel, i2.id as target LIMIT 10

5. Developer assignments (WITH relationship):
MATCH (i:Issue)-[r:ASSIGNED_TO]->(p:Person)
RETURN i.id as source, type(r) as rel, p.name as target LIMIT 10

IMPORTANT: If query is just counting or listing nodes WITHOUT relationships,
do NOT use type(r) - only use it when you have a MATCH with relationship variable [r].
"""


class SimpleLLMPipeline:
    """Pure LLM-driven pipeline"""

    def __init__(self):
        self.client = InferenceClient(token=HF_TOKEN)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def llm_generate_cypher(self, user_query: str) -> str:
        """LLM generates Cypher with FULL graph structure"""

        prompt = f"""{SCHEMA}

User question: {user_query}

Determine if this question asks about:
A) RELATIONSHIPS between entities → Use MATCH (a)-[r]->(b) pattern
B) Just LISTING or COUNTING entities → Use MATCH (i:Issue) without relationships

Then generate appropriate Cypher:

TYPE A - Relationship queries (e.g., "what relates to X", "who is assigned", "dependencies"):
MATCH (i:Issue)-[r:RELATES_TO]->(j:Issue)
RETURN i.id as source, type(r) as rel, j.id as target LIMIT 10

TYPE B - Listing/Counting (e.g., "find bugs", "how many", "list issues"):
MATCH (i:Issue {{type: 'Bug', status: 'Closed'}})
RETURN i.id, i.summary, i.priority, i.status LIMIT 10

CRITICAL: If you use MATCH without [r], DO NOT use type(r) in RETURN!

Return ONLY the Cypher query, no explanation."""

        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=512,
            temperature=0.1
        )

        cypher = response.choices[0].message.content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        return cypher

    def execute_cypher(self, cypher: str) -> list:
        """Execute on Neo4j"""
        try:
            with self.driver.session() as session:
                return [dict(record) for record in session.run(cypher)]
        except Exception as e:
            # Return error info for retry
            raise Exception(f"Cypher error: {str(e)}")

    def fetch_related_edges(self, results: list) -> dict:
        """Fetch 1-hop neighboring edges from direct results"""

        # Extract entity IDs from results
        entity_ids = set()
        for record in results:
            for value in record.values():
                if isinstance(value, str) and value.startswith('MRM-'):
                    entity_ids.add(value)

        if not entity_ids:
            return {"edges": [], "nodes": 0, "edge_count": 0}

        # Fetch neighboring edges
        context_query = """
        UNWIND $ids AS entity_id
        MATCH (n:Issue {id: entity_id})-[r]-(neighbor)
        WHERE neighbor.id IS NOT NULL
        RETURN n.id as source, type(r) as rel,
               neighbor.id as target, labels(neighbor)[0] as target_type
        LIMIT 50
        """

        try:
            with self.driver.session() as session:
                context_results = session.run(context_query, ids=list(entity_ids)[:10])
                context_edges = [dict(record) for record in context_results]

                # Count unique edges and nodes
                unique_edges = len(context_edges)
                unique_nodes = len(set([e.get('source') for e in context_edges] +
                                      [e.get('target') for e in context_edges]))

                return {
                    "edges": context_edges,
                    "nodes": unique_nodes,
                    "edge_count": unique_edges
                }
        except Exception as e:
            print(f"  Context retrieval error: {e}")
            return {"edges": [], "nodes": 0, "edge_count": 0}

#     def llm_analyze_results(self, user_query: str, results: list, context: dict = None) -> dict:
#         """LLM analyzes results with context and decides what to visualize + suggestions + natural response"""
#
#         results_json = json.dumps(results[:15], indent=2, default=str)
#
#         # Add context information if available
#         context_info = ""
#         if context and context.get('edges'):
#             context_json = json.dumps(context['edges'][:20], indent=2, default=str)
#             context_info = f"""
#
# Additional Context (1-hop neighboring edges):
# {context_json}
#
# This context shows related edges that weren't in the main query but are connected to the results.
# """
#
#         prompt = f"""User asked: {user_query}
#
# Main query results:
# {results_json}
# {context_info}
#
# Analyze these results (including context if provided) and return JSON with:
# 1. "response": Natural language answer (2-3 sentences, explain what you found INCLUDING insights from context)
# 2. "nodes": list of ALL node IDs/names to show in graph (from main results + useful context)
# 3. "edges": list of {{"source": "...", "target": "...", "label": "...", "is_context": true/false}}
#    - is_context=false for main query edges
#    - is_context=true for context edges
# 4. "suggestions": 5 natural language follow-up questions (based on available relationships in context)
#
# Return JSON only:
# {{
#     "response": "I found 2 related issues for MRM-488. Based on context, these bugs also affect the Web Interface component and are assigned to Martin Stockhammer.",
#     "nodes": ["MRM-488", "MRM-615", "MRM-487", "Web Interface", "Martin Stockhammer"],
#     "edges": [
#         {{"source": "MRM-488", "target": "MRM-615", "label": "RELATES_TO", "is_context": false}},
#         {{"source": "MRM-488", "target": "Web Interface", "label": "AFFECTS", "is_context": true}}
#     ],
#     "suggestions": ["Who is assigned to MRM-615?", "What other components are affected?", ...]
# }}"""
#
#         response = self.client.chat_completion(
#             messages=[{"role": "user", "content": prompt}],
#             model=MODEL,
#             max_tokens=1536,
#             temperature=0.3
#         )
#
#         content = response.choices[0].message.content.strip()
#
#         # Parse JSON
#         if "```json" in content:
#             content = content.split("```json")[1].split("```")[0].strip()
#         elif "```" in content:
#             content = content.split("```")[1].split("```")[0].strip()
#
#         try:
#             return json.loads(content)
#         except json.JSONDecodeError as e:
#             print(f"  JSON parse error: {e}")
#             print("  Extracting basic info from malformed response...")
#
#             # Fallback: Extract nodes manually
#             nodes = []
#             edges = []
#
#             # Extract Issue IDs from results
#             for record in results[:10]:
#                 for val in record.values():
#                     if isinstance(val, str) and val.startswith('MRM-'):
#                         if val not in nodes:
#                             nodes.append(val)
#
#             # Try to extract edges from context
#             if context and context.get('edges'):
#                 for ctx_edge in context['edges'][:10]:
#                     edges.append({
#                         "source": ctx_edge.get('source'),
#                         "target": ctx_edge.get('target'),
#                         "label": ctx_edge.get('rel'),
#                         "is_context": True
#                     })
#
#             return {
#                 "response": f"Found {len(results)} results. (Note: Full analysis unavailable due to formatting issue)",
#                 "nodes": nodes[:15],
#                 "edges": edges[:15],
#                 "suggestions": [
#                     "Show more details about these issues",
#                     "Who is assigned to these?",
#                     "What components are affected?",
#                     "Show related bugs",
#                     "Check dependencies"
#                 ]
#             }

    def draw_graph(self, graph_spec: dict, filename: str) -> str:
        """Simple matplotlib drawing from LLM specification"""

        G = nx.DiGraph()

        # Add nodes
        for node in graph_spec.get("nodes", []):
            G.add_node(node)

        # Add edges
        for edge in graph_spec.get("edges", []):
            G.add_edge(
                edge["source"],
                edge["target"],
                label=edge.get("label", "")
            )

        # Draw
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Color nodes
        node_colors = ['#FF6B6B' if str(n).startswith('MRM-') else '#4ECDC4'
                       for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=3000, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

        # Draw edges with different styles for direct vs context
        direct_edges = [(e["source"], e["target"]) for e in graph_spec.get("edges", [])
                        if not e.get("is_context", False)]
        context_edges = [(e["source"], e["target"]) for e in graph_spec.get("edges", [])
                         if e.get("is_context", False)]

        # Direct edges - solid, darker
        if direct_edges:
            nx.draw_networkx_edges(G, pos, edgelist=direct_edges, edge_color='#333',
                                   arrows=True, arrowsize=20, width=2.5, style='solid')

        # Context edges - dashed, lighter
        if context_edges:
            nx.draw_networkx_edges(G, pos, edgelist=context_edges, edge_color='#999',
                                   arrows=True, arrowsize=15, width=1.5, style='dashed')

        # Edge labels
        edge_labels = {(e["source"], e["target"]): e.get("label", "")
                       for e in graph_spec.get("edges", [])}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        # Add legend if we have both types of edges
        if direct_edges and context_edges:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#333', linewidth=2.5, label='Direct (from query)'),
                Line2D([0], [0], color='#999', linewidth=1.5, linestyle='--', label='Context (related)')
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.title(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Save
        png_path = os.path.join(RESULTS_DIR, f"{filename}.png")
        plt.savefig(png_path, dpi=200, bbox_inches='tight')
        plt.close()

        return png_path

    def generate_plantuml(self, graph_spec: dict, user_query: str) -> str:
        """Generate PlantUML code from graph specification"""

        prompt = f"""Generate PlantUML code to visualize this graph.

User question: {user_query}

Nodes: {graph_spec.get('nodes', [])}
Edges: {graph_spec.get('edges', [])}

Create a clear PlantUML diagram:
- Use object diagram or component diagram style
- Color code nodes: #FF6B6B for MRM- issues, #4ECDC4 for others
- Show all relationships with labels
- Keep it simple and readable

Return ONLY PlantUML code starting with @startuml and ending with @enduml.
No explanations."""

        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=1024,
            temperature=0.2
        )

        puml_code = response.choices[0].message.content.strip()

        # Clean up
        if "```plantuml" in puml_code:
            puml_code = puml_code.split("```plantuml")[1].split("```")[0].strip()
        elif "```" in puml_code:
            puml_code = puml_code.split("```")[1].split("```")[0].strip()

        return puml_code

    def render_plantuml(self, puml_code: str, filename: str) -> tuple:
        """Save PlantUML and try to render to PNG"""

        # Save .puml file
        puml_path = os.path.join(RESULTS_DIR, f"{filename}.puml")
        with open(puml_path, 'w') as f:
            f.write(puml_code)

        # Try to render PNG using PlantUML online server
        png_path = os.path.join(RESULTS_DIR, f"{filename}_plantuml.png")

        try:
            import urllib.request
            import zlib

            # PlantUML encoding
            def encode_plantuml(text):
                zlibbed = zlib.compress(text.encode('utf-8'))[2:-4]
                b64chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
                result = ""
                for i in range(0, len(zlibbed), 3):
                    if i + 2 < len(zlibbed):
                        b1, b2, b3 = zlibbed[i], zlibbed[i + 1], zlibbed[i + 2]
                        result += b64chars[b1 >> 2]
                        result += b64chars[((b1 & 0x3) << 4) | (b2 >> 4)]
                        result += b64chars[((b2 & 0xF) << 2) | (b3 >> 6)]
                        result += b64chars[b3 & 0x3F]
                    elif i + 1 < len(zlibbed):
                        b1, b2 = zlibbed[i], zlibbed[i + 1]
                        result += b64chars[b1 >> 2]
                        result += b64chars[((b1 & 0x3) << 4) | (b2 >> 4)]
                        result += b64chars[(b2 & 0xF) << 2]
                    else:
                        b1 = zlibbed[i]
                        result += b64chars[b1 >> 2]
                        result += b64chars[(b1 & 0x3) << 4]
                return result

            encoded = encode_plantuml(puml_code)
            url = f"http://www.plantuml.com/plantuml/png/{encoded}"

            urllib.request.urlretrieve(url, png_path)

            if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                return puml_path, png_path

        except Exception as e:
            print(f"  PlantUML render warning: {e}")

        return puml_path, None

    def run(self, user_query: str) -> dict:
        """Run complete pipeline"""

        print(f"\n{'='*70}")
        print(f"QUERY: {user_query}")
        print(f"{'='*70}")

        # Step 1: LLM generates Cypher
        print("\n[1/5] LLM: Generating Cypher query...")
        cypher = self.llm_generate_cypher(user_query)
        print(f"  Cypher: {cypher[:80]}...")

        # Step 2: Execute (with retry on error)
        print("\n[2/5] Neo4j: Executing query...")
        try:
            results = self.execute_cypher(cypher)
            print(f"  Results: {len(results)} records")
        except Exception as e:
            print(f"  Error: {e}")
            print("  Retrying with corrected query...")

            # Retry: Ask LLM to fix the error
            retry_prompt = f"""This Cypher query has a syntax error:

{cypher}

Error message: {str(e)}

The error says a variable is not defined. This happens when you RETURN variables that were never declared in MATCH.

RULES TO FIX:
1. If query is just LISTING or COUNTING nodes (no relationships):
   - DON'T use MATCH with -[r]->
   - DON'T return type(r), source, rel, target
   - DO use: MATCH (i:Issue) RETURN i.id, i.summary, i.status

2. If query involves RELATIONSHIPS:
   - DO use: MATCH (a)-[r:REL_TYPE]->(b)
   - DO return: a.id as source, type(r) as rel, b.id as target

Examples:
- List bugs: MATCH (i:Issue {{type: 'Bug'}}) RETURN i.id, i.priority, i.status LIMIT 10
- Count bugs: MATCH (i:Issue {{type: 'Bug'}}) RETURN count(i) as total
- Relationships: MATCH (i:Issue)-[r:RELATES_TO]->(j:Issue) RETURN i.id, type(r), j.id LIMIT 10

Return ONLY the corrected Cypher query that fixes the undefined variable error."""

            retry_response = self.client.chat_completion(
                messages=[{"role": "user", "content": retry_prompt}],
                model=MODEL,
                max_tokens=512,
                temperature=0.1
            )

            cypher = retry_response.choices[0].message.content.strip()
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            print(f"  Retrying with: {cypher[:80]}...")

            try:
                results = self.execute_cypher(cypher)
                print(f"  Results: {len(results)} records")
            except Exception as retry_error:
                print(f"  Retry also failed: {retry_error}")
                # Return empty result
                return {
                    "query": user_query,
                    "cypher": cypher,
                    "results": [],
                    "response": f"Query failed: {str(retry_error)}. Try rephrasing your question.",
                    "suggestions": ["Try a simpler query", "Ask about specific issues", "Show all bugs"],
                    "image": None,
                    "context": {"edges": [], "nodes": 0}
                }

        if not results:
            print("  No results found!")
            return {
                "query": user_query,
                "cypher": cypher,
                "results": [],
                "suggestions": ["Try a different query"],
                "image": None,
                "context": {"edges": [], "nodes": 0}
            }

        # Step 3: Fetch related edges (context)
        print("\n[3/6] Fetching related edges (1-hop context)...")
        context = self.fetch_related_edges(results)
        print(f"  Context: {context['edge_count']} edges, {context['nodes']} unique nodes")

        # Step 4: LLM analyzes with context
        print("\n[4/6] LLM: Analyzing results + context + generating suggestions...")
        analysis = self.llm_analyze_results(user_query, results, context)
        print(f"  Nodes to show: {len(analysis['nodes'])}")
        print(f"  Edges to show: {len(analysis['edges'])}")
        print(f"  Suggestions: {len(analysis['suggestions'])}")

        # Step 5: Draw matplotlib graph
        print("\n[5/6] Drawing network graph...")
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', user_query[:40]).lower()
        image_path = self.draw_graph(analysis, safe_name)
        print(f"  Network graph: {image_path}")

        # Step 6: Generate and render PlantUML
        print("\n[6/6] Generating PlantUML diagram...")
        puml_code = self.generate_plantuml(analysis, user_query)
        puml_file, puml_image = self.render_plantuml(puml_code, safe_name)
        if puml_image:
            print(f"  PlantUML image: {puml_image}")
        else:
            print(f"  PlantUML file: {puml_file} (PNG render failed)")

        return {
            "query": user_query,
            "cypher": cypher,
            "results": results,
            "context": context,
            "response": analysis.get("response", "Results retrieved successfully."),
            "nodes": analysis["nodes"],
            "edges": analysis["edges"],
            "suggestions": analysis["suggestions"],
            "image": image_path,
            "plantuml_code": puml_code,
            "plantuml_file": puml_file,
            "plantuml_image": puml_image
        }

    def display(self, result: dict):
        """Display results"""

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")

        print(f"\n💬 RESPONSE:")
        print(f"  {result.get('response', 'No response generated')}\n")

        print(f"📝 CYPHER:\n{result['cypher']}\n")

        print(f"📊 DATA ({len(result['results'])} records):")
        for i, r in enumerate(result['results'][:3], 1):
            print(f"  {i}. {r}")
        if len(result['results']) > 3:
            print(f"  ... +{len(result['results'])-3} more")

        if 'nodes' in result and 'edges' in result:
            print(f"\n🔗 GRAPH ({len(result['nodes'])} nodes, {len(result['edges'])} edges):")
            for edge in result['edges'][:5]:
                print(f"  {edge['source']} --[{edge['label']}]--> {edge['target']}")
            if len(result['edges']) > 5:
                print(f"  ... +{len(result['edges'])-5} more edges")

        print(f"\n💡 SUGGESTIONS:")
        for i, s in enumerate(result['suggestions'], 1):
            print(f"  {i}. {s}")

        print(f"\n🖼️  NETWORK GRAPH: {result.get('image', 'None')}")
        if result.get('plantuml_image'):
            print(f"📐 PLANTUML IMAGE: {result['plantuml_image']}")
        elif result.get('plantuml_file'):
            print(f"📐 PLANTUML FILE: {result['plantuml_file']}")

    def close(self):
        self.driver.close()


def interactive():
    """Interactive demo"""

    print("="*70)
    print("  SIMPLE LLM PIPELINE")
    print("  Everything decided by LLM - no manual logic")
    print("="*70)
    print("\nExamples:")
    print("  - What issues does MRM-488 relate to?")
    print("  - Which developer has most unresolved bugs?")
    print("  - Show bugs that block other issues")
    print("\nType 'quit' to exit\n")

    pipeline = SimpleLLMPipeline()

    try:
        while True:
            question = input("Your question: ").strip()

            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                break

            result = pipeline.run(question)
            pipeline.display(result)

            # Open image
            if result['image'] and os.path.exists(result['image']):
                os.system(f'xdg-open "{result["image"]}" 2>/dev/null &')

    finally:
        pipeline.close()

    print("\nDone!")


if __name__ == "__main__":
    interactive()
