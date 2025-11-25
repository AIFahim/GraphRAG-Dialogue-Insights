#!/usr/bin/env python3
"""
Vision-based Graph Structure Extractor
Extracts entities, relationships, and reasoning paths from graph visualizations
Uses HuggingFace Inference API (like its_llm_pipeline.py)
"""

import json
import os
import base64
from typing import Dict, List
from collections import Counter, defaultdict, deque
from huggingface_hub import InferenceClient
from PIL import Image


# Load from environment (same as its_llm_pipeline.py)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables only.")

HF_TOKEN = os.environ.get("HF_TOKEN")


class VisionGraphExtractor:
    """Extract graph structure from visualization images using HuggingFace API"""

    def __init__(self, model_name: str = "zai-org/GLM-4.5V"):
        """
        Initialize with HuggingFace Inference API

        Vision models available on HF Inference API:
        - zai-org/GLM-4.5V (recommended - cutting-edge vision-language model)

        Note: Most vision models (Llama-Vision, Qwen2-VL) are NOT available via API
        """
        print(f"Initializing HuggingFace Inference API")
        print(f"Vision model: {model_name}")

        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not set! Set it in .env or environment")

        self.client = InferenceClient(token=HF_TOKEN)
        self.model = model_name
        print("✅ API client ready (no download needed!)")

    def _call_vision_api(self, image_path: str, prompt: str, max_tokens: int = 512) -> str:
        """
        Common method for all vision API calls

        Args:
            image_path: Path to graph visualization image
            prompt: Text prompt for the vision model
            max_tokens: Maximum tokens in response

        Returns:
            Raw response text from vision model

        Raises:
            Exception: If API call fails (no silent fallbacks)
        """
        # Load image and convert to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Call HF Inference API with image
        response = self.client.chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
            }],
            model=self.model,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()

    def extract_entities(self, image_path: str) -> List[str]:
        """
        Step 1: Extract all entity names from visualization
        """
        prompt = """Look at this graph visualization.
Extract ALL node labels/names you can see.

Return ONLY a JSON list of entity names.
Example: ["MRM-488", "MRM-615", "WebUI", "Martin"]

JSON list:"""

        # Call vision API (raises exception on error - no fallback)
        result = self._call_vision_api(image_path, prompt, max_tokens=512)

        # Clean special tokens from vision model response
        result = result.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '').strip()

        # Parse JSON response
        try:
            entities = json.loads(result)
            if isinstance(entities, list):
                return entities
        except json.JSONDecodeError:
            # Try to extract from malformed JSON
            import re
            entities = re.findall(r'"([^"]+)"', result)
            if entities:
                return entities

        # If all parsing fails, raise error (no silent fallback!)
        raise ValueError(f"Failed to parse entities from response: {result[:100]}...")

    def extract_relationships(self, image_path: str, entities: List[str]) -> List[Dict]:
        """
        Step 2: Extract relationships between entities
        """
        prompt = f"""Graph nodes: {entities}

Look at the arrows/edges in this graph.
For each edge, identify:
- Source node
- Target node
- Relationship type (label on the edge)

Return as JSON array:
[
  {{"source": "MRM-488", "target": "MRM-615", "type": "RELATES_TO"}},
  {{"source": "MRM-488", "target": "MRM-487", "type": "RELATES_TO"}}
]

JSON array:"""

        # Call vision API (raises exception on error)
        result = self._call_vision_api(image_path, prompt, max_tokens=1024)

        # Clean special tokens from vision model response
        result = result.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '').strip()

        # Parse JSON response
        try:
            relationships = json.loads(result)
            if isinstance(relationships, list):
                return relationships
        except json.JSONDecodeError:
            pass

        # If parsing fails, raise error (no silent fallback!)
        raise ValueError(f"Failed to parse relationships from response: {result[:100]}...")

    def analyze_topology(self, image_path: str, entities: List[str], relationships: List[Dict]) -> Dict:
        """
        Step 3: Analyze graph topology and structure
        """
        prompt = f"""Analyze this graph structure.

Nodes: {entities}
Edges: {len(relationships)} connections

Identify:
1. Central/hub nodes (nodes with many connections) - can be MULTIPLE
2. Graph topology type (star, multi-hub, clustered, tree, chain)
3. Any visual clusters or groupings

Return as JSON:
{{
  "central_nodes": ["MRM-488", "MRM-500"],
  "topology_type": "multi-hub",
  "has_clusters": true
}}

JSON:"""

        # Call vision API (raises exception on error)
        result = self._call_vision_api(image_path, prompt, max_tokens=512)

        # Clean special tokens from vision model response
        result = result.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '').strip()

        # Parse JSON response
        try:
            topology = json.loads(result)
            return topology
        except json.JSONDecodeError:
            pass

        # If parsing fails, raise error (no silent fallback!)
        raise ValueError(f"Failed to parse topology from response: {result[:100]}...")

    def extract_complete_structure(self, image_path: str) -> Dict:
        """
        Complete extraction: entities + relationships + topology
        """
        print(f"\n{'='*60}")
        print(f"Extracting from: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # Step 1: Extract entities
        print("\n[1/3] Extracting entities via HF API...")
        entities = self.extract_entities(image_path)
        print(f"  Found {len(entities)} entities: {entities[:5]}...")

        # Step 2: Extract relationships
        print("\n[2/3] Extracting relationships via HF API...")
        relationships = self.extract_relationships(image_path, entities)
        print(f"  Found {len(relationships)} relationships")

        # Step 3: Analyze topology
        print("\n[3/3] Analyzing topology via HF API...")
        topology = self.analyze_topology(image_path, entities, relationships)
        print(f"  Topology: {topology.get('topology_type', 'unknown')}")
        print(f"  Central nodes: {topology.get('central_nodes', [])}")

        return {
            'entities': entities,
            'relationships': relationships,
            'topology': topology,
            'image_path': image_path
        }

class ReasoningPathBuilder:
    """Build reasoning paths from extracted graph structure"""

    @staticmethod
    def build_reasoning_path(
        graph_structure: Dict,
        query: str,
        cypher: str = None,
        response: str = None
    ) -> Dict:
        """
        Build complete reasoning path from extracted structure
        """
        entities = graph_structure['entities']
        relationships = graph_structure['relationships']
        topology = graph_structure['topology']

        # Calculate metrics
        depth = ReasoningPathBuilder._calculate_depth(
            relationships,
            topology.get('central_nodes', [entities[0]] if entities else [])
        )

        clusters = ReasoningPathBuilder._detect_clusters(
            relationships,
            topology.get('central_nodes', [])
        )

        reasoning_path = {
            # Query context
            'query': query,
            'query_type': ReasoningPathBuilder._classify_query_type(query),
            'cypher_query': cypher,
            'response': response,

            # Graph structure (from vision extraction)
            'central_nodes': topology.get('central_nodes', []),
            'entities': entities,
            'relationships': relationships,

            # Topology
            'topology': {
                'type': topology.get('topology_type', 'unknown'),
                'has_multiple_hubs': len(topology.get('central_nodes', [])) > 1,
                'has_clusters': topology.get('has_clusters', False),
                'clusters': clusters
            },

            # Pattern metrics
            'pattern': {
                'node_count': len(entities),
                'edge_count': len(relationships),
                'depth': depth,
                'density': len(relationships) / max(len(entities), 1)
            },

            # Visual features
            'visual_features': topology.get('visual_features', {}),

            # Image reference
            'image_path': graph_structure['image_path']
        }

        return reasoning_path

    @staticmethod
    def _calculate_depth(relationships: List[Dict], start_nodes: List[str]) -> int:
        """Calculate maximum depth from central nodes"""
        if not relationships or not start_nodes:
            return 0

        # Build adjacency
        adjacency = defaultdict(set)
        for rel in relationships:
            adjacency[rel.get('source', '')].add(rel.get('target', ''))

        # BFS from all start nodes
        max_depth = 0
        for start in start_nodes:
            visited = {start}
            queue = deque([(start, 0)])

            while queue:
                node, depth = queue.popleft()
                max_depth = max(max_depth, depth)

                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return max_depth

    @staticmethod
    def _detect_clusters(relationships: List[Dict], central_nodes: List[str]) -> Dict:
        """Detect node clusters around central nodes"""
        if not central_nodes:
            return {}

        # Build adjacency
        adjacency = defaultdict(set)
        all_nodes = set()
        for rel in relationships:
            src = rel.get('source', '')
            tgt = rel.get('target', '')
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)
            all_nodes.add(src)
            all_nodes.add(tgt)

        # Assign nodes to clusters
        clusters = {f"cluster_{i}": set([node]) for i, node in enumerate(central_nodes)}

        for node in all_nodes:
            if node not in central_nodes:
                # Find nearest central
                nearest = ReasoningPathBuilder._find_nearest_central(
                    node, central_nodes, adjacency
                )
                cluster_idx = central_nodes.index(nearest)
                clusters[f"cluster_{cluster_idx}"].add(node)

        # Convert sets to lists
        return {k: list(v) for k, v in clusters.items()}

    @staticmethod
    def _find_nearest_central(node: str, central_nodes: List[str], adjacency: Dict) -> str:
        """BFS to find nearest central node"""
        visited = {node}
        queue = deque([node])

        while queue:
            current = queue.popleft()

            if current in central_nodes:
                return current

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return central_nodes[0] if central_nodes else node

    @staticmethod
    def _classify_query_type(query: str) -> str:
        """Classify query type from text"""
        q = query.lower()
        if 'relate' in q or 'relationship' in q:
            return 'relationship'
        elif 'depend' in q or 'dependency' in q:
            return 'dependency'
        elif 'assign' in q or 'who' in q:
            return 'assignment'
        elif 'bug' in q or 'issue' in q:
            return 'issue'
        return 'general'


if __name__ == "__main__":
    # Test extraction
    extractor = VisionGraphExtractor()

    test_image = "/home/aifahim/PycharmProjects/GraphRAG-Dialogue-Insights/data/seoss_extracted/results/what_issues_does_mrm_488_relate_to_.png"

    if os.path.exists(test_image):
        structure = extractor.extract_complete_structure(test_image)

        print("\n" + "="*60)
        print("EXTRACTED STRUCTURE")
        print("="*60)
        print(json.dumps(structure, indent=2))

        # Build reasoning path
        reasoning_path = ReasoningPathBuilder.build_reasoning_path(
            structure,
            query="What issues does MRM-488 relate to?"
        )

        print("\n" + "="*60)
        print("REASONING PATH")
        print("="*60)
        print(json.dumps(reasoning_path, indent=2))
