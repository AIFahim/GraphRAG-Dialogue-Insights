#!/usr/bin/env python3
"""
Reasoning Template Database
Separate database for storing learned visualization patterns
NEVER mixes with ground truth data
"""

import json
import os
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from neo4j import GraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Storage configuration
TEMPLATE_STORAGE_DIR = "/home/aifahim/PycharmProjects/GraphRAG-Dialogue-Insights/data/reasoning_templates"
IMAGE_STORAGE = os.path.join(TEMPLATE_STORAGE_DIR, "images")
METADATA_STORAGE = os.path.join(TEMPLATE_STORAGE_DIR, "metadata")
KV_STORE_FILE = os.path.join(METADATA_STORAGE, "kv_store_templates.json")

# Create directories
os.makedirs(IMAGE_STORAGE, exist_ok=True)
os.makedirs(METADATA_STORAGE, exist_ok=True)

# Database configuration
# NOTE: Run a separate Neo4j instance on port 7688 OR use this same instance with clear labels
TEMPLATE_NEO4J_URI = "bolt://127.0.0.1:7687"  # Same instance for now
TEMPLATE_NEO4J_USER = "neo4j"
TEMPLATE_NEO4J_PASSWORD = "password"


class TemplateDatabase:
    """
    Separate database for reasoning templates
    Stores learned patterns WITHOUT corrupting ground truth
    """

    def __init__(self):
        # Connect to template database
        self.driver = GraphDatabase.driver(
            TEMPLATE_NEO4J_URI,
            auth=(TEMPLATE_NEO4J_USER, TEMPLATE_NEO4J_PASSWORD)
        )

        # Embedding model for similarity search
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")

        # Load key-value store
        self.kv_store = self._load_kv_store()

        # Initialize schema
        self._initialize_schema()

    def store_template(
        self,
        reasoning_path: Dict,
        image_path: str,
        plantuml_path: Optional[str] = None,
        user_helpful: bool = False,
        helpfulness_score: float = 0.0,
        user_feedback: str = None
    ) -> str:
        """
        Store reasoning template in separate database

        Returns: template_id
        """
        # Generate unique ID
        template_id = self._generate_template_id(reasoning_path)

        # Copy images to permanent storage
        stored_image = self._store_image(template_id, image_path, "network")
        stored_plantuml = None
        if plantuml_path and os.path.exists(plantuml_path):
            stored_plantuml = self._store_image(template_id, plantuml_path, "plantuml")

        # Generate embeddings
        query_embedding = self.embedding_model.encode(
            reasoning_path['query']
        ).tolist()

        # Store in Neo4j (with CLEAR label: ReasoningTemplate)
        with self.driver.session() as session:
            session.run("""
                CREATE (rt:ReasoningTemplate {
                    template_id: $template_id,

                    // Query info
                    query: $query,
                    query_type: $query_type,
                    query_embedding: $query_embedding,
                    cypher_query: $cypher,
                    response: $response,

                    // Extracted structure (from vision!)
                    central_nodes: $central_nodes,
                    entities: $entities,
                    relationships: $relationships,

                    // Topology
                    topology_type: $topology_type,
                    has_multiple_hubs: $has_multiple_hubs,
                    has_clusters: $has_clusters,
                    clusters: $clusters,

                    // Pattern metrics
                    node_count: $node_count,
                    edge_count: $edge_count,
                    depth: $depth,
                    density: $density,

                    // Visual features
                    visual_features: $visual_features,

                    // Image references
                    network_image_path: $network_image,
                    plantuml_image_path: $plantuml_image,

                    // Validation & learning
                    is_validated: $validated,
                    helpfulness_score: $score,
                    user_feedback: $feedback,
                    reuse_count: 0,
                    success_count: 0,
                    success_rate: 0.0,

                    // Metadata
                    created_at: datetime(),
                    last_used: null,
                    source: 'vision_extraction',
                    extraction_method: 'Qwen2-VL'
                })
            """, {
                'template_id': template_id,
                'query': reasoning_path['query'],
                'query_type': reasoning_path['query_type'],
                'query_embedding': query_embedding,
                'cypher': reasoning_path.get('cypher_query', ''),
                'response': reasoning_path.get('response', ''),
                'central_nodes': json.dumps(reasoning_path['central_nodes']),
                'entities': json.dumps(reasoning_path['entities']),
                'relationships': json.dumps(reasoning_path['relationships']),
                'topology_type': reasoning_path['topology']['type'],
                'has_multiple_hubs': reasoning_path['topology']['has_multiple_hubs'],
                'has_clusters': reasoning_path['topology']['has_clusters'],
                'clusters': json.dumps(reasoning_path['topology']['clusters']),
                'node_count': reasoning_path['pattern']['node_count'],
                'edge_count': reasoning_path['pattern']['edge_count'],
                'depth': reasoning_path['pattern']['depth'],
                'density': reasoning_path['pattern']['density'],
                'visual_features': json.dumps(reasoning_path.get('visual_features', {})),
                'network_image': stored_image,
                'plantuml_image': stored_plantuml,
                'validated': user_helpful,
                'score': helpfulness_score,
                'feedback': user_feedback
            })

        # Update KV store
        self._update_kv_store(template_id, {
            'query': reasoning_path['query'],
            'central_nodes': reasoning_path['central_nodes'],
            'topology_type': reasoning_path['topology']['type'],
            'image_path': stored_image,
            'timestamp': datetime.now().isoformat(),
            'validated': user_helpful,
            'score': helpfulness_score
        })

        print(f"\n✅ Stored reasoning template: {template_id}")
        print(f"   Query: {reasoning_path['query'][:60]}...")
        print(f"   Topology: {reasoning_path['topology']['type']}")
        print(f"   Central nodes: {reasoning_path['central_nodes']}")
        print(f"   Validated: {user_helpful}, Score: {helpfulness_score}")
        print(f"   Image: {stored_image}")

        return template_id

    def find_similar_templates(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.6,
        require_validated: bool = False
    ) -> List[Dict]:
        """
        Find similar reasoning templates using hybrid similarity

        Combines:
        1. Semantic similarity (query embeddings)
        2. Structural similarity (pattern matching)
        3. Historical success rate
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (rt:ReasoningTemplate)
                WHERE ($require_validated = false OR rt.is_validated = true)
                  AND rt.helpfulness_score >= $min_score
                RETURN rt.template_id as template_id,
                       rt.query as template_query,
                       rt.query_embedding as query_emb,
                       rt.query_type as query_type,
                       rt.central_nodes as central_nodes,
                       rt.entities as entities,
                       rt.relationships as relationships,
                       rt.topology_type as topology,
                       rt.network_image_path as image_path,
                       rt.plantuml_image_path as plantuml_path,
                       rt.response as response,
                       rt.cypher_query as cypher,
                       rt.helpfulness_score as score,
                       rt.reuse_count as reuse_count,
                       rt.success_rate as success_rate,
                       rt.depth as depth,
                       rt.node_count as node_count
            """, {
                'min_score': min_score,
                'require_validated': require_validated
            })

            templates = []
            for record in result:
                # Calculate semantic similarity
                template_emb = np.array(record['query_emb'])
                semantic_sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    template_emb.reshape(1, -1)
                )[0][0]

                # Calculate structural similarity
                structural_sim = self._calculate_structural_similarity(
                    query,
                    json.loads(record['entities']),
                    record['query_type']
                )

                # Combined score
                combined_score = (
                    0.5 * semantic_sim +           # Semantic
                    0.3 * structural_sim +         # Structural
                    0.2 * record['success_rate']   # Historical
                )

                templates.append({
                    'template_id': record['template_id'],
                    'query': record['template_query'],
                    'query_type': record['query_type'],
                    'semantic_similarity': float(semantic_sim),
                    'structural_similarity': float(structural_sim),
                    'combined_score': float(combined_score),
                    'central_nodes': json.loads(record['central_nodes']),
                    'entities': json.loads(record['entities']),
                    'relationships': json.loads(record['relationships']),
                    'topology': record['topology'],
                    'image_path': record['image_path'],
                    'plantuml_path': record['plantuml_path'],
                    'response': record['response'],
                    'cypher': record['cypher'],
                    'helpfulness_score': record['score'],
                    'reuse_count': record['reuse_count'],
                    'success_rate': record['success_rate'],
                    'depth': record['depth'],
                    'node_count': record['node_count']
                })

            # Sort by combined score
            templates.sort(key=lambda x: x['combined_score'], reverse=True)

            return templates[:top_k]

    def mark_template_reused(
        self,
        template_id: str,
        was_helpful: bool
    ):
        """
        Update template metrics when reused
        This is the LEARNING component!
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (rt:ReasoningTemplate {template_id: $template_id})
                SET rt.reuse_count = rt.reuse_count + 1,
                    rt.success_count = rt.success_count + CASE WHEN $helpful THEN 1 ELSE 0 END,
                    rt.success_rate = toFloat(rt.success_count + CASE WHEN $helpful THEN 1 ELSE 0 END) /
                                     toFloat(rt.reuse_count + 1),
                    rt.last_used = datetime()
            """, {
                'template_id': template_id,
                'helpful': was_helpful
            })

        print(f"✅ Updated template {template_id} reuse metrics (helpful: {was_helpful})")

    def get_template_stats(self) -> Dict:
        """Get statistics about stored templates"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (rt:ReasoningTemplate)
                RETURN count(rt) as total_templates,
                       sum(CASE WHEN rt.is_validated THEN 1 ELSE 0 END) as validated_count,
                       avg(rt.helpfulness_score) as avg_score,
                       sum(rt.reuse_count) as total_reuses,
                       avg(rt.success_rate) as avg_success_rate
            """)

            record = result.single()
            return {
                'total_templates': record['total_templates'],
                'validated_templates': record['validated_count'],
                'average_helpfulness': record['avg_score'],
                'total_reuses': record['total_reuses'],
                'average_success_rate': record['avg_success_rate']
            }

    def _generate_template_id(self, reasoning_path: Dict) -> str:
        """Generate unique template ID"""
        hash_input = f"{reasoning_path['query']}_{reasoning_path.get('cypher_query', '')}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _store_image(self, template_id: str, source_path: str, image_type: str) -> str:
        """Copy image to permanent storage"""
        ext = os.path.splitext(source_path)[1]
        dest_filename = f"{template_id}_{image_type}{ext}"
        dest_path = os.path.join(IMAGE_STORAGE, dest_filename)
        shutil.copy2(source_path, dest_path)
        return dest_path

    def _calculate_structural_similarity(
        self,
        query: str,
        template_entities: List[str],
        template_query_type: str
    ) -> float:
        """Calculate structural similarity between queries"""
        import re

        # Extract entities from query
        query_entities = re.findall(r'MRM-\d+', query)

        # Query type match
        query_type = self._classify_query_type(query)
        type_match = 1.0 if query_type == template_query_type else 0.5

        # Entity overlap
        if not query_entities:
            entity_sim = 0.0
        else:
            overlap = len(set(query_entities) & set(template_entities))
            entity_sim = overlap / len(query_entities)

        return 0.6 * type_match + 0.4 * entity_sim

    def _classify_query_type(self, query: str) -> str:
        """Classify query type"""
        q = query.lower()
        if 'relate' in q:
            return 'relationship'
        elif 'depend' in q:
            return 'dependency'
        elif 'assign' in q or 'who' in q:
            return 'assignment'
        return 'general'

    def _load_kv_store(self) -> Dict:
        """Load key-value store"""
        if os.path.exists(KV_STORE_FILE):
            with open(KV_STORE_FILE, 'r') as f:
                return json.load(f)
        return {}

    def _update_kv_store(self, template_id: str, data: Dict):
        """Update key-value store"""
        self.kv_store[template_id] = data
        with open(KV_STORE_FILE, 'w') as f:
            json.dump(self.kv_store, f, indent=2)

    def _initialize_schema(self):
        """Initialize database schema"""
        with self.driver.session() as session:
            # Create index for faster queries
            session.run("""
                CREATE INDEX template_id_index IF NOT EXISTS
                FOR (rt:ReasoningTemplate) ON (rt.template_id)
            """)
            session.run("""
                CREATE INDEX query_type_index IF NOT EXISTS
                FOR (rt:ReasoningTemplate) ON (rt.query_type)
            """)

    def close(self):
        """Close database connection"""
        self.driver.close()


if __name__ == "__main__":
    # Test
    db = TemplateDatabase()
    stats = db.get_template_stats()
    print(f"\nTemplate Database Stats:")
    print(f"  Total templates: {stats['total_templates']}")
    print(f"  Validated: {stats['validated_templates']}")
    print(f"  Avg helpfulness: {stats['average_helpfulness']:.2f}")
    print(f"  Total reuses: {stats['total_reuses']}")
    db.close()
