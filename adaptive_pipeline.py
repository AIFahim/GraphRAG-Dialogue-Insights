#!/usr/bin/env python3
"""
Adaptive GraphRAG Pipeline
Integrates vision extraction + template storage with existing pipeline
Learns from user feedback to improve over time
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poc-subgraph-imaging'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from its_llm_pipeline import SimpleLLMPipeline
from src.vision_extractor import VisionGraphExtractor, ReasoningPathBuilder
from src.template_database import TemplateDatabase


class AdaptiveGraphRAGPipeline:
    """
    Self-improving GraphRAG system that learns from user validation

    Flow:
    1. Check for similar past queries (template retrieval)
    2. Run normal pipeline (LLM + Neo4j + visualization)
    3. Extract structure from generated visualization (vision model)
    4. Ask user for validation
    5. Store as template if helpful
    6. Learn patterns over time
    """

    def __init__(self):
        print("Initializing Adaptive GraphRAG Pipeline...")

        # Core pipeline
        self.pipeline = SimpleLLMPipeline()

        # Vision extraction (for multimodal analysis)
        self.vision_extractor = VisionGraphExtractor()

        # Template database (separate storage!)
        self.template_db = TemplateDatabase()

        print("✅ Adaptive pipeline ready!\n")

    def run_adaptive_query(
        self,
        user_query: str,
        auto_validate: bool = False,
        assumed_helpful: bool = True,
        assumed_score: float = 0.8
    ) -> dict:
        """
        Run query with adaptive learning

        Args:
            user_query: The question to answer
            auto_validate: Skip user prompts (for testing)
            assumed_helpful: Assumed validation if auto
            assumed_score: Assumed score if auto
        """
        print(f"\n{'='*70}")
        print(f"ADAPTIVE QUERY: {user_query}")
        print(f"{'='*70}")

        # STEP 1: Check for similar templates
        print("\n[STEP 1] Searching for similar past queries...")
        similar_templates = self.template_db.find_similar_templates(
            query=user_query,
            top_k=3,
            min_score=0.5,
            require_validated=True
        )

        if similar_templates:
            print(f"\n🔍 Found {len(similar_templates)} similar patterns:")
            for i, template in enumerate(similar_templates, 1):
                print(f"\n  {i}. Query: {template['query'][:60]}...")
                print(f"     Similarity: {template['combined_score']:.3f}")
                print(f"     Topology: {template['topology']}")
                print(f"     Success rate: {template['success_rate']:.2%}")
                print(f"     Reused: {template['reuse_count']} times")
                print(f"     Image: {template['image_path']}")
        else:
            print("  No similar templates found (will create new one)")

        # STEP 2: Run normal pipeline
        print("\n[STEP 2] Running ITS LLM Pipeline...")
        result = self.pipeline.run(user_query)

        # Handle empty results
        if not result.get('nodes'):
            print(f"\n⚠️  No results found for query")
            print(f"   Query returned 0 records from Neo4j")
            return {
                **result,
                'reasoning_path': None,
                'template_id': None,
                'similar_templates': similar_templates,
                'user_validated': False,
                'helpfulness_score': 0.0
            }

        print(f"\n✅ Pipeline completed:")
        print(f"   Nodes: {len(result['nodes'])}")
        print(f"   Edges: {len(result['edges'])}")
        print(f"   Network graph: {result.get('image', 'N/A')}")
        if result.get('plantuml_image'):
            print(f"   PlantUML: {result['plantuml_image']}")

        # STEP 3: Extract structure from visualization
        print("\n[STEP 3] Extracting graph structure from visualization...")
        graph_structure = self.vision_extractor.extract_complete_structure(
            result['image']
        )

        # STEP 4: Build reasoning path
        print("\n[STEP 4] Building reasoning path...")
        reasoning_path = ReasoningPathBuilder.build_reasoning_path(
            graph_structure=graph_structure,
            query=user_query,
            cypher=result['cypher'],
            response=result.get('response', '')
        )

        print(f"\n✅ Reasoning path built:")
        print(f"   Central nodes: {reasoning_path['central_nodes']}")
        print(f"   Topology: {reasoning_path['topology']['type']}")
        print(f"   Depth: {reasoning_path['pattern']['depth']}")
        print(f"   Pattern: {reasoning_path['pattern']['node_count']} nodes, "
              f"{reasoning_path['pattern']['edge_count']} edges")

        # STEP 5: User validation
        if auto_validate:
            user_helpful = assumed_helpful
            helpfulness_score = assumed_score
            user_feedback = "Auto-validated for testing"
        else:
            print("\n[STEP 5] User Validation")
            print(f"\nVisualization saved to: {result['image']}")
            user_helpful = input("Was this visualization helpful? (y/n): ").lower() == 'y'

            if user_helpful:
                helpfulness_score = float(input("Rate helpfulness (0-1): "))
                user_feedback = input("Optional feedback: ") or None
            else:
                helpfulness_score = 0.3
                user_feedback = input("What was wrong? ") or None

        # STEP 6: Store as template
        print("\n[STEP 6] Storing reasoning template...")
        template_id = self.template_db.store_template(
            reasoning_path=reasoning_path,
            image_path=result['image'],
            plantuml_path=result.get('plantuml_image'),
            user_helpful=user_helpful,
            helpfulness_score=helpfulness_score,
            user_feedback=user_feedback
        )

        # STEP 7: Update similar templates if reused
        if similar_templates and user_helpful:
            print("\n[STEP 7] Updating reused template metrics...")
            # Mark most similar as reused
            best_match = similar_templates[0]
            self.template_db.mark_template_reused(
                template_id=best_match['template_id'],
                was_helpful=user_helpful
            )

        # Return complete result
        return {
            **result,
            'reasoning_path': reasoning_path,
            'template_id': template_id,
            'similar_templates': similar_templates,
            'user_validated': user_helpful,
            'helpfulness_score': helpfulness_score
        }

    def show_learning_stats(self):
        """Display what the system has learned"""
        stats = self.template_db.get_template_stats()

        print("\n" + "="*70)
        print("ADAPTIVE SYSTEM LEARNING STATISTICS")
        print("="*70)
        print(f"\n📊 Total Templates Stored: {stats['total_templates']}")
        print(f"✅ User-Validated Templates: {stats['validated_templates']}")
        print(f"⭐ Average Helpfulness Score: {stats['average_helpfulness']:.2f}/1.0")
        print(f"🔄 Total Pattern Reuses: {stats['total_reuses']}")
        print(f"📈 Average Success Rate: {stats['average_success_rate']:.1%}")
        print("\n" + "="*70)

    def close(self):
        """Clean up resources"""
        self.pipeline.close()
        self.template_db.close()


def interactive_mode():
    """Interactive mode for testing"""
    pipeline = AdaptiveGraphRAGPipeline()

    print("\n" + "="*70)
    print("  ADAPTIVE GRAPHRAG INTERACTIVE MODE")
    print("="*70)
    print("\nCommands:")
    print("  - Type a question to query")
    print("  - 'stats' to see learning statistics")
    print("  - 'quit' to exit")
    print("\n" + "="*70)

    try:
        while True:
            user_input = input("\n💬 Your query: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if user_input.lower() == 'stats':
                pipeline.show_learning_stats()
                continue

            # Run adaptive query
            result = pipeline.run_adaptive_query(user_input)

            print(f"\n✅ Query completed! Template ID: {result['template_id']}")

    finally:
        pipeline.close()
        print("\n✅ Pipeline closed. Goodbye!")


def batch_test_mode():
    """Test mode with example queries"""
    pipeline = AdaptiveGraphRAGPipeline()

    test_queries = [
        "What issues does MRM-488 relate to?",
        "Show me relationships for MRM-615",
        "What does MRM-487 depend on?",
    ]

    print("\n" + "="*70)
    print("  BATCH TEST MODE")
    print("="*70)
    print(f"\nRunning {len(test_queries)} test queries...")

    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'='*70}")

        result = pipeline.run_adaptive_query(
            user_query=query,
            auto_validate=True,
            assumed_helpful=True,
            assumed_score=0.85
        )

        print(f"\n✅ Query {i} completed!")

    # Show what was learned
    pipeline.show_learning_stats()
    pipeline.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        batch_test_mode()
    else:
        interactive_mode()
