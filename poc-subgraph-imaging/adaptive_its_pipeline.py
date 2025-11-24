#!/usr/bin/env python3
"""
TRUE Adaptive ITS Pipeline
Extends SimpleLLMPipeline to ACTUALLY use learned templates in LLM prompts

KEY DIFFERENCE from base pipeline:
- Retrieves similar templates from database
- Passes template data + raw user feedback to LLM
- LLM generates Cypher using learned patterns
- LLM follows user feedback instructions
"""

import os
import sys
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from its_llm_pipeline import SimpleLLMPipeline, SCHEMA
from template_database import TemplateDatabase


class AdaptiveITSPipeline(SimpleLLMPipeline):
    """
    Enhanced pipeline that injects learned templates into LLM prompts

    Inherits from SimpleLLMPipeline but overrides llm_generate_cypher()
    to include template context
    """

    def __init__(self):
        # Initialize base pipeline (Neo4j, HuggingFace client, etc.)
        super().__init__()

        # Initialize template database for learning
        print("Initializing template database for adaptive learning...")
        self.template_db = TemplateDatabase()
        print("✅ Template database connected")

    def llm_generate_cypher(self, user_query: str) -> str:
        """
        OVERRIDE: Generate Cypher with learned context from templates

        This is the KEY method that makes the system truly adaptive!

        Flow:
        1. Find similar successful templates (semantic search)
        2. Retrieve ALL user feedbacks (learn from successes + failures)
        3. Build enhanced prompt with templates + feedbacks
        4. LLM generates Cypher using learned patterns

        Args:
            user_query: The user's question

        Returns:
            Generated Cypher query (informed by learning!)
        """

        print(f"\n{'─'*70}")
        print("🧠 ADAPTIVE CYPHER GENERATION (WITH LEARNING)")
        print(f"{'─'*70}")

        # PART 1: Find similar successful templates
        print("\n[Learning Step 1] Searching for similar successful templates...")

        similar_templates = self.template_db.find_similar_templates(
            query=user_query,
            top_k=3,              # Top 3 most similar
            min_score=0.5,        # Minimum similarity threshold
            require_validated=True  # Only use user-validated templates
        )

        if similar_templates:
            print(f"  ✅ Found {len(similar_templates)} similar templates:")
            for i, t in enumerate(similar_templates, 1):
                print(f"     {i}. \"{t['query'][:50]}...\" (similarity: {t['combined_score']:.3f}, score: {t['helpfulness_score']:.2f})")
        else:
            print(f"  ℹ️  No similar templates found (generating fresh)")

        # PART 2: Get ALL user feedbacks (learn from all past queries)
        print("\n[Learning Step 2] Retrieving ALL user feedbacks...")

        all_feedbacks = self._get_all_feedbacks()

        positive_feedbacks = [f for f in all_feedbacks if f['score'] >= 0.8]
        negative_feedbacks = [f for f in all_feedbacks if f['score'] < 0.5]

        print(f"  ✅ Retrieved {len(all_feedbacks)} total feedbacks:")
        print(f"     - Positive (score >= 0.8): {len(positive_feedbacks)}")
        print(f"     - Negative (score < 0.5): {len(negative_feedbacks)}")

        # PART 3: Build enhanced prompt
        print("\n[Learning Step 3] Building enhanced prompt with learned context...")

        if similar_templates or all_feedbacks:
            # Build learning context
            learned_context = self._build_learned_context(
                similar_templates=similar_templates,
                all_feedbacks=all_feedbacks
            )

            # Enhanced prompt with learning
            enhanced_prompt = f"""{SCHEMA}

{learned_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW USER QUERY: {user_query}

INSTRUCTIONS:
1. Analyze the similar successful patterns above
2. Read ALL user feedback carefully (both positive and negative)
3. STRICTLY FOLLOW what users said they wanted
4. AVOID patterns users complained about
5. Include relationship types users valued (DUPLICATES, AFFECTS, etc.)
6. Apply the successful Cypher patterns shown above
7. Match the graph structure users found helpful

CRITICAL: Return ONLY the Cypher query, NO explanations, NO comments.

Generate Cypher query:"""

            print(f"  ✅ Enhanced prompt built:")
            print(f"     - Included {len(similar_templates)} template patterns")
            print(f"     - Included {len(positive_feedbacks)} positive feedbacks")
            print(f"     - Included {len(negative_feedbacks)} negative feedbacks")
        else:
            # No learning available yet - use base prompt
            print(f"  ℹ️  No learning context available, using base prompt")
            enhanced_prompt = f"""{SCHEMA}

User question: {user_query}

Determine if this question asks about:
A) RELATIONSHIPS between entities → Use MATCH (a)-[r]->(b) pattern
B) Just LISTING or COUNTING entities → Use MATCH (i:Issue) without relationships

Return ONLY the Cypher query, no explanation."""

        # PART 4: Generate with LLM
        print("\n[Learning Step 4] Generating Cypher with LLM (using learned context)...")

        response = self.client.chat_completion(
            messages=[{"role": "user", "content": enhanced_prompt}],
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            max_tokens=512,
            temperature=0.1
        )

        cypher = response.choices[0].message.content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()

        print(f"  ✅ Cypher generated (informed by {len(similar_templates)} templates)")
        print(f"{'─'*70}\n")

        return cypher

    def _build_learned_context(self, similar_templates: list, all_feedbacks: list) -> str:
        """
        Build comprehensive learning context from templates and feedbacks

        Args:
            similar_templates: Top N similar templates (with full data)
            all_feedbacks: ALL user feedbacks (positive and negative)

        Returns:
            Formatted string to inject into LLM prompt
        """

        context = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        context += "📚 LEARNED PATTERNS FROM SIMILAR SUCCESSFUL QUERIES:\n"
        context += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        # Add selected template details
        for i, template in enumerate(similar_templates, 1):
            context += f"""
Pattern {i} (Similarity: {template['combined_score']:.3f}, Score: {template['helpfulness_score']}/1.0):

  Query: "{template['query']}"

  Successful Cypher:
  {template['cypher']}

  Graph Structure Extracted:
  - Entities shown: {template['entities'][:5]}{'...' if len(template['entities']) > 5 else ''}
  - Relationships: {template['relationships'][:3] if template['relationships'] else 'N/A'}
  - Topology: {template['topology']}
  - Central nodes: {template['central_nodes']}
  - Depth: {template.get('depth', 'N/A')} hops
  - Node count: {template['node_count']}

  Performance Metrics:
  - User helpfulness: {template['helpfulness_score']:.2f}/1.0
  - Reuse count: {template['reuse_count']} times
  - Success rate: {template['success_rate']:.1%}

  USER FEEDBACK (FOLLOW THIS):
  "{template.get('user_feedback', 'No feedback provided')}"

"""

        # Add all feedbacks section
        context += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        context += "💡 LEARN FROM ALL USER FEEDBACK (What to do and NOT do):\n"
        context += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        # Positive feedbacks (what users loved)
        positive = [f for f in all_feedbacks if f['score'] >= 0.8 and f['feedback']]
        if positive:
            context += "\n✅ WHAT USERS LOVED (Do this):\n"
            for feedback in positive[:5]:  # Top 5
                context += f'   • "{feedback["feedback"]}"\n'

        # Negative feedbacks (what users disliked)
        negative = [f for f in all_feedbacks if f['score'] < 0.5 and f['feedback']]
        if negative:
            context += "\n❌ WHAT USERS COMPLAINED ABOUT (Avoid this):\n"
            for feedback in negative[:5]:  # Top 5
                context += f'   • "{feedback["feedback"]}"\n'

        context += "\n"

        return context

    def _get_all_feedbacks(self) -> list:
        """
        Retrieve ALL user feedbacks from template database

        Returns list of {feedback, score, query} dicts
        """

        with self.template_db.driver.session() as session:
            result = session.run("""
                MATCH (rt:ReasoningTemplate)
                WHERE rt.user_feedback IS NOT NULL
                RETURN rt.user_feedback as feedback,
                       rt.helpfulness_score as score,
                       rt.query as query
                ORDER BY rt.helpfulness_score DESC
            """)

            return [dict(record) for record in result]

    def run(self, user_query: str) -> dict:
        """
        Run pipeline with TRUE adaptive learning

        This overrides the base run() to use adaptive Cypher generation
        Everything else (context fetching, analysis, visualization) stays the same
        """

        print(f"\n{'='*70}")
        print(f"🧠 ADAPTIVE QUERY (TRUE LEARNING): {user_query}")
        print(f"{'='*70}")

        # Use adaptive Cypher generation (with templates!)
        # This calls our overridden llm_generate_cypher() above
        return super().run(user_query)

    def close(self):
        """Clean up resources"""
        super().close()
        self.template_db.close()


# ============================================================================
# TESTING & COMPARISON
# ============================================================================

def test_adaptive_learning():
    """
    Test script to demonstrate adaptive learning
    """

    print("\n" + "="*70)
    print("  ADAPTIVE LEARNING TEST")
    print("="*70)

    pipeline = AdaptiveITSPipeline()

    test_query = "Show me all relationships for MRM-681"

    print(f"\nTest query: {test_query}")
    print("\nThis will:")
    print("1. Search for similar templates")
    print("2. Retrieve all user feedbacks")
    print("3. Build enhanced prompt with learning")
    print("4. Generate Cypher using learned patterns")
    print("5. LLM will see and apply user feedback!")

    result = pipeline.run(test_query)

    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Cypher: {result['cypher']}")
    print(f"Results: {len(result['results'])} records")
    print(f"Response: {result.get('response', '')[:200]}...")

    pipeline.close()


if __name__ == "__main__":
    test_adaptive_learning()
