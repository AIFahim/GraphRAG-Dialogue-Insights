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

⚠️  MANDATORY INSTRUCTIONS - STRICT COMPLIANCE REQUIRED ⚠️

1. COMPREHENSIVE RELATIONSHIP TYPE INCLUSION MANDATORY:
   - For "show all relationships" queries, you MUST include ALL standard types
   - MANDATORY types to include: RELATES_TO, DUPLICATES, DEPENDS_UPON, AFFECTS, ASSIGNED_TO, REPORTED_BY
   - Vision-validated types from above confirm which types were successful - but include ALL types for completeness
   - Format: WHERE type(r) IN ['RELATES_TO', 'DUPLICATES', 'DEPENDS_UPON', 'AFFECTS', 'ASSIGNED_TO', 'REPORTED_BY']
   - Using only a subset of types = INCOMPLETE QUERY = SYSTEM FAILURE

2. HOW TO APPLY RELATIONSHIP TYPES:
   - ALWAYS use: WHERE type(r) IN ['RELATES_TO', 'DUPLICATES', 'DEPENDS_UPON', 'AFFECTS', 'ASSIGNED_TO', 'REPORTED_BY']
   - This ensures comprehensive coverage of all relationship types
   - DO NOT use generic -[r]- without WHERE clause
   - DO NOT use only a subset like ['DUPLICATES'] - you MUST include all 6 types
   - Vision-validated types above show which types users care about, but query must be comprehensive

3. USER FEEDBACK GOVERNS YOUR BEHAVIOR:
   - What users explicitly approved (✅) → You MUST replicate this behavior
   - What users explicitly rejected (❌) → You MUST avoid this behavior completely
   - User validation overrides default LLM preferences

4. QUERY STRUCTURE REQUIREMENTS:
   - Use MATCH (i:Issue {{id: 'XXX'}})-[r]-(related)  ← CRITICAL: BIDIRECTIONAL (no arrow!)
   - NEVER use -[r]-> or <-[r]-, ALWAYS use -[r]- to capture both directions
   - Add WHERE type(r) IN [<all validated types from above>]
   - Return i.id, type(r), related.id or related.name, labels(related)[0] as type
   - Include ALL relationship types shown in vision-validated list

5. STRICTLY AVOID REJECTED PATTERNS:
   - Any behavior users complained about is forbidden
   - Generic queries without relationship type filters are REJECTED
   - Incomplete queries missing validated types are REJECTED

This is an adaptive system. Vision analysis PROVED which relationship types existed
in successful queries. Your task is to include ALL validated types WITHOUT FAIL.

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
            # Extract unique relationship types from template
            rel_types = set()
            if template.get('relationships'):
                for rel in template['relationships']:
                    if isinstance(rel, dict) and 'type' in rel:
                        rel_types.add(rel['type'])

            rel_types_list = sorted(rel_types) if rel_types else []
            rel_types_str = f"WHERE type(r) IN {rel_types_list}" if rel_types_list else "No filter specified"

            context += f"""
Pattern {i} (Similarity: {template['combined_score']:.3f}, Score: {template['helpfulness_score']}/1.0):

  Query: "{template['query']}"

  Original Cypher (may be incomplete):
  {template['cypher']}

  ⚠️  VISION-VALIDATED RELATIONSHIP TYPES THAT WERE ACTUALLY PRESENT:
  {rel_types_list}

  ⚠️  YOU MUST INCLUDE THESE TYPES IN YOUR QUERY:
  {rel_types_str}

  Graph Structure Extracted:
  - Entities shown: {template['entities'][:5]}{'...' if len(template['entities']) > 5 else ''}
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

    def llm_analyze_results(self, user_query: str, results: list, context: dict = None) -> dict:
        """
        OVERRIDE: Analyze results with learned visualization patterns

        Makes the SECOND LLM call also adaptive (not just Cypher generation).
        Reuses _build_learned_context() to avoid code duplication.
        """

        print(f"\n[Adaptive Analysis] Using learned visualization patterns...")

        # Retrieve templates (same as Cypher generation)
        templates = self.template_db.find_similar_templates(
            query=user_query,
            top_k=2,
            min_score=0.6,
            require_validated=True
        )

        all_feedbacks = self._get_all_feedbacks()

        # Build base prompt parts
        results_json = json.dumps(results[:15], indent=2, default=str)

        context_info = ""
        if context and context.get('edges'):
            context_json = json.dumps(context['edges'][:20], indent=2, default=str)
            context_info = f"""

Additional Context (1-hop neighboring edges):
{context_json}
"""

        # Reuse learned context builder if templates exist
        learned_viz_context = ""
        if templates or all_feedbacks:
            learned_viz_context = self._build_learned_context(templates, all_feedbacks)
            learned_viz_context += """

APPLY LEARNED PATTERNS TO VISUALIZATION:
- Include relationship types that were successful (especially DUPLICATES if in results)
- Match node/edge count from similar successful visualizations
- Emphasize relationships users valued in feedback
"""

        # Build complete prompt
        prompt = f"""User asked: {user_query}

Main query results:
{results_json}
{context_info}

{learned_viz_context}

⚠️  CRITICAL: The "Additional Context" above contains 1-hop neighboring edges.
These are edges BETWEEN the entities in the main results

YOU MUST include ALL context edges in your visualization, especially DUPLICATES relationships.

Analyze and return JSON with:
1. "response": Natural language answer (2-3 sentences, mention DUPLICATES if present in context)
2. "nodes": ALL node IDs from main results + context
3. "edges": ALL edges from both main results AND context
   - For main result edges: {{"source": "...", "target": "...", "label": "...", "is_context": false}}
   - For context edges: {{"source": "...", "target": "...", "label": "...", "is_context": true}}
4. "suggestions": 5 follow-up questions

Return ONLY valid JSON, no explanations (REPLACE XYZ and ABC with real numbers, just take as example):

{{
    "response": Natural language answer (2-3 sentences, explain what you found INCLUDING insights from context),
    "nodes": ["MRM-615", "MRM-686", "MRM-952", ...],
    "edges": [
        {{"source": "MRM-615", "target": "MRM-686", "label": "RELATES_TO", "is_context": false}},
        {{"source": "MRM-XYZ", "target": "MRM-ABC", "label": "DUPLICATES", "is_context": true}},
        ...
    ],
    "suggestions": ["Who is assigned?", "What components affected?", ...]
}}"""

        # Call LLM
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            max_tokens=1536,
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print("  Extracting basic info from malformed response...")

            # Fallback: Extract nodes manually
            nodes = []
            edges = []

            # Extract Issue IDs from results
            for record in results[:10]:
                for val in record.values():
                    if isinstance(val, str) and val.startswith('MRM-'):
                        if val not in nodes:
                            nodes.append(val)

            # Try to extract edges from context
            if context and context.get('edges'):
                for ctx_edge in context['edges'][:10]:
                    edges.append({
                        "source": ctx_edge.get('source'),
                        "target": ctx_edge.get('target'),
                        "label": ctx_edge.get('rel'),
                        "is_context": True
                    })

            return {
                "response": f"Found {len(results)} results. (Note: Full analysis unavailable due to formatting issue)",
                "nodes": nodes[:15],
                "edges": edges[:15],
                "suggestions": [
                    "Show more details about these issues",
                    "Who is assigned to these?",
                    "What components are affected?",
                    "Show related bugs",
                    "Check dependencies"
                ]
            }

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
