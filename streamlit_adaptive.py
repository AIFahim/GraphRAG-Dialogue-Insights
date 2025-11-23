#!/usr/bin/env python3
"""
Adaptive GraphRAG Streamlit UI
Side-by-side comparison: With vs Without Adaptive Learning
"""
import streamlit as st
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poc-subgraph-imaging'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from its_llm_pipeline import SimpleLLMPipeline
from src.vision_extractor import VisionGraphExtractor, ReasoningPathBuilder
from src.template_database import TemplateDatabase

# Page config
st.set_page_config(
    page_title="Adaptive GraphRAG Demo",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'base_pipeline' not in st.session_state:
    st.session_state.base_pipeline = SimpleLLMPipeline()
if 'vision_extractor' not in st.session_state:
    st.session_state.vision_extractor = VisionGraphExtractor()
if 'template_db' not in st.session_state:
    st.session_state.template_db = TemplateDatabase()
if 'history' not in st.session_state:
    st.session_state.history = []
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = True

# Header
st.title("🧠 Adaptive GraphRAG Issue Tracking System")
st.markdown("**Compare: Standard vs Adaptive Learning**")
st.markdown("---")

# Sidebar - Stats & Controls
with st.sidebar:
    st.header("📊 Adaptive Learning Stats")

    stats = st.session_state.template_db.get_template_stats()

    st.metric("Total Templates", stats['total_templates'] or 0)
    st.metric("Validated Templates", stats['validated_templates'] or 0)
    st.metric("Avg Helpfulness", f"{stats['average_helpfulness'] or 0:.2f}/1.0")
    st.metric("Total Reuses", stats['total_reuses'] or 0)
    st.metric("Success Rate", f"{stats['average_success_rate'] or 0:.1%}")

    st.markdown("---")

    st.header("⚙️ Settings")
    st.session_state.comparison_mode = st.checkbox(
        "Show Comparison",
        value=True,
        help="Compare standard vs adaptive side-by-side"
    )

    min_similarity = st.slider(
        "Min Similarity Threshold",
        0.0, 1.0, 0.5,
        help="Minimum similarity to suggest templates"
    )

    st.markdown("---")
    st.header("💡 Example Questions")
    examples = [
        "What issues does MRM-488 relate to?",
        "Show bugs that relate to other bugs",
        "Which developer has the most bugs?",
        "What components does MRM-615 affect?",
    ]

    for idx, example in enumerate(examples):
        if st.button(example, key=f"ex_{idx}", use_container_width=True):
            st.session_state.current_query = example
            st.rerun()

# Main area
if st.session_state.comparison_mode:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🤖 Standard Pipeline")
        st.caption("Fresh generation every time")

    with col_right:
        st.subheader("🧠 Adaptive Pipeline")
        st.caption("Learns from past queries")
else:
    st.subheader("🧠 Adaptive GraphRAG")

# Query input
query = st.text_input(
    "🔎 Ask a question about the issue tracking system:",
    placeholder="e.g., What issues does MRM-488 relate to?",
    value=st.session_state.get('current_query', '')
)

run_query = st.button("🚀 Run Query", type="primary", use_container_width=True)

# Process query
if run_query and query:
    st.session_state.current_query = query

    if st.session_state.comparison_mode:
        # COMPARISON MODE: Show both side-by-side
        col_standard, col_adaptive = st.columns(2)

        # Standard pipeline (left)
        with col_standard:
            with st.spinner("Running standard pipeline..."):
                standard_result = st.session_state.base_pipeline.run(query)

                st.success("✅ Standard completed")

                if standard_result.get('response'):
                    st.info(f"💬 {standard_result['response']}")

                # Show visualization
                if standard_result.get('image') and os.path.exists(standard_result['image']):
                    st.image(standard_result['image'], caption="Network Graph", use_container_width=True)

                # Metrics
                st.metric("Nodes", len(standard_result.get('nodes', [])))
                st.metric("Edges", len(standard_result.get('edges', [])))

                with st.expander("🔍 Details"):
                    st.code(standard_result['cypher'], language='cypher')
                    st.json(standard_result['results'][:3])

        # Adaptive pipeline (right)
        with col_adaptive:
            # Search for similar templates
            similar_templates = st.session_state.template_db.find_similar_templates(
                query=query,
                top_k=3,
                min_score=min_similarity,
                require_validated=True
            )

            if similar_templates:
                st.success(f"🔍 Found {len(similar_templates)} similar patterns!")

                best = similar_templates[0]
                st.info(f"""
📚 **Most Similar Template:**
- Query: "{best['query'][:60]}..."
- Similarity: {best['combined_score']:.3f}
- Reused: {best['reuse_count']} times
- Success: {best['success_rate']:.1%}
                """)

                # Show past visualization
                if best['image_path'] and os.path.exists(best['image_path']):
                    st.image(
                        best['image_path'],
                        caption=f"Past visualization (reused {best['reuse_count']} times)",
                        use_container_width=True
                    )
            else:
                st.warning("No similar templates found (will create new one)")

            # Run with adaptive
            with st.spinner("Running adaptive pipeline..."):
                adaptive_result = st.session_state.base_pipeline.run(query)

                # Extract structure from visualization
                if adaptive_result.get('image') and os.path.exists(adaptive_result['image']):
                    with st.spinner("Extracting structure from visualization..."):
                        graph_structure = st.session_state.vision_extractor.extract_complete_structure(
                            adaptive_result['image']
                        )

                        # Show extracted info
                        st.success("✅ Adaptive completed + Vision extracted!")

                        if graph_structure['entities']:
                            st.info(f"""
🎯 **Extracted from Image:**
- Entities: {len(graph_structure['entities'])} nodes
- Relationships: {len(graph_structure['relationships'])} edges
- Topology: {graph_structure['topology'].get('topology_type', 'unknown')}
- Central nodes: {len(graph_structure['topology'].get('central_nodes', []))}
                            """)

                            # Build reasoning path
                            reasoning_path = ReasoningPathBuilder.build_reasoning_path(
                                graph_structure=graph_structure,
                                query=query,
                                cypher=adaptive_result['cypher'],
                                response=adaptive_result.get('response', '')
                            )

                            # Store for validation
                            st.session_state.last_result = {
                                'reasoning_path': reasoning_path,
                                'image_path': adaptive_result['image'],
                                'plantuml_path': adaptive_result.get('plantuml_image')
                            }

                            # Show topology
                            with st.expander("🔍 Reasoning Path Details"):
                                st.json({
                                    'central_nodes': reasoning_path['central_nodes'],
                                    'topology': reasoning_path['topology'],
                                    'pattern': reasoning_path['pattern']
                                })

                        # Show current visualization
                        st.image(
                            adaptive_result['image'],
                            caption="Current visualization",
                            use_container_width=True
                        )
    else:
        # ADAPTIVE ONLY MODE
        # Search for similar
        similar_templates = st.session_state.template_db.find_similar_templates(
            query=query,
            top_k=3,
            min_score=min_similarity,
            require_validated=True
        )

        if similar_templates:
            st.success(f"🔍 Found {len(similar_templates)} similar patterns!")

            # Show similar templates
            for i, template in enumerate(similar_templates[:2], 1):
                with st.expander(f"Similar Query {i}: {template['query'][:50]}... (Score: {template['combined_score']:.3f})"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Reused:** {template['reuse_count']} times")
                        st.write(f"**Success:** {template['success_rate']:.1%}")
                        st.write(f"**Topology:** {template['topology']}")
                    with col_b:
                        if template['image_path'] and os.path.exists(template['image_path']):
                            st.image(template['image_path'], use_container_width=True)

        # Run pipeline
        with st.spinner("Running adaptive pipeline..."):
            result = st.session_state.base_pipeline.run(query)

            # Check if we got results
            if not result.get('nodes') or len(result.get('nodes', [])) == 0:
                st.warning("⚠️ No results found - query returned 0 records from Neo4j")
                st.info(f"Query: `{result.get('cypher', 'N/A')}`")
                st.session_state.last_result = None
            # Extract and store
            elif result.get('image'):
                graph_structure = st.session_state.vision_extractor.extract_complete_structure(
                    result['image']
                )

                reasoning_path = ReasoningPathBuilder.build_reasoning_path(
                    graph_structure=graph_structure,
                    query=query,
                    cypher=result['cypher'],
                    response=result.get('response', '')
                )

                st.session_state.last_result = {
                    'reasoning_path': reasoning_path,
                    'image_path': result['image'],
                    'plantuml_path': result.get('plantuml_image'),
                    'query': query
                }

                # Display
                st.success("✅ Query completed!")
                if result.get('response'):
                    st.info(f"💬 {result['response']}")

                st.image(result['image'], use_container_width=True)

                with st.expander("🔍 Extracted Structure"):
                    st.json(reasoning_path)

# Validation form - OUTSIDE query processing block so it persists on rerun
if st.session_state.get('last_result'):
    st.markdown("---")
    with st.form("validation_form"):
        st.subheader("⭐ Validate This Visualization")
        st.write(f"Query: *{st.session_state.last_result.get('query', 'Unknown')}*")

        helpful = st.radio("Was this visualization helpful?", ["Yes", "No"])
        score = st.slider("Helpfulness Score", 0.0, 1.0, 0.8, 0.1)
        feedback = st.text_area("Optional feedback:")

        submitted = st.form_submit_button("💾 Submit Validation")

        if submitted:
            # Store template
            template_id = st.session_state.template_db.store_template(
                reasoning_path=st.session_state.last_result['reasoning_path'],
                image_path=st.session_state.last_result['image_path'],
                plantuml_path=st.session_state.last_result.get('plantuml_path'),
                user_helpful=(helpful == "Yes"),
                helpfulness_score=score if helpful == "Yes" else 0.3,
                user_feedback=feedback if feedback else None
            )

            st.success(f"✅ Stored as template: {template_id}")
            st.balloons()

            # Clear state and rerun
            st.session_state.last_result = None
            st.rerun()

# History section
st.markdown("---")
st.header("📈 Learning Progress")

col_hist1, col_hist2, col_hist3 = st.columns(3)

with col_hist1:
    st.metric(
        "Templates Stored",
        stats['total_templates'] or 0,
        help="Total reasoning patterns learned"
    )

with col_hist2:
    st.metric(
        "Pattern Reuses",
        stats['total_reuses'] or 0,
        help="How many times patterns were reused"
    )

with col_hist3:
    st.metric(
        "Success Rate",
        f"{stats['average_success_rate'] or 0:.1%}",
        help="% of reused patterns that were helpful"
    )

# Show recent templates
with st.expander("📚 View Stored Templates"):
    # Get all templates
    with st.session_state.template_db.driver.session() as session:
        result = session.run("""
            MATCH (rt:ReasoningTemplate)
            RETURN rt.template_id as id,
                   rt.query as query,
                   rt.topology_type as topology,
                   rt.reuse_count as reuses,
                   rt.success_rate as success,
                   rt.helpfulness_score as score,
                   rt.network_image_path as image
            ORDER BY rt.reuse_count DESC, rt.success_rate DESC
            LIMIT 10
        """)

        templates = [dict(record) for record in result]

    if templates:
        for template in templates:
            col_t1, col_t2, col_t3 = st.columns([3, 1, 1])

            with col_t1:
                st.write(f"**{template['query'][:60]}...**")
                st.caption(f"Topology: {template['topology']}")

            with col_t2:
                st.write(f"Reused: {template['reuses']}")
                st.write(f"Score: {template['score']:.2f}")

            with col_t3:
                if template['success'] is not None:
                    st.write(f"Success: {template['success']:.0%}")

            st.markdown("---")
    else:
        st.info("No templates stored yet. Run some queries to build the template library!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🧠 Adaptive Learning System | Vision: GLM-4.5V | LLM: Qwen 2.5 Coder | DB: Neo4j
</div>
""", unsafe_allow_html=True)
