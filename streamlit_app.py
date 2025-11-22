#!/usr/bin/env python3
"""
Streamlit UI for Simple LLM Pipeline
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import os
import sys

# Add poc-subgraph-imaging to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poc-subgraph-imaging'))

from its_llm_pipeline import SimpleLLMPipeline

# Page config
st.set_page_config(
    page_title="GraphRAG ITS Demo",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = SimpleLLMPipeline()
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# Header
st.title("🔍 GraphRAG Issue Tracking System Demo")
st.markdown("**LLM-powered Cypher generation + Graph visualization**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.markdown("""
    This demo uses:
    - **LLM**: Qwen 2.5 Coder
    - **Database**: Neo4j
    - **Dataset**: SEOSS Apache Archiva

    **Pipeline:**
    1. LLM generates Cypher
    2. Neo4j executes query
    3. LLM analyzes results
    4. Graph visualization
    """)

    st.markdown("---")
    st.header("💡 Example Questions")
    examples = [
        "What issues does MRM-488 relate to?",
        "Show bugs that relate to other bugs",
        "Which developer has the most bugs?",
        "Find critical bugs that were closed",
        "Show duplicate issues",
    ]

    for idx, example in enumerate(examples):
        if st.button(example, key=f"ex_{idx}", use_container_width=True):
            # Clear old result and run example query
            st.session_state.current_result = None
            try:
                example_result = st.session_state.pipeline.run(example)
                st.session_state.history.insert(0, example_result)
                st.session_state.current_result = example_result
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    # Query input
    query = st.text_input(
        "🔎 Ask a question about the issue tracking system:",
        placeholder="e.g., What issues does MRM-488 relate to?",
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    submit = st.button("🚀 Run Query", type="primary", use_container_width=True)

# Process manual query from text input
if submit and query:
    # Clear old result first
    st.session_state.current_result = None

    with st.spinner("🤖 LLM generating Cypher query..."):
        try:
            result = st.session_state.pipeline.run(query)
            st.session_state.history.insert(0, result)
            st.session_state.current_result = result
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Display current result if exists
if st.session_state.current_result:
    result = st.session_state.current_result

    # Display results
    st.success("✅ Query completed!")

    # Natural language response (prominent display)
    if result.get('response'):
        st.info(f"💬 **Response:** {result['response']}")

    # Tabs for different outputs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Graph", "🔍 Results", "💡 Suggestions", "🔗 Context", "📝 Cypher"])

    with tab1:
        st.subheader("Graph Visualization")

        # Graph stats at top
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Nodes", len(result.get('nodes', [])))
        col_b.metric("Edges", len(result.get('edges', [])))
        col_c.metric("Results", len(result.get('results', [])))

        st.markdown("---")

        # Show both visualizations
        viz_tab1, viz_tab2 = st.tabs(["🔷 Network Graph", "📐 PlantUML Diagram"])

        with viz_tab1:
            if result.get('image') and os.path.exists(result['image']):
                st.markdown("**Network Graph (Matplotlib)**")
                st.image(result['image'], use_container_width=True)
            else:
                st.info("No network graph generated")

        with viz_tab2:
            if result.get('plantuml_image') and os.path.exists(result['plantuml_image']):
                st.markdown("**PlantUML Diagram**")
                st.image(result['plantuml_image'], use_container_width=True)
            elif result.get('plantuml_file'):
                st.warning("PlantUML code generated but PNG render failed")
                with st.expander("View PlantUML Code"):
                    st.code(result.get('plantuml_code', ''), language='plantuml')
            else:
                st.info("No PlantUML diagram generated")

    with tab2:
        st.subheader("Query Results")
        if result.get('results'):
            st.write(f"**{len(result['results'])} records found**")

            # Show first 5 results in expandable sections
            for i, record in enumerate(result['results'][:5], 1):
                with st.expander(f"Record {i}", expanded=(i==1)):
                    st.json(record)

            if len(result['results']) > 5:
                st.info(f"+ {len(result['results']) - 5} more records")
        else:
            st.warning("No results found")

    with tab3:
        st.subheader("Follow-up Suggestions")
        if result.get('suggestions'):
            for i, suggestion in enumerate(result['suggestions'], 1):
                st.markdown(f"**{i}.** {suggestion}")
                if st.button(f"▶️ Ask this question", key=f"ask_{i}_{len(st.session_state.history)}", use_container_width=True):
                    # Clear old result and run the suggestion query
                    st.session_state.current_result = None
                    try:
                        followup_result = st.session_state.pipeline.run(suggestion)
                        st.session_state.history.insert(0, followup_result)
                        st.session_state.current_result = followup_result
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                st.markdown("---")
        else:
            st.info("No suggestions generated")

    with tab4:
        st.subheader("Related Edges Context")
        context = result.get('context', {})

        if context and context.get('edge_count', 0) > 0:
            st.write(f"**Found {context['edge_count']} related edges from {context['nodes']} unique nodes**")
            st.markdown("These are 1-hop neighboring relationships that provide additional context.")

            # Show context edges
            st.markdown("#### Context Edges:")
            for i, edge in enumerate(context.get('edges', [])[:15], 1):
                col_ctx1, col_ctx2 = st.columns([3, 1])
                col_ctx1.write(f"`{edge.get('source')}` --[**{edge.get('rel')}**]--> `{edge.get('target')}`")
                col_ctx2.write(f"_{edge.get('target_type', 'Unknown')}_")

            if len(context.get('edges', [])) > 15:
                st.info(f"+ {len(context['edges']) - 15} more context edges")

            # Summary stats
            st.markdown("#### Context Summary:")
            rel_types = {}
            for edge in context.get('edges', []):
                rel = edge.get('rel', 'Unknown')
                rel_types[rel] = rel_types.get(rel, 0) + 1

            for rel, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- **{rel}**: {count} edges")

        else:
            st.info("No additional context retrieved (query may not have returned Issue nodes)")

    with tab5:
        st.subheader("Generated Cypher Query")
        st.code(result['cypher'], language="cypher")

# History section
if st.session_state.history:
    st.markdown("---")
    st.header("📜 Query History")

    for i, hist_result in enumerate(st.session_state.history[:5]):
        with st.expander(f"Query {i+1}: {hist_result['query'][:60]}...", expanded=False):
            # Show response
            if hist_result.get('response'):
                st.write(f"**💬 Response:** {hist_result['response']}")
                st.markdown("---")

            col_hist1, col_hist2 = st.columns(2)

            with col_hist1:
                st.write("**Cypher:**")
                st.code(hist_result['cypher'][:150] + "...", language="cypher")

            with col_hist2:
                st.write("**Stats:**")
                st.write(f"- Results: {len(hist_result.get('results', []))}")
                st.write(f"- Nodes: {len(hist_result.get('nodes', []))}")
                st.write(f"- Edges: {len(hist_result.get('edges', []))}")

            if hist_result.get('image') and os.path.exists(hist_result['image']):
                st.image(hist_result['image'], width=400)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with ❤️ using Streamlit | LLM: Qwen 2.5 Coder | Database: Neo4j
</div>
""", unsafe_allow_html=True)
