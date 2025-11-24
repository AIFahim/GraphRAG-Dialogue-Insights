#!/usr/bin/env python3
"""
TRUE Adaptive GraphRAG Streamlit UI
Uses AdaptiveITSPipeline that ACTUALLY passes templates to LLM

KEY DIFFERENCE from streamlit_adaptive.py:
- Uses AdaptiveITSPipeline instead of SimpleLLMPipeline
- Templates and feedbacks are INJECTED into LLM prompts
- LLM generates Cypher using learned patterns
- TRUE feedback-driven learning!
"""
import streamlit as st
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poc-subgraph-imaging'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adaptive_its_pipeline import AdaptiveITSPipeline
from vision_extractor import VisionGraphExtractor, ReasoningPathBuilder
from template_database import TemplateDatabase

# Page config
st.set_page_config(
    page_title="TRUE Adaptive GraphRAG",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'adaptive_pipeline' not in st.session_state:
    st.session_state.adaptive_pipeline = AdaptiveITSPipeline()
if 'vision_extractor' not in st.session_state:
    st.session_state.vision_extractor = VisionGraphExtractor()
if 'template_db' not in st.session_state:
    st.session_state.template_db = TemplateDatabase()

# Header
st.title("🧠 TRUE Adaptive GraphRAG")
st.markdown("**System that ACTUALLY learns from your feedback**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Learning Stats")

    stats = st.session_state.template_db.get_template_stats()

    st.metric("Templates", stats['total_templates'] or 0)
    st.metric("Avg Score", f"{stats['average_helpfulness'] or 0:.2f}/1.0")
    st.metric("Reuses", stats['total_reuses'] or 0)
    st.metric("Success", f"{stats['average_success_rate'] or 0:.1%}")

    st.markdown("---")

    # Show what system learned
    st.header("🎓 System Has Learned")

    all_templates = st.session_state.template_db.get_all_templates()

    if all_templates:
        positive = [t for t in all_templates if t['helpfulness_score'] >= 0.8 and t['user_feedback']]
        if positive:
            st.subheader("✅ User Loved:")
            for t in positive[:3]:
                st.success(f'"{t["user_feedback"][:80]}..."')

        negative = [t for t in all_templates if t['helpfulness_score'] < 0.5 and t['user_feedback']]
        if negative:
            st.subheader("❌ User Disliked:")
            for t in negative[:3]:
                st.error(f'"{t["user_feedback"][:80]}..."')
    else:
        st.info("No feedback yet - system will learn as you use it!")

# Main content
query = st.text_input(
    "🔎 Ask a question:",
    placeholder="e.g., Show me all relationships for MRM-681"
)

run_btn = st.button("🚀 Run with TRUE Learning", type="primary", use_container_width=True)

if run_btn and query:
    # Search for similar (show user what LLM will see)
    with st.expander("🔍 What LLM Will See", expanded=True):
        similar = st.session_state.template_db.find_similar_templates(query, top_k=2, min_score=0.5)

        if similar:
            st.write("**Similar Successful Patterns:**")
            for t in similar:
                st.write(f"- \"{t['query'][:60]}...\" (score: {t['helpfulness_score']:.2f}, similarity: {t['combined_score']:.3f})")
                if t.get('user_feedback'):
                    st.info(f"Feedback: \"{t['user_feedback']}\"")

        all_fb = st.session_state.template_db.get_all_templates()
        if all_fb:
            st.write(f"\n**All User Feedbacks ({len(all_fb)} total):**")
            for t in all_fb[:5]:
                if t.get('user_feedback'):
                    emoji = "✅" if t['helpfulness_score'] >= 0.8 else ("❌" if t['helpfulness_score'] < 0.5 else "⚠️")
                    st.write(f"{emoji} \"{t['user_feedback'][:60]}...\" (score: {t['helpfulness_score']:.2f})")

    # Run with adaptive learning
    with st.spinner("🧠 Running with TRUE adaptive learning..."):
        result = st.session_state.adaptive_pipeline.run(query)

        if result.get('nodes'):
            st.success("✅ Query completed with learning applied!")

            if result.get('response'):
                st.info(f"💬 {result['response']}")

            # Show visualization
            if result.get('image') and os.path.exists(result['image']):
                st.image(result['image'], use_container_width=True)

            # Extract and store
            graph_structure = st.session_state.vision_extractor.extract_complete_structure(
                result['image']
            )

            reasoning_path = ReasoningPathBuilder.build_reasoning_path(
                graph_structure=graph_structure,
                query=query,
                cypher=result['cypher'],
                response=result.get('response', '')
            )

            # Store for validation
            st.session_state.last_result = {
                'reasoning_path': reasoning_path,
                'image_path': result['image'],
                'plantuml_path': result.get('plantuml_image'),
                'query': query
            }

            with st.expander("🔍 Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Generated Cypher:**")
                    st.code(result['cypher'], language='cypher')
                with col2:
                    st.write("**Extracted Structure:**")
                    st.json({
                        'entities': graph_structure['entities'][:5],
                        'relationships': len(graph_structure['relationships']),
                        'topology': graph_structure['topology']
                    })
        else:
            st.warning("No results found")

# Validation form (OUTSIDE query block!)
if st.session_state.get('last_result'):
    st.markdown("---")
    with st.form("validation"):
        st.subheader("⭐ Validate & Teach the System")
        st.write(f"Query: *{st.session_state.last_result.get('query')}*")

        helpful = st.radio("Helpful?", ["Yes", "No"])
        score = st.slider("Score", 0.0, 1.0, 0.8, 0.1)

        st.write("**💡 Your feedback teaches the system:**")
        st.caption("Be specific! Say what you liked/disliked. The LLM will read and apply this.")

        feedback = st.text_area(
            "Feedback (tell the system what to do differently):",
            placeholder='e.g., "Perfect! Shows duplicates I needed" or "Missing AFFECTS relationships"'
        )

        submit = st.form_submit_button("💾 Submit & Teach System")

        if submit:
            template_id = st.session_state.template_db.store_template(
                reasoning_path=st.session_state.last_result['reasoning_path'],
                image_path=st.session_state.last_result['image_path'],
                plantuml_path=st.session_state.last_result.get('plantuml_path'),
                user_helpful=(helpful == "Yes"),
                helpfulness_score=score if helpful == "Yes" else 0.3,
                user_feedback=feedback if feedback else None
            )

            st.success(f"✅ Stored! LLM will see this feedback on future similar queries")
            st.balloons()

            st.session_state.last_result = None
            st.rerun()

# Footer
st.markdown("---")
st.caption("🧠 TRUE Adaptive Learning: Templates + Feedbacks passed to LLM")
