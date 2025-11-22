# GraphRAG ITS Demo - Streamlit UI

## Quick Start

### 1. Install Streamlit
```bash
pip install streamlit
```

### 2. Run the App
```bash
cd /mnt/c/Users/User/PycharmProjects/GraphRAG-Dialogue-Insights/data/seoss_extracted
streamlit run streamlit_app.py
```

### 3. Open Browser
- Streamlit will automatically open: `http://localhost:8501`

---

## Features

✅ **Interactive Query Input**
- Type natural language questions
- Click example questions in sidebar

✅ **LLM-Powered Cypher Generation**
- Automatic Cypher query generation
- Full graph structure (source-relationship-target)

✅ **Visual Graph Display**
- Network graph visualization
- Color-coded nodes (Issues = red, Others = teal)
- Labeled relationships

✅ **Query Results**
- Expandable JSON records
- First 5 results shown

✅ **Follow-up Suggestions**
- LLM-generated natural language suggestions
- Click to ask follow-up questions

✅ **Query History**
- See past 5 queries
- View graphs and results

---

## Example Questions

```
What issues does MRM-488 relate to?
Show bugs that relate to other bugs
Which developer has the most bugs?
Find critical bugs that were closed
Show duplicate issues
```

---

## UI Layout

```
┌─────────────────────────────────────────────┐
│  🔍 GraphRAG Issue Tracking System Demo     │
├─────────────────────────────────────────────┤
│  Sidebar:                                    │
│  - Example questions                         │
│  - Clear history button                      │
│                                              │
│  Main Area:                                  │
│  - Query input box                           │
│  - Run button                                │
│                                              │
│  Results (4 tabs):                           │
│  📊 Graph    - Network visualization         │
│  🔍 Results  - JSON data                     │
│  💡 Suggest  - Follow-up questions           │
│  📝 Cypher   - Generated query               │
│                                              │
│  History:                                    │
│  - Past 5 queries with graphs                │
└─────────────────────────────────────────────┘
```

---

## Files

- **`streamlit_app.py`** - Streamlit UI (main file)
- **`simple_llm_pipeline.py`** - Backend LLM pipeline
- **`results/`** - Generated graph images

---

## Troubleshooting

**Port already in use:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Can't find module:**
```bash
# Make sure you're in the correct directory
cd /mnt/c/Users/User/PycharmProjects/GraphRAG-Dialogue-Insights/data/seoss_extracted
```

**Neo4j not running:**
```bash
# Start Neo4j first
sudo systemctl start neo4j
# or
docker start neo4j
```
