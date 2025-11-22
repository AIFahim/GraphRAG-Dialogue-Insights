# GraphRAG ITS Demo - Complete Guide

## 🚀 Quick Start

### Option 1: Streamlit UI (Recommended for Demo)
```bash
cd /mnt/c/Users/User/PycharmProjects/GraphRAG-Dialogue-Insights/data/seoss_extracted
streamlit run streamlit_app.py
```

### Option 2: Command Line
```bash
python simple_llm_pipeline.py
```

---

## ✨ Features

### 1. **Natural Language Response** 💬
- LLM generates human-readable answer
- Example: *"I found 2 related issues for MRM-488. MRM-488 relates to MRM-615 and MRM-487, both dealing with property resolution problems in POMs."*

### 2. **LLM-Powered Cypher Generation** 🤖
- Automatic Cypher query from natural language
- Full graph structure (source-relationship-target)
- No manual keyword matching

### 3. **Graph Visualization** 📊
- Network graph with colored nodes
- Red = Issue nodes (MRM-xxx)
- Teal = Other entities (Person, Component)
- Labeled relationships

### 4. **Smart Suggestions** 💡
- LLM generates 5 follow-up questions
- Based on actual query results
- Clickable in Streamlit UI

### 5. **Query History** 📜
- Last 5 queries saved
- Mini-graphs shown
- Natural language responses preserved

---

## 📋 Pipeline Flow

```
User Question
    ↓
[STEP 1] LLM generates Cypher query
    ↓
[STEP 2] Neo4j executes query
    ↓
[STEP 3] LLM analyzes results
    ├─ Natural language response
    ├─ Nodes to visualize
    ├─ Edges to show
    └─ Follow-up suggestions
    ↓
[STEP 4] Matplotlib draws graph
    ↓
Output: Response + Graph + Suggestions
```

---

## 🎯 Example Queries

### Simple Queries
```
What issues does MRM-488 relate to?
Show bugs that relate to other bugs
Find duplicate issues
```

### Complex Queries
```
Show me critical priority bugs that were closed but have related bugs that are still open
Find the shortest path of relationships between MRM-488 and MRM-1345
Which developer has the most bugs assigned?
```

---

## 💻 Streamlit UI Layout

```
┌─────────────────────────────────────────┐
│  🔍 GraphRAG ITS Demo                   │
├─────────────────────────────────────────┤
│  Sidebar:                                │
│  ├─ Example questions (clickable)        │
│  ├─ Clear history                        │
│  └─ About info                           │
│                                          │
│  Main Area:                              │
│  ├─ Question input                       │
│  ├─ Run button                           │
│  │                                        │
│  ├─ 💬 Natural Response (prominent)      │
│  │                                        │
│  └─ 4 Tabs:                              │
│     ├─ 📊 Graph (visualization)          │
│     ├─ 🔍 Results (JSON data)            │
│     ├─ 💡 Suggestions (clickable)        │
│     └─ 📝 Cypher (generated query)       │
│                                          │
│  History:                                │
│  └─ Last 5 queries with responses        │
└─────────────────────────────────────────┘
```

---

## 📊 Output Example

**Query:** "What issues does MRM-488 relate to?"

**💬 Natural Response:**
> I found 2 related issues for MRM-488. MRM-488 relates to MRM-615 and MRM-487, both dealing with property resolution problems in POMs.

**📝 Cypher:**
```cypher
MATCH (i:Issue {id: 'MRM-488'})-[r:RELATES_TO]-(related:Issue)
RETURN i.id AS source, type(r) AS rel, related.id AS target
LIMIT 10
```

**📊 Graph:** 3 nodes, 2 edges
- MRM-488 → MRM-615 [RELATES_TO]
- MRM-488 → MRM-487 [RELATES_TO]

**💡 Suggestions:**
1. What components are affected?
2. Who is assigned to these bugs?
3. Are there any workarounds available?
4. What is the impact on users?
5. Is there a plan to fix this issue?

---

## 🛠️ Technical Details

### LLM Model
- **Qwen/Qwen2.5-Coder-7B-Instruct**
- Temperature: 0.1-0.3 (deterministic)
- Max tokens: 512-1536

### Database
- **Neo4j** (bolt://localhost:7687)
- Dataset: SEOSS Apache Archiva
- 4,700+ issues with trace links

### Relationships
- `RELATES_TO` - Issue relates to Issue
- `DEPENDS_UPON` - Issue depends on Issue
- `DUPLICATES` - Issue duplicates Issue
- `ASSIGNED_TO` - Issue assigned to Person
- `AFFECTS` - Issue affects Component

---

## 📁 Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Streamlit web UI |
| `simple_llm_pipeline.py` | Backend LLM pipeline |
| `load_seoss_to_neo4j.py` | Data loader |
| `results/` | Generated images |

---

## 🎓 Demo Tips for Professor

1. **Start with simple query:** "What issues does MRM-488 relate to?"
2. **Show natural response:** Point out the human-readable answer
3. **Explain Cypher generation:** LLM wrote the query automatically
4. **Show graph:** Visual representation of trace links
5. **Click suggestions:** Demonstrate follow-up questions
6. **Try complex query:** "Show critical bugs that were closed but have related bugs still open"
7. **Show history:** Past queries preserved with responses

---

## ⚠️ Known Limitations

- Complex multi-relationship queries may fail (LLM generates invalid Cypher)
- Aggregation queries show disconnected nodes (no edges)
- Cycle detection queries not fully supported
- ~70% success rate on complex queries

---

## 🔧 Troubleshooting

**Neo4j not running:**
```bash
sudo systemctl start neo4j
```

**Port in use:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Module not found:**
```bash
pip install streamlit networkx matplotlib neo4j huggingface_hub
```

---

## 📞 Support

Questions? Check:
- `README_STREAMLIT.md` - Streamlit setup
- `simple_llm_pipeline.py` - Code comments
- Neo4j browser: http://localhost:7474
