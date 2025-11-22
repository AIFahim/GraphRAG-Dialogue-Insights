# Setup Guide - ITS Demo

## 📊 Data Flow

```
archiva.sqlite3 (SQLite)
        ↓
load_seoss_to_neo4j.py (run once)
        ↓
Neo4j Database (graph)
        ↓
streamlit_app.py (query anytime)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_its_demo.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

Get HuggingFace token: https://huggingface.co/settings/tokens

### 3. Start Neo4j
```bash
# Make sure Neo4j is running on bolt://localhost:7687
sudo systemctl start neo4j
# Or use Docker:
docker run -p 7474:7474 -p 7687:7687 neo4j
```

### 4. Load Data (One Time Only)
```bash
cd data/seoss_extracted
python load_seoss_to_neo4j.py
```

This loads `archiva.sqlite3` into Neo4j (takes ~2 minutes).

### 5. Run Demo
```bash
cd ../..  # Back to root
streamlit run streamlit_app.py
```

Open: http://localhost:8501

---

## 🗄️ Data Source

**File:** `data/seoss_extracted/archiva.sqlite3` (21MB)

**Contents:**
- 4,700+ issues from Apache Maven Archiva project
- Trace links: RELATES_TO, DEPENDS_UPON, DUPLICATES, etc.
- Developers, components, assignments

**Schema:**
- `issues` table → Issue nodes
- `issue_link` table → Relationship edges
- `person` table → Developer nodes
- `component` table → Component nodes

---

## ✅ Verification

After loading, check Neo4j browser: http://localhost:7474

Run:
```cypher
MATCH (i:Issue) RETURN count(i) as total_issues
```

Should show ~4,700 issues.

---

## 🔧 Troubleshooting

**No data in Neo4j?**
- Re-run: `python data/seoss_extracted/load_seoss_to_neo4j.py`

**Import error?**
- Install: `pip install python-dotenv`

**HF_TOKEN error?**
- Set in `.env` file or: `export HF_TOKEN="your_token"`

**Neo4j connection failed?**
- Check Neo4j is running: `systemctl status neo4j`
- Check credentials in `.env`
