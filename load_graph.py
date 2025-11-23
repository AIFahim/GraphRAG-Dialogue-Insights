#!/usr/bin/env python3
"""
Load knowledge graph into Neo4j
"""
from neo4j import GraphDatabase
import time

# Configuration
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

def load_graph():
    """Load the improved knowledge graph"""

    # Wait a bit for Neo4j to fully start
    print("Waiting for Neo4j to be ready...")
    time.sleep(5)

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Read the Cypher script
    print("Reading knowledge graph script...")
    with open('src/knowledge-graph/improved_version_knowledge_graph.txt', 'r') as f:
        cypher_script = f.read()

    # Split by semicolons and newlines to get individual statements
    # The script uses MERGE statements which are safe to run individually
    statements = []
    current_statement = []

    for line in cypher_script.split('\n'):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('//'):
            continue

        current_statement.append(line)

        # If line ends with closing parenthesis or bracket, it's likely end of statement
        if line.endswith(')') or line.endswith('}'):
            statements.append(' '.join(current_statement))
            current_statement = []

    # Execute statements
    print(f"\nExecuting {len(statements)} statements...")
    with driver.session() as session:
        for i, statement in enumerate(statements, 1):
            try:
                session.run(statement)
                if i % 10 == 0:
                    print(f"  Executed {i}/{len(statements)} statements...")
            except Exception as e:
                print(f"  Warning on statement {i}: {e}")
                print(f"  Statement: {statement[:100]}...")

        print("\nKnowledge graph loaded successfully!")

        # Show statistics
        print("\n" + "="*60)
        print("GRAPH STATISTICS")
        print("="*60)

        result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
        print("\nNodes:")
        for record in result:
            print(f"  - {record['type']}: {record['count']}")

        result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC")
        print("\nRelationships:")
        for record in result:
            print(f"  - {record['type']}: {record['count']}")

    driver.close()
    print("\nDone!")

if __name__ == "__main__":
    load_graph()
