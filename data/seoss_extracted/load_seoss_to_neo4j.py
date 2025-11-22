"""
Load SEOSS Dataset into Neo4j

Converts SQLite tables into Neo4j graph:
- Issues become nodes
- issue_link rows become relationships
- change_set_link rows become relationships
- etc.
"""
import sqlite3
from neo4j import GraphDatabase

# Configuration
SQLITE_DB = "archiva.sqlite3"
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def load_to_neo4j():
    """Load SEOSS data into Neo4j"""

    # Connect to SQLite
    print("Connecting to SQLite...")
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    neo4j_driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    with neo4j_driver.session() as session:

        # 1. Clear existing SEOSS data
        print("\n[1/7] Clearing existing SEOSS data...")
        session.run("MATCH (n) WHERE n:Issue OR n:Person OR n:Commit OR n:Component DETACH DELETE n")

        # 2. Create Issue nodes
        print("\n[2/7] Creating Issue nodes...")
        cursor.execute("""
            SELECT issue_id, type, summary, description, priority, status,
                   resolution, assignee, reporter, created_date
            FROM issue
        """)
        issues = cursor.fetchall()

        for issue in issues:
            session.run("""
                CREATE (i:Issue {
                    id: $id,
                    type: $type,
                    summary: $summary,
                    description: $description,
                    priority: $priority,
                    status: $status,
                    resolution: $resolution,
                    created_date: $created_date
                })
            """, {
                'id': issue['issue_id'],
                'type': issue['type'],
                'summary': issue['summary'],
                'description': issue['description'][:500] if issue['description'] else '',
                'priority': issue['priority'],
                'status': issue['status'],
                'resolution': issue['resolution'],
                'created_date': issue['created_date']
            })
        print(f"   Created {len(issues)} Issue nodes")

        # 3. Create Person nodes (from assignees and reporters)
        print("\n[3/7] Creating Person nodes...")
        cursor.execute("""
            SELECT DISTINCT assignee as name FROM issue WHERE assignee != 'None'
            UNION
            SELECT DISTINCT reporter as name FROM issue WHERE reporter != 'None'
        """)
        persons = cursor.fetchall()

        for person in persons:
            session.run("""
                MERGE (p:Person {name: $name})
            """, {'name': person['name']})
        print(f"   Created {len(persons)} Person nodes")

        # 4. Create Component nodes
        print("\n[4/7] Creating Component nodes...")
        cursor.execute("SELECT DISTINCT component FROM issue_component")
        components = cursor.fetchall()

        for comp in components:
            session.run("""
                MERGE (c:Component {name: $name})
            """, {'name': comp['component']})
        print(f"   Created {len(components)} Component nodes")

        # 5. Create Issue-to-Issue relationships (TRACE LINKS!)
        print("\n[5/7] Creating Issue-to-Issue relationships (TRACE LINKS)...")
        cursor.execute("""
            SELECT source_issue_id, target_issue_id, outward_label
            FROM issue_link
            WHERE outward_label IS NOT NULL
        """)
        issue_links = cursor.fetchall()

        link_counts = {}
        for link in issue_links:
            # Normalize relationship type
            rel_type = link['outward_label'].upper().replace(' ', '_').replace('-', '_')

            session.run(f"""
                MATCH (source:Issue {{id: $source_id}})
                MATCH (target:Issue {{id: $target_id}})
                CREATE (source)-[:{rel_type}]->(target)
            """, {
                'source_id': link['source_issue_id'],
                'target_id': link['target_issue_id']
            })

            link_counts[rel_type] = link_counts.get(rel_type, 0) + 1

        print(f"   Created {len(issue_links)} Issue-to-Issue relationships:")
        for rel_type, count in sorted(link_counts.items()):
            print(f"     - {rel_type}: {count}")

        # 6. Create Issue-to-Person relationships
        print("\n[6/7] Creating Issue-to-Person relationships...")

        # ASSIGNED_TO
        cursor.execute("""
            SELECT issue_id, assignee
            FROM issue
            WHERE assignee != 'None'
        """)
        assignments = cursor.fetchall()

        for assign in assignments:
            session.run("""
                MATCH (i:Issue {id: $issue_id})
                MATCH (p:Person {name: $person_name})
                CREATE (i)-[:ASSIGNED_TO]->(p)
            """, {
                'issue_id': assign['issue_id'],
                'person_name': assign['assignee']
            })
        print(f"   Created {len(assignments)} ASSIGNED_TO relationships")

        # REPORTED_BY
        cursor.execute("""
            SELECT issue_id, reporter
            FROM issue
            WHERE reporter != 'None'
        """)
        reports = cursor.fetchall()

        for report in reports:
            session.run("""
                MATCH (i:Issue {id: $issue_id})
                MATCH (p:Person {name: $person_name})
                CREATE (i)-[:REPORTED_BY]->(p)
            """, {
                'issue_id': report['issue_id'],
                'person_name': report['reporter']
            })
        print(f"   Created {len(reports)} REPORTED_BY relationships")

        # 7. Create Issue-to-Component relationships
        print("\n[7/7] Creating Issue-to-Component relationships...")
        cursor.execute("SELECT issue_id, component FROM issue_component")
        comp_links = cursor.fetchall()

        for link in comp_links:
            session.run("""
                MATCH (i:Issue {id: $issue_id})
                MATCH (c:Component {name: $comp_name})
                CREATE (i)-[:AFFECTS]->(c)
            """, {
                'issue_id': link['issue_id'],
                'comp_name': link['component']
            })
        print(f"   Created {len(comp_links)} AFFECTS relationships")

        # Final statistics
        print("\n" + "="*60)
        print("GRAPH LOADED SUCCESSFULLY!")
        print("="*60)

        result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
        print("\nNodes:")
        for record in result:
            print(f"  - {record['type']}: {record['count']}")

        result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC")
        print("\nRelationships:")
        for record in result:
            print(f"  - {record['type']}: {record['count']}")

    # Close connections
    sqlite_conn.close()
    neo4j_driver.close()


if __name__ == "__main__":
    load_to_neo4j()
