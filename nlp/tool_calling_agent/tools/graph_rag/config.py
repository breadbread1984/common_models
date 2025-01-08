#!/usr/bin/python3

tool_name = "招股说明书"
tool_description = "当你有招股说明书相关问题，可以调用这个工具"
input_description = "招股说明书相关的问题"
output_description = "招股说明书相关的答案"

neo4j_host = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "neo4j"
neo4j_db = "neo4j"

node_types = ['Person', 'Location', 'Organization', 'Product', 'Time', 'Event']
rel_types = [
  ('Person', 'CEO_of', 'Organization'),
  ('Person', 'Birth_In', 'Location'),
  ('Person', 'Work_For', 'Organization'),
  ('Person', 'Work_Time', 'Time'),
  ('Person', 'Participate_In', 'Event'),
  ('Person', 'Member_Of', 'Organization'),
  ('Event', 'Happened_Before', 'Event'),
  ('Organization', 'Based_In', 'Location'),
  ('Product', 'Launches_From', 'Location'),
  ('Organization', 'Develops', 'Product'),
  ('Organization', 'Collaborates_With', 'Organization'),
  ('Product', 'Associated_With', 'Organization')
]
examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {{title: 'Casino'}})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {{title: 'Schindler\\'s List'}})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]
