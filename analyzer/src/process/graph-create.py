#%%
import pandas as pd
from tqdm.auto import tqdm
import pickle
import numpy as np
import random
import json

df = pd.read_csv('../data/clustered.csv.gz', index_col="Id")
filtered = {}
with open('../data/filtered_dict.pt', 'rb') as fil:
    filtered = pickle.load(fil)


#%% neo4j filling
def fill_neo4j():
    from neo4j import GraphDatabase

    def add_nodes(tx, entity, art_count):
        tx.run("CREATE (ne:NamedEntity {name: $entity}) ", entity=entity)
        for article_id, count in art_count:
            row = df.loc[article_id]
            tx.run("""
            MATCH (ne:NamedEntity {name: $entity})
            WITH ne
            MERGE (p:Publisher {name: $p})
            MERGE (a:Article {url: $url})
            SET  a.name = $name
            MERGE (p)-[:PUBLISHED {date: $published}]->(a)
            MERGE (a)-[:HAS {count: $count}]->(ne)
            """,
                   entity=entity,
                   p=row.SiteName,
                   name=row.Title,
                   url=row.Url,
                   published=row.Created,
                   count=count)

    def constraint(tx):
        tx.run("CREATE CONSTRAINT ON (a:Article) ASSERT a.url IS UNIQUE;")

    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "admin"),
                                  encrypted=False)

    with driver.session() as session:
        # session.write_transaction(constraint)
        for entity, art_count in tqdm(filtered.items()):
            session.write_transaction(add_nodes, entity, art_count)

    driver.close()


#%% JSON Graph


def create_json_graph(top_num=30, top_per=10):
    file_name = f"../data/graph-{top_num}-{top_per}.json"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)

    def gen_id(start: str):
        i = 0
        while True:
            yield f"{start}{i}"
            i += 1

    sizes = pd.Series({k: len(v) for k, v in filtered.items()})

    top_n = list(sizes.nlargest(top_num).index)

    reduced = dict(
        (k, sorted(filtered[k], key=lambda x: x[1])[-top_per:]) for k in top_n)
    included = {i[0] for v in reduced.values() for i in v}

    node_id_it = iter(gen_id("n"))
    edge_id_it = iter(gen_id("e"))

    def to_graph_json():
        nodes = []
        edges = []

        entities = {
            e: {
                "id": next(node_id_it),
                "label": e,
                "size": 3,
                "color": "#4fbddc",
                "x": random.randint(0, 300),
                "y": random.randint(0, 200),
            }
            for e in reduced.keys()
        }

        articles = {
            i: {
                "id": i,
                "label": df.loc[i].Title,
                "size": 3,
                "color": "#f37839",
                "x": random.randint(0, 300),
                "y": random.randint(0, 200),
            }
            for i in included
        }

        for entity, art_count in tqdm(reduced.items()):
            for article_id, count in art_count:
                edges.append({
                    "id": next(edge_id_it),
                    "label": count,
                    "target": entities[entity]["id"],
                    "source": article_id
                })

        nodes = list(entities.values()) + list(articles.values())

        return {
            'nodes': nodes,
            'edges': edges,
        }

    graph = to_graph_json()
    with open(file_name, 'wt') as f:
        json.dump(graph, f, cls=NpEncoder)


#%%
if __name__ == "__main__":
    fill_neo4j()