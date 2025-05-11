from igraph import Graph
import random
import numpy as np
from config import DATA_PATH, MODEL_NAME, OUTPUT_HTML
from preprocessing import preprocess_posts, build_hashtag_edges, build_description_edges
from embedding_and_clustering import analyze_communities
from visualization import summarize_and_visualize_communities

df_post = preprocess_posts(DATA_PATH)

edges_hashtag = build_hashtag_edges(df_post)
edges_description, valid_embeddings, valid_post_ids = build_description_edges(df_post, model_name=MODEL_NAME)

post_ids = df_post['post_id'].astype(str).tolist()

g = Graph()
g.add_vertices(post_ids)
g.add_edges(edges_description)
g.add_edges([(str(e[0]), str(e[1])) for e in edges_hashtag])

g.vs['like_count'] = df_post['like_count'].fillna(0).astype(int).tolist()
g.vs['save_count'] = df_post['save_count'].fillna(0).astype(int).tolist()

random.seed(42)
np.random.seed(42)

communities = g.community_multilevel()
g.vs['community'] = communities.membership

community_ids = set(g.vs['community'])
community_sizes = {cid: len([v for v in g.vs if v['community'] == cid]) for cid in community_ids}
filtered_communities = {cid: size for cid, size in community_sizes.items() if size >= 70}
sorted_communities = sorted(filtered_communities.items(), key=lambda x: x[1], reverse=True)

valid_community_ids = set(filtered_communities.keys())
g.vs['community'] = [cid if cid in valid_community_ids else -1 for cid in g.vs['community']]

print(f"Detected {len(filtered_communities)} Communities")
print("Sorted Communities by Size (Filtered):")
for cid, size in sorted_communities:
    print(f"Community {cid}: {size} nodes")

community_analysis = analyze_communities(
    g,
    df_post,
    valid_embeddings,
    valid_post_ids,
    sorted_communities
)

final_df = summarize_and_visualize_communities(
    community_analysis,
    g,
    df_post,
    output_filename=OUTPUT_HTML
)

# 결과 DataFrame 출력
print(final_df.head())
