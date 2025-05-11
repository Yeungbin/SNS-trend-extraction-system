import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from igraph import Graph


class TransformerBlock(torch.nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        query = query + attn_output
        ffn_output = self.ffn(query)
        query = query + ffn_output
        return query


def apply_transformer_blocks(query_vectors, key_value_vectors, num_blocks=3, input_dim=64):
    query_vectors = torch.tensor(query_vectors, dtype=torch.float32).unsqueeze(0)
    key_value_vectors = torch.tensor(key_value_vectors, dtype=torch.float32).unsqueeze(0)

    transformer_blocks = [TransformerBlock(input_dim) for _ in range(num_blocks)]
    for block in transformer_blocks:
        query_vectors = block(query_vectors, key_value_vectors, key_value_vectors)
    return query_vectors.squeeze(0).detach().numpy()


def get_closest_descriptions(rep_vector, community_vectors, descriptions, hashtags, top_k=3):
    similarities = cosine_similarity([rep_vector], community_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(descriptions[idx], hashtags[idx]) for idx in top_indices]


def reduce_embeddings(valid_embeddings):
    tsne = TSNE(n_components=64, random_state=42, method='exact')
    return tsne.fit_transform(valid_embeddings)


def analyze_communities(g, df_post, valid_embeddings, valid_post_ids, sorted_communities):
    reduced_embeddings = reduce_embeddings(valid_embeddings)
    valid_indices = [i for i, _ in enumerate(valid_embeddings)]

    community_analysis = []

    for cid, size in sorted_communities:
        node_indices = [
            valid_indices.index(idx) for idx, v in enumerate(g.vs) if v['community'] == cid and idx in valid_indices
        ]

        if not node_indices:
            continue

        community_vectors = reduced_embeddings[node_indices]
        community_descriptions = [df_post.iloc[valid_post_ids.index(g.vs[idx]['name'])]['description'] for idx in node_indices]
        community_hashtags = [df_post.iloc[valid_post_ids.index(g.vs[idx]['name'])]['filtered_hashtags'] for idx in node_indices]
        community_likes = [df_post.iloc[valid_post_ids.index(g.vs[idx]['name'])]['like_count'] for idx in node_indices]

        total_likes = sum(community_likes)

        updated_vectors = apply_transformer_blocks(
            query_vectors=community_vectors,
            key_value_vectors=community_vectors,
            num_blocks=3,
            input_dim=64
        )

        representative_vector = np.mean(updated_vectors, axis=0)

        closest_descriptions = get_closest_descriptions(
            representative_vector, updated_vectors, community_descriptions, community_hashtags, top_k=10
        )

        community_data = {
            "Community": cid,
            "Size": size,
            "Total Likes": total_likes,
            "Descriptions and Hashtags": [{"description": desc, "hashtags": tags} for desc, tags in closest_descriptions]
        }
        community_analysis.append(community_data)

    return community_analysis
