import pandas as pd
import re
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


def split_hashtags(hashtag_str):
    if isinstance(hashtag_str, str):
        if '#' in hashtag_str:
            return [tag.strip() for tag in hashtag_str.split('#') if tag.strip()]
        elif '[' in hashtag_str:
            return [tag.strip().strip("'").strip() for tag in hashtag_str.strip('[]').split(',') if tag.strip()]
    return []

def clean_hashtags(hashtag_list):
    if not isinstance(hashtag_list, list):
        return []

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002500-\U00002BEF" u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" u"\U0001f926-\U0001f937" u"\U00010000-\U0010FFFF"
        u"\u2640-\u2642" u"\u2600-\u2B55" u"\u200D" u"\u23CF" u"\u23E9" u"\u231A"
        u"\uFE0F" u"\u3030"
        "]+", flags=re.UNICODE
    )

    cleaned_hashtags = []
    for hashtag in hashtag_list:
        cleaned = emoji_pattern.sub("", hashtag.lower())
        cleaned_hashtags.append(cleaned)

    return list(set(cleaned_hashtags))

def remove_hashtags(hashtags, keywords):
    return [tag for tag in hashtags if not any(keyword in tag.lower() for keyword in keywords)]


def preprocess_posts(filepath):
    df_post = pd.read_csv(filepath)
    columns = ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0']
    df_post = df_post.drop(labels=columns, axis=1, errors='ignore')

    # 해시태그 처리
    df_post['hashtag_list'] = df_post['hashtags'].apply(split_hashtags)
    df_post['cleaned_hashtags'] = df_post['hashtag_list'].apply(clean_hashtags)

    remove_keywords = ['fyp', 'foryou', 'viral','fyy','tiktok','trending','trend', 'funny','pov',
                       'xuhong','parati','meme','foru','haha', 'for you page','for you',
                       'foryoupage', 'tiktokfamous', 'viralpost', 'video', 'justforfun', 'follow']

    df_post['filtered_hashtags'] = df_post['cleaned_hashtags'].apply(
        lambda tags: [tag for tag in tags if not any(keyword in tag for keyword in remove_keywords)] if tags else []
    )

    return df_post


def build_hashtag_edges(df_post):
    post_hashtags = df_post[['post_id', 'filtered_hashtags']].dropna(subset=['filtered_hashtags'])
    hashtag_to_posts = defaultdict(list)
    post_to_hashtags = defaultdict(set)

    for _, row in post_hashtags.iterrows():
        post_id = row['post_id']
        hashtags = row['filtered_hashtags']
        post_to_hashtags[post_id] = set(hashtags)
        for tag in hashtags:
            hashtag_to_posts[tag].append(post_id)

    edges_hashtag_limited = set()

    for posts in hashtag_to_posts.values():
        for i in range(len(posts)):
            for j in range(i + 1, len(posts)):
                common_hashtags = post_to_hashtags[posts[i]] & post_to_hashtags[posts[j]]
                if len(common_hashtags) >= 3:
                    edges_hashtag_limited.add((posts[i], posts[j]))

    return list(edges_hashtag_limited)


def build_description_edges(df_post, model_name="Twitter/twhin-bert-base"):
    post_ids = df_post['post_id'].astype(str).tolist()
    descriptions = df_post['description'].tolist()
    filtered_hashtags = df_post['filtered_hashtags'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []

    for i, desc in enumerate(descriptions):
        if pd.notna(desc) and desc.strip():
            inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            embeddings.append(outputs.pooler_output.cpu().detach().numpy().flatten())
        else:
            hashtags = filtered_hashtags[i]
            if hashtags:
                hashtags_str = " ".join(hashtags)
                inputs = tokenizer(hashtags_str, return_tensors="pt", truncation=True, padding=True, max_length=128)
                outputs = model(**inputs)
                embeddings.append(outputs.pooler_output.cpu().detach().numpy().flatten())
            else:
                embeddings.append(None)

    valid_embeddings = np.array([emb for emb in embeddings if emb is not None])
    similarities = cosine_similarity(valid_embeddings)

    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    valid_post_ids = [post_ids[idx] for idx in valid_indices]

    node_edge_count_description = defaultdict(int)
    edges_description_limited = []

    for i in range(len(similarities)):
        potential_edges = []
        for j in range(i + 1, len(similarities)):
            if similarities[i, j] > 0.95:
                potential_edges.append((similarities[i, j], valid_post_ids[i], valid_post_ids[j]))

        potential_edges.sort(reverse=True, key=lambda x: x[0])

        for _, post1, post2 in potential_edges:
            if node_edge_count_description[post1] < 3 and node_edge_count_description[post2] < 3:
                edges_description_limited.append((post1, post2))
                node_edge_count_description[post1] += 1
                node_edge_count_description[post2] += 1

    return edges_description_limited, valid_embeddings, valid_post_ids
