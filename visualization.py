import pandas as pd
import plotly.graph_objects as go

def summarize_and_visualize_communities(
    community_analysis_results,
    graph,
    df_post,
    output_filename="tiktok_posts_visualization.html"
):
    """
    커뮤니티 분석 결과를 바탕으로 plotly로 시각화 및 HTML 저장

    Parameters:
        community_analysis_results (list of dict): 커뮤니티 분석 결과
        graph (igraph.Graph): 커뮤니티 포함 그래프
        df_post (pd.DataFrame): TikTok 원본 데이터 (blue_badge 포함)
        output_filename (str): 저장할 HTML 파일 경로
    """

    # 블루뱃지 여부 그래프에 추가
    if "blue_badge" not in graph.vs.attributes():
        blue_badge_mapping = dict(zip(df_post["post_id"].astype(str), df_post["blue_badge"]))
        graph.vs["blue_badge"] = [blue_badge_mapping.get(v["name"], 0) for v in graph.vs]

    df_analysis = pd.DataFrame(community_analysis_results)

    valid_nodes = [v.index for v in graph.vs if v["community"] != -1]
    subgraph = graph.subgraph(valid_nodes)

    predefined_colors = ["orange", "green", "blue", "yellow", "red", "purple"]
    community_ids = list(set(subgraph.vs["community"]))
    community_colors = {
        cid: predefined_colors[i % len(predefined_colors)]
        for i, cid in enumerate(community_ids)
    }

    layout = subgraph.layout("fr")
    x_coords = [layout[i][0] * 10 for i in range(len(subgraph.vs))]
    y_coords = [layout[i][1] * 10 for i in range(len(subgraph.vs))]

    fig = go.Figure()

    # 엣지 추가
    edge_x = []
    edge_y = []
    for edge in subgraph.es:
        source = edge.source
        target = edge.target
        edge_x.extend([x_coords[source], x_coords[target], None])
        edge_y.extend([y_coords[source], y_coords[target], None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="lightgrey"),
            hoverinfo="none",
        )
    )

    # 노드 추가
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    node_size = []

    for i, vertex in enumerate(subgraph.vs):
        node_x.append(x_coords[i])
        node_y.append(y_coords[i])
        node_color.append(community_colors[vertex["community"]])
        node_text.append(
            f"Community: {vertex['community']}<br>"
            f"Post ID: {vertex['name']}<br>"
            f"Likes: {vertex['like_count']}<br>"
            f"Saves: {vertex['save_count']}"
        )
        if vertex["blue_badge"] == 1:
            node_size.append(25)
        else:
            node_size.append(15)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=[
                    1 if vertex["blue_badge"] == 1 else 0.4 for vertex in subgraph.vs
                ],
                line=dict(width=1, color="black"),
            ),
            text=node_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="TIKTOK Posts Visualization",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        width=1200,
    )

    fig.write_html(output_filename)
    print(f"시각화 결과가 '{output_filename}'에 저장되었습니다.")

    return df_analysis
