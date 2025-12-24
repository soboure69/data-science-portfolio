import os
import sys

import dash
from dash import Input, Output, State, dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.recommendation_engine import RecommendationEngine


rec_engine = RecommendationEngine()
products_df = rec_engine.load_products()
rec_engine.build_features()

app = dash.Dash(__name__)
app.title = "Smart Product Recommender"

colors = {
    "primary": "#1f77b4",
    "success": "#2ca02c",
    "background": "#f8f9fa",
    "text": "#2c3e50",
}

category_options = ["All"] + sorted(products_df["category"].dropna().unique().tolist())

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Smart Product Recommender", style={"margin": 0, "color": colors["primary"]}),
                        html.P(
                            "Content-based recommendations (TF-IDF + numeric features + cosine similarity)",
                            style={"margin": "6px 0 0 0", "color": colors["text"]},
                        ),
                    ],
                    style={"flex": 1},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Total products: ", style={"fontWeight": "600"}),
                                html.Span(str(len(products_df)), style={"color": colors["primary"], "fontSize": "18px"}),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("Avg rating: ", style={"fontWeight": "600"}),
                                html.Span(f"{products_df['rating'].mean():.2f}", style={"color": colors["success"], "fontSize": "18px"}),
                            ],
                            style={"marginTop": "6px"},
                        ),
                    ],
                    style={"textAlign": "right"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "padding": "18px 20px",
                "backgroundColor": "white",
                "borderBottom": f"2px solid {colors['primary']}",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Controls", style={"color": colors["primary"], "marginTop": 0}),
                        html.Label("Product"),
                        dcc.Dropdown(
                            id="product-dropdown",
                            options=[
                                {"label": f"{row['name']} (⭐ {row['rating']:.1f})", "value": row["product_id"]}
                                for _, row in products_df.sort_values(["category", "name"]).iterrows()
                            ],
                            placeholder="Choose a product...",
                            clearable=True,
                        ),
                        html.Div(style={"height": "10px"}),
                        html.Label("Category filter (mini-exercice UX)"),
                        dcc.Dropdown(
                            id="category-filter",
                            options=[{"label": c, "value": c} for c in category_options],
                            value="All",
                            clearable=False,
                        ),
                        html.Div(style={"height": "10px"}),
                        html.Label("Max price"),
                        dcc.Slider(
                            id="max-price",
                            min=float(products_df["price"].min()),
                            max=float(products_df["price"].max()),
                            step=1,
                            value=float(products_df["price"].max()),
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(style={"height": "14px"}),
                        html.Button(
                            "Get Recommendations",
                            id="recommend-btn",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "backgroundColor": colors["primary"],
                                "color": "white",
                                "border": "none",
                                "borderRadius": "6px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(id="selected-product", style={"marginTop": "14px"}),
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "16px",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                        "width": "320px",
                    },
                ),
                html.Div(
                    [
                        html.Div(id="ux-insight", style={"marginBottom": "12px"}),
                        html.Div(id="recommendations-container", style={"marginBottom": "16px"}),
                        html.Div(
                            [
                                html.H3("Analytics", style={"color": colors["primary"], "marginTop": 0}),
                                html.Div(
                                    [
                                        dcc.Graph(id="similarity-chart", style={"flex": 1}),
                                        dcc.Graph(id="category-chart", style={"flex": 1}),
                                    ],
                                    style={"display": "flex", "gap": "14px"},
                                ),
                                dcc.Graph(id="price-chart"),
                                dcc.Graph(id="price-dist-chart"),
                            ],
                            style={
                                "backgroundColor": "white",
                                "padding": "16px",
                                "borderRadius": "10px",
                                "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                            },
                        ),
                    ],
                    style={"flex": 1},
                ),
            ],
            style={"display": "flex", "gap": "16px", "padding": "18px"},
        ),
        dcc.Store(id="recommendations-store"),
    ],
    style={"backgroundColor": colors["background"], "minHeight": "100vh"},
)


@app.callback(
    [Output("selected-product", "children"), Output("recommendations-store", "data")],
    Input("recommend-btn", "n_clicks"),
    [State("product-dropdown", "value"), State("category-filter", "value"), State("max-price", "value")],
    prevent_initial_call=True,
)
def update_recommendations(n_clicks, selected_product, category, max_price):
    if not selected_product:
        return html.Div("Please select a product."), None

    selected = rec_engine.get_product(selected_product)

    recs = rec_engine.recommend(
        selected_product,
        n_recommendations=6,
        category=category,
        max_price=max_price,
    )

    data = {
        "selected": selected,
        "category": category,
        "max_price": float(max_price) if max_price is not None else None,
        "recommendations": [
            {"product_id": r.product_id, "score": float(r.score)}
            for r in recs
        ],
    }

    selected_html = html.Div(
        [
            html.H4(selected["name"], style={"margin": "10px 0 6px 0"}),
            html.Div(f"Category: {selected['category']}"),
            html.Div(f"Price: ${float(selected['price']):.2f}"),
            html.Div(f"Rating: {float(selected['rating']):.1f} (reviews: {int(selected['num_reviews'])})"),
        ],
        style={"padding": "10px", "backgroundColor": "#ffffff", "borderRadius": "8px", "border": "1px solid #e9ecef"},
    )

    return selected_html, data


@app.callback(
    Output("recommendations-container", "children"),
    Input("recommendations-store", "data"),
)
def update_cards(data):
    if not data or not data.get("recommendations"):
        if not data:
            return html.Div("Select a product and click 'Get Recommendations' to see results.")

        category = data.get("category")
        max_price = data.get("max_price")
        selected = data.get("selected", {})

        return html.Div(
            [
                html.Div(
                    "No recommendations match the current filters.",
                    style={"fontWeight": "600", "marginBottom": "6px"},
                ),
                html.Div(
                    [
                        html.Div(f"Selected: {selected.get('name', '')}"),
                        html.Div(f"Category filter: {category}"),
                        html.Div(f"Max price: ${float(max_price):.2f}" if max_price is not None else "Max price: -"),
                    ],
                    style={"color": colors["text"], "fontSize": "13px"},
                ),
            ],
            style={
                "backgroundColor": "white",
                "border": "1px solid #e9ecef",
                "borderRadius": "10px",
                "padding": "12px",
            },
        )

    category = data.get("category")
    max_price = data.get("max_price")

    cards = []
    cards.append(
        html.Div(
            [
                html.Div(
                    f"Recommendations: {len(data['recommendations'])} | Category: {category} | Max price: ${float(max_price):.2f}",
                    style={"fontSize": "13px", "color": colors["text"]},
                )
            ],
            style={"marginBottom": "10px"},
        )
    )
    for rec in data["recommendations"]:
        row = products_df[products_df["product_id"] == rec["product_id"]].iloc[0]
        cards.append(
            html.Div(
                [
                    html.H4(row["name"], style={"margin": "0 0 6px 0"}),
                    html.Div(f"Category: {row['category']}"),
                    html.Div(f"Price: ${float(row['price']):.2f}"),
                    html.Div(f"Similarity: {float(rec['score']):.2%}", style={"fontWeight": "600", "color": colors["success"]}),
                ],
                style={
                    "backgroundColor": "white",
                    "border": f"1px solid #e9ecef",
                    "borderRadius": "10px",
                    "padding": "12px",
                    "marginBottom": "10px",
                },
            )
        )

    return html.Div(cards)


@app.callback(
    Output("ux-insight", "children"),
    Input("recommendations-store", "data"),
)
def update_ux_insight(data):
    if not data:
        return html.Div(
            "Tip: use the category filter to trade off diversity vs relevance.",
            style={
                "backgroundColor": "white",
                "border": "1px solid #e9ecef",
                "borderRadius": "10px",
                "padding": "12px",
                "color": colors["text"],
            },
        )

    selected = data.get("selected", {})
    category = data.get("category")
    rec_ids = [r["product_id"] for r in data.get("recommendations", [])]
    rec_df = products_df[products_df["product_id"].isin(rec_ids)]

    total_categories = int(products_df["category"].nunique())
    rec_categories = int(rec_df["category"].nunique()) if not rec_df.empty else 0

    explanation = (
        "When you set a category filter, recommendations become more *focused* (higher relevance within the same category) "
        "but usually less *diverse* (fewer different categories)."
    )

    return html.Div(
        [
            html.Div("UX mini-exercise: Category filter impact", style={"fontWeight": "600", "marginBottom": "6px"}),
            html.Div(explanation, style={"fontSize": "13px", "marginBottom": "8px", "color": colors["text"]}),
            html.Div(
                [
                    html.Div(f"Selected: {selected.get('name', '')}", style={"fontSize": "13px"}),
                    html.Div(f"Filter: {category}", style={"fontSize": "13px"}),
                    html.Div(
                        f"Diversity: {rec_categories}/{total_categories} categories in current recommendations",
                        style={"fontSize": "13px"},
                    ),
                ],
                style={"color": colors["text"]},
            ),
        ],
        style={
            "backgroundColor": "white",
            "border": "1px solid #e9ecef",
            "borderRadius": "10px",
            "padding": "12px",
        },
    )


@app.callback(Output("similarity-chart", "figure"), Input("recommendations-store", "data"))
def update_similarity_chart(data):
    if not data or not data.get("recommendations"):
        return go.Figure()

    rows = []
    for rec in data["recommendations"]:
        row = products_df[products_df["product_id"] == rec["product_id"]].iloc[0]
        rows.append({"name": row["name"], "score": rec["score"]})

    df = pd.DataFrame(rows)
    fig = px.bar(df, x="name", y="score", title="Similarity scores")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(tickformat=".0%")
    return fig


@app.callback(Output("category-chart", "figure"), Input("recommendations-store", "data"))
def update_category_chart(data):
    if not data or not data.get("recommendations"):
        return go.Figure()

    rec_ids = [r["product_id"] for r in data["recommendations"]]
    df = products_df[products_df["product_id"].isin(rec_ids)][["category"]]
    fig = px.pie(df, names="category", title="Recommended categories")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


@app.callback(Output("price-chart", "figure"), Input("recommendations-store", "data"))
def update_price_chart(data):
    if not data:
        return go.Figure()

    selected_id = data["selected"]["product_id"]
    rec_ids = [r["product_id"] for r in data.get("recommendations", [])]

    subset = products_df[products_df["product_id"].isin([selected_id] + rec_ids)][
        ["product_id", "name", "price"]
    ].copy()
    subset["type"] = subset["product_id"].apply(lambda pid: "selected" if pid == selected_id else "recommended")

    fig = px.scatter(
        subset,
        x="price",
        y="name",
        color="type",
        title="Price comparison",
        color_discrete_map={"selected": colors["primary"], "recommended": colors["success"]},
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    return fig


@app.callback(Output("price-dist-chart", "figure"), Input("recommendations-store", "data"))
def update_price_distribution(data):
    fig = go.Figure()
    if not data:
        return fig

    selected_price = float(data["selected"]["price"])
    rec_ids = [r["product_id"] for r in data.get("recommendations", [])]
    rec_prices = products_df[products_df["product_id"].isin(rec_ids)]["price"].astype(float).tolist()

    fig.add_trace(
        go.Histogram(
            x=products_df["price"].astype(float),
            name="All products",
            opacity=0.45,
            marker_color="#b0bec5",
        )
    )

    if rec_prices:
        fig.add_trace(
            go.Histogram(
                x=rec_prices,
                name="Recommended",
                opacity=0.65,
                marker_color=colors["success"],
            )
        )

    fig.add_vline(x=selected_price, line_width=3, line_dash="dash", line_color=colors["primary"])
    fig.update_layout(
        title="Price distribution (context)",
        barmode="overlay",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="Price")
    fig.update_yaxes(title="Count")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
