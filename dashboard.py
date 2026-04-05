from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from datetime import date

# -----------------------------
# Load prediction CSVs
# -----------------------------
files = {
    "track": "data/track_predictions.csv",
    "court1": "data/court1_predictions.csv",
    "court2": "data/court2_predictions.csv",
    "court3": "data/court3_predictions.csv",
    "court4": "data/court4_predictions.csv",
    "entire room": "data/multi_predictions.csv"   # multipurpose room
}

# Load all CSVs into a dictionary of DataFrames
dfs = {}
for key, path in files.items():
    df = pd.read_csv(path)

    # Parse time_bin into datetime
    df["time_bin"] = pd.to_datetime(df["time_bin"])

    # Extract date and time separately
    df["date"] = df["time_bin"].dt.date
    df["time"] = df["time_bin"].dt.time

    dfs[key] = df

# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__)
app.title = "SDC Multipurpose Room Activity Tracker"

app.layout = html.Div([

    # Top row: MTU logo (left) and UIL enterprise logo (right)
    html.Div([
        html.Img(
            src='assets/MTU_Logo.png',
            style={'height': '70px', 'display': 'inline-block'}
        ),
        html.Img(
            src='assets/UIL_Logo.png',
            style={'height': '120px', 'display': 'inline-block'}  # enlarged for balance
        ),
    ], style={
        "width": "100%",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "marginBottom": "10px"
    }),

    # Title centered
    html.H1(
        "SDC Multipurpose Room Activity Tracker",
        style={"textAlign": "center", "marginBottom": "20px"}
    ),

    # Main content row (left: images + controls, right: graph)
    html.Div([

        # LEFT COLUMN
        html.Div([
            html.Img(
                src='assets/SDC_Multipurpose_BirdsEye.jpeg',
                style={'width': '100%', 'marginBottom': '20px'}
            ),

            html.H3("Locations"),
            dcc.Dropdown(
                id="location-dropdown",
                options=[
                    {"label": "Multipurpose Room", "value": "entire room"},
                    {"label": "Court 1", "value": "court1"},
                    {"label": "Court 2", "value": "court2"},
                    {"label": "Court 3", "value": "court3"},
                    {"label": "Court 4", "value": "court4"},
                    {"label": "Track", "value": "track"},
                ],
                placeholder="Select a location",
                style={"width": "90%", "marginBottom": "20px"}
            ),

            html.H3("Date"),
            dcc.DatePickerSingle(
                id="date-picker",
                min_date_allowed=date(2026, 1, 1),
                max_date_allowed=date(2027, 12, 31),
                date=date(2026, 4, 6),
                display_format="YYYY-MM-DD",
                style={"width": "90%"}
            ),
        ],
        style={
            "width": "30%",
            "display": "inline-block",
            "verticalAlign": "top",
            "padding": "10px"
        }),

        # RIGHT COLUMN (Graph)
        html.Div([
            dcc.Graph(id="activity-graph")
        ],
        style={
            "width": "68%",
            "display": "inline-block",
            "padding": "10px"
        }),

    ],
    style={"display": "flex", "justifyContent": "space-between"}),

])

# -----------------------------
# Callback: update graph
# -----------------------------
@app.callback(
    Output("activity-graph", "figure"),
    Input("location-dropdown", "value"),
    Input("date-picker", "date")
)
def update_graph(location, selected_date):

    if location is None:
        return px.bar(title="Select a location to view predictions")

    df = dfs[location]

    # Filter by selected date
    selected_date = pd.to_datetime(selected_date).date()
    filtered = df[df["date"] == selected_date]

    if filtered.empty:
        return px.bar(title=f"No prediction data for {selected_date}")

    # Build bar graph (NO TITLE)
    fig = px.bar(
        filtered,
        x="time",
        y="prediction"
    )

    # -----------------------------
    # Michigan Tech Theme (Gold + Black)
    # -----------------------------
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Predicted People Count",
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            size=14,
            color="#000000"
        )
    )

    fig.update_traces(
        marker_color="#FFCD00",        # Tech Gold
        marker_line_color="#000000",   # Black outline
        marker_line_width=1.2
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
