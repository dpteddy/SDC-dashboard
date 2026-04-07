from dash import Dash, html, dcc, Input, Output, State, callback_context
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
    "entire room": "data/multi_predictions.csv"
}

dfs = {}
for key, path in files.items():
    df = pd.read_csv(path)
    df["time_bin"] = pd.to_datetime(df["time_bin"])
    df["date"] = df["time_bin"].dt.date
    df["time"] = df["time_bin"].dt.time
    df["time_short"] = df["time"].astype(str).str[:-3]
    dfs[key] = df

# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__)
app.title = "SDC Multipurpose Room Activity Tracker"

app.layout = html.Div([

    # Header logos
    html.Div([
        html.Img(src='assets/MTU_Logo.png', style={'height': '60px'}),
        html.Img(src='assets/UIL_Logo.png', style={'height': '160px'})
    ], style={
        "width": "100%", "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "marginBottom": "10px"
    }),

    html.H1("SDC Multipurpose Room Activity Tracker",
            style={"textAlign": "center", "marginBottom": "20px"}),

    # Main content row
    html.Div([

        # LEFT COLUMN
        html.Div([
            html.Img(src='assets/SDC_Multipurpose_BirdsEye.jpeg',
                     style={'width': '100%', 'marginBottom': '20px'}),

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
                min_date_allowed=date(2026, 4, 6),
                max_date_allowed=date(2026, 4, 12),
                date=date(2026, 4, 6),
                display_format="YYYY-MM-DD",
                style={"width": "90%"}
            ),

            # Store current window index
            dcc.Store(id="time-index", data=0)

        ], style={
            "width": "30%", "display": "inline-block",
            "verticalAlign": "top", "padding": "10px"
        }),

        # RIGHT COLUMN (Graph + navigation buttons)
        html.Div([

            dcc.Graph(id="activity-graph"),

            # Navigation buttons under the graph
            html.Div([
                html.Button("←", id="prev-btn", n_clicks=0,
                            style={
                                "fontSize": "22px",
                                "padding": "6px 14px",
                                "float": "left"
                            }),

                html.Button("→", id="next-btn", n_clicks=0,
                            style={
                                "fontSize": "22px",
                                "padding": "6px 14px",
                                "float": "right"
                            }),
            ], style={
                "width": "100%",
                "display": "block",
                "marginTop": "10px",
                "overflow": "auto"
            })

        ], style={
            "width": "68%", "display": "inline-block",
            "padding": "10px"
        }),

    ], style={"display": "flex", "justifyContent": "space-between"}),

])

# -----------------------------
# Callback: update graph + window navigation
# -----------------------------
@app.callback(
    Output("activity-graph", "figure"),
    Output("time-index", "data"),
    Input("location-dropdown", "value"),
    Input("date-picker", "date"),
    Input("prev-btn", "n_clicks"),
    Input("next-btn", "n_clicks"),
    State("time-index", "data")
)
def update_graph(location, selected_date, prev_clicks, next_clicks, index):

    if location is None:
        return px.bar(title="Select a location to view predictions"), index

    df = dfs[location]
    selected_date = pd.to_datetime(selected_date).date()
    filtered = df[df["date"] == selected_date]

    if filtered.empty:
        return px.bar(title=f"No prediction data for {selected_date}"), index

    # Convert to lists for slicing
    times = filtered["time_short"].tolist()
    preds = filtered["prediction"].tolist()

    window_size = 18  # number of bars visible at once

    # Determine which button was clicked
    ctx = callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "prev-btn":
            index = max(0, index - window_size)
        elif button_id == "next-btn":
            index = min(len(times) - window_size, index + window_size)

    # Slice the window
    end = index + window_size
    times_window = times[index:end]
    preds_window = preds[index:end]

    fig = px.bar(
        x=times_window,
        y=preds_window
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Predicted People Count",
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis_tickfont=dict(size=12)
    )

    fig.update_xaxes(tickangle=0)

    fig.update_traces(
        marker_color="#FFCD00",
        marker_line_color="#000000",
        marker_line_width=1.2
    )

    return fig, index


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
