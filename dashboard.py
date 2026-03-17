from dash import Dash, html, dcc
from datetime import date

app = Dash(__name__)

app.layout = html.Div([

    html.H3("Date"),

    
    dcc.DatePickerSingle(
        id="date-picker",
        min_date_allowed=date(2026, 1, 1),
        max_date_allowed=date.today(),
        date=date.today(),
        display_format="YYYY-MM-DD",
        style={"width": "300px"}
    ),

    html.Br(),

    html.H3("Locations"),

    dcc.Dropdown(
        id="location-dropdown",
        options=[
            {"label": "Multipurpose Room", "value": "multipurpose"},
            {"label": "Court 1", "value": "court1"},
            {"label": "Court 2", "value": "court2"},
            {"label": "Court 3", "value": "court3"},
            {"label": "Court 4", "value": "court4"},
            {"label": "Track", "value": "track"},
        ],
        placeholder="Select a location",
        style={"width": "300px"}
    )

])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)