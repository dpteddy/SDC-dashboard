from dash import Dash, html, dcc
from datetime import date

# 6 different files for data (1 for each dropdown option) 
# recorded in 30 minumutes intervals with dates and 
# has the location and amount of people in there

app = Dash(__name__)

# Dashboard Title
app.title = "SDC Multipurpose Room Activity Tracker"

# Dashboard Layout
app.layout = html.Div([

    # MTU logo
    html.Img(src='/assets/MTU_Logo.png', style={'height':'10%', 'width':'10%'}),

    # Breaks help put a graphical component into the row of objects
    html.Br(),

    # Multipurpose room birdseye view
    html.Img(src='/assets/SDC_Multipurpose_BirdsEye.jpeg', style={'height': '30%', 'width': '30%'}),

    html.Br(),

    # Locations Dropdown
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
    ),

     # Date Selection
    html.H3("Date"),

    dcc.DatePickerSingle(
        id="date-picker",
        min_date_allowed=date(2026, 1, 1),
        max_date_allowed=date.today(),
        date=date.today(),
        display_format="YYYY-MM-DD",
        style={"width": "300px"},
    )

])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)