
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from flask_caching import Cache


# Load dataframe
df = pd.read_csv('ArizonaPitching25.csv')

# Get list of all pitchers
all_pitchers = sorted(df['Pitcher'].dropna().unique())

# Define swing events
swing = ['StrikeSwinging', 'InPlay', 'FoulBallNotFieldable']

# Define statistics
stats_options = [
    {'label': 'Whiff Rate %', 'value': 'whiff_rate'},
    {'label': 'Contact Rate %', 'value': 'contact_rate'},
    {'label': 'In-Play Rate %', 'value': 'inplay_rate'},
    {'label': 'Strike Rate %', 'value': 'strike_rate'},
    {'label': 'Chase Rate %', 'value': 'chase_rate'},
    {'label': 'Swing Rate %', 'value': 'swing_rate'}
]


# Helper functions for statistic calculations

# Calculate whiff rate: whiffs / swings
def calculate_whiff_rate(nearby_df):
    swings = nearby_df[nearby_df['IsSwing']]
    if len(swings) == 0:
        return 0
    whiffs = nearby_df[nearby_df['IsWhiff']]
    return (len(whiffs) / len(swings)) * 100

# Calculate contact rate: contact / swings
def calculate_contact_rate(nearby_df):
    swings = nearby_df[nearby_df['IsSwing']]
    if len(swings) == 0:
        return 0
    contact = nearby_df[nearby_df['IsContact']]
    return (len(contact) / len(swings)) * 100

# Calculate inplay rate: balls in play / swings
def calculate_inplay_rate(nearby_df):
    swings = nearby_df[nearby_df['IsSwing']]
    if len(swings) == 0:
        return 0
    inplay = nearby_df[nearby_df['IsInPlay']]
    return (len(inplay) / len(swings)) * 100

# Calculate strike rate: strikes / all pitches
def calculate_strike_rate(nearby_df):
    if len(nearby_df) == 0:
        return 0
    strikes = nearby_df[nearby_df['IsStrike']]
    return (len(strikes) / len(nearby_df)) * 100

# Calculate chase rate: swings outside zone / pitches outside zone
def calculate_chase_rate(nearby_df):
    outside_zone = nearby_df[~nearby_df['InZone']]
    if len(outside_zone) == 0:
        return 0
    chases = outside_zone[outside_zone['IsSwing']]
    return (len(chases) / len(outside_zone)) * 100

# Calculate swing rate: swings / all pitches
def calculate_swing_rate(nearby_df):
    if len(nearby_df) == 0:
        return 0
    swings = nearby_df[nearby_df['IsSwing']]
    return (len(swings) / len(nearby_df)) * 100

# Mapping of stat types to calculation functions
STAT_FUNCTIONS = {
    'chase_rate': calculate_chase_rate,
    'contact_rate': calculate_contact_rate,
    'inplay_rate': calculate_inplay_rate,
    'strike_rate': calculate_strike_rate,
    'swing_rate': calculate_swing_rate,
    'whiff_rate': calculate_whiff_rate
}

# Dash app

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})


@cache.memoize()
# Calculate the statistic grid for a specific player and statistic
def calculate_stat_grid(player_name, stat_type, resolution=200, radius=0.4, sigma=4):
    # DF for selected player
    player_df = df[df['Pitcher'] == player_name].copy()
    
    # Create boolean columns for different pitch outcomes
    player_df['IsSwing'] = player_df['PitchCall'].isin(swing)
    player_df['IsWhiff'] = player_df['PitchCall'] == 'StrikeSwinging'
    player_df['IsContact'] = player_df['PitchCall'].isin(['InPlay', 'FoulBallNotFieldable'])
    player_df['IsInPlay'] = player_df['PitchCall'] == 'InPlay'
    player_df['IsStrike'] = player_df['PitchCall'].str.contains('Strike', na=False)
    
    # Determine if pitch is in strike zone
    zone_width = 17/12
    player_df['InZone'] = (
        (player_df['PlateLocSide'].abs() <= zone_width/2) & 
        (player_df['PlateLocHeight'] >= 1.5) & 
        (player_df['PlateLocHeight'] <= 3.5)
    )
    
    # Create grid
    x_axis = np.linspace(-2, 2, resolution)
    y_axis = np.linspace(0, 5, resolution)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    

    
    # Calculate stat for each grid point
    stat_grid = np.zeros_like(x_grid)
    
	# Get pitch locations as arrays
    pitch_x = player_df['PlateLocSide'].values
    pitch_y = player_df['PlateLocHeight'].values
    
    # Loop through every point in grid
    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            # Vectorization for speedup
            dx = pitch_x - x_grid[i, j]
            dy = pitch_y - y_grid[i, j]
            distances = np.sqrt(dx**2 + dy**2)
            mask = distances <= radius
            
            nearby = player_df[mask]
            
			# Set calculation to 0 when pitches less than 3 for reliability
            if len(nearby) < 3:
                stat_grid[i, j] = 0
                continue
            
            # Call the appropriate stat function
            stat_grid[i, j] = STAT_FUNCTIONS[stat_type](nearby)
    
    # Apply smoothing
    stat_grid_smooth = gaussian_filter(stat_grid, sigma=sigma)
    
    return x_axis, y_axis, stat_grid_smooth, len(player_df)


# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Pitcher Statistics Heatmap", className="text-center text-light mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Pitcher:", className="text-light"),
            dcc.Dropdown(
                id='pitcher-dropdown',
                options=[{'label': pitcher, 'value': pitcher} for pitcher in all_pitchers],
                value='Alberini, Hunter',
                clearable=False,
                className="mb-3",
                style={'color': '#000000'}
            )
        ], width=4),
        
        dbc.Col([
            html.Label("Select Statistic:", className="text-light"),
            dcc.Dropdown(
                id='stat-dropdown',
                options=stats_options,
                value='chase_rate',
                clearable=False,
                className="mb-3",
                style={'color': '#000000'}
            )
        ], width=4),
        
        dbc.Col([
            html.Label("Display Options:", className="text-light"),
            dcc.Checklist(
                id='contour-toggle',
                options=[{'label': ' Show Contour Lines', 'value': 'show'}],
                value=['show'],
                className="text-light"
            )
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
            dcc.Graph(id='whiff-heatmap', style={'height': '80vh'}))
        ])
    ])
], fluid=True, style={'backgroundColor': '#0a0a1a'})

@app.callback(
    Output('whiff-heatmap', 'figure'),
    [Input('pitcher-dropdown', 'value'),
     Input('stat-dropdown', 'value'),
     Input('contour-toggle', 'value')]
)

# Update with new selection
def update_heatmap(pitcher_name, stat_type, contour_options):
    # Calculate grid
    x_axis, y_axis, stat_grid_smooth, num_pitches = calculate_stat_grid(pitcher_name, stat_type)
    
	
    # Get stat name for title
    stat_names = {
        'whiff_rate': 'Whiff Rate',
        'contact_rate': 'Contact Rate',
        'inplay_rate': 'In-Play Rate',
        'strike_rate': 'Strike Rate',
        'chase_rate': 'Chase Rate',
        'swing_rate': 'Swing Rate'
    }
    stat_name = stat_names.get(stat_type, 'Statistic')
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        x=x_axis,
        y=y_axis,
        z=stat_grid_smooth,
        colorscale='Plasma',
        zmin=0,
        zmax=100,
        colorbar=dict(
            title=dict(text=f"{stat_name} (%)", font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        # Allows user to hover over heatmap and see specific data
        hovertemplate='X: %{x:.2f} ft<br>Y: %{y:.2f} ft<br>' + stat_name + ': %{z:.1f}%<extra></extra>'
    ))
    
    # Add contour lines if enabled
    if 'show' in contour_options:
        fig.add_trace(go.Contour(
            x=x_axis,
            y=y_axis,
            z=stat_grid_smooth,
            contours=dict(
                start=10,
                end=100,
                size=10,
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            line=dict(width=1, color='white'),
            showscale=False,
            opacity=0.4,
            hoverinfo='skip'
        ))
    
    # Add strike zone rectangle
    zone_width = 17/12
    fig.add_shape(
        type="rect",
        x0=-zone_width/2, y0=1.5,
        x1=zone_width/2, y1=3.5,
        line=dict(color="white", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{stat_name} Map for: {pitcher_name} ({num_pitches} Pitches)',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Plate Location Side (ft)',
            range=[-2, 2],
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='Plate Location Height (ft)',
            range=[0, 5],
            gridcolor='rgba(255,255,255,0.1)'
        ),
        plot_bgcolor='#0a0a1a',
        paper_bgcolor='#0a0a1a',
        font=dict(color='white'),
        height=700
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    
