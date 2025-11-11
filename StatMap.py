
# Authors: Maya Vanderpool and Antoine

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Define File Path and Column Names
# NOTE: Please ensure this file path is valid when running locally.
FILE_PATH = 'ArizonaPitching25.csv'
SIDE_FIELD_COLUMN = 'PlateLocSide'
HEIGHT_FIELD_COLUMN = 'PlateLocHeight'
DATE_COLUMN = 'Date' 
FILTER_COLUMN = 'Pitcher'

# Load Data
try:
    # Load with 'low_memory=False' if you have mixed types in columns
    df = pd.read_csv(FILE_PATH, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Please check the path.")
    exit()


# --- Date Preprocessing ---
DATE_COLUMN = 'Date'
if DATE_COLUMN in df.columns:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df.dropna(subset=[DATE_COLUMN], inplace=True)
    
    unique_dates = sorted(df[DATE_COLUMN].dt.date.unique())
    
    date_map = {date: i for i, date in enumerate(unique_dates)}
    inverse_date_map = {i: date for i, date in enumerate(unique_dates)}
    
    MIN_DATE_INDEX = 0
    MAX_DATE_INDEX = len(unique_dates) - 1
else:
    print(f"Error: Date column '{DATE_COLUMN}' not found in CSV. Date slider disabled.")
    MIN_DATE_INDEX = 0
    MAX_DATE_INDEX = 0
    date_map = {}
    inverse_date_map = {}


# --- Helper function to format date marks for the slider ---
def format_date_marks(date_indices, step=5):
    marks = {}
    if not inverse_date_map:
        return marks
        
    # Only show a subset of marks to keep the slider clean
    for i, date_index in enumerate(date_indices):
        # Show marks at every 'step' interval and the very last date
        if (i % step == 0 and i != 0) or date_index == date_indices[-1] or date_index == date_indices[0]:
            # Ensure the index exists in the map before accessing
            if date_index in inverse_date_map:
                date_obj = inverse_date_map[date_index]
                marks[date_index] = {'label': date_obj.strftime('%b %d'), 'style': {'color': 'white', 'fontSize': '10px'}}
            
    return marks


# Get list of all pitchers
all_pitchers = sorted(df['Pitcher'].dropna().unique())

# Define swing events
swing = ['StrikeSwinging', 'InPlay', 'FoulBallNotFieldable']

# Define radius for nearby pitch calculation 
radius = 0.3  # feet

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
app = Dash(__name__)


# Calculate the statistic grid for a specific player and statistic
def calculate_stat_grid(player_df, stat_type):
    
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
    x_axis = np.linspace(-2, 2)
    y_axis = np.linspace(0, 5)
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
    
    # Apply Gaussian smoothing 
    stat_grid_smooth = gaussian_filter(stat_grid, sigma=2)
   
    return x_axis, y_axis, stat_grid_smooth


# Layout
app.layout = html.Div([
    html.H1("Pitcher Statistics Heatmap", style={'textAlign': 'center'}),
    
    html.Div([
        # Pitcher Dropdown
        html.Div([
            html.Label("Select Pitcher:", style={'color': 'white'}),
            dcc.Dropdown(
                id='pitcher-dropdown',
                options=[{'label': 'All Pitchers', 'value': 'ALL'}] + \
                   [{'label': pitcher, 'value': pitcher} for pitcher in all_pitchers],
                value='ALL',
                clearable=False,
                style={'width': '300px', 'color': 'black'}
            )
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        
        # Statistic Dropdown
        html.Div([
            html.Label("Select Statistic:", style={'color': 'white'}),
            dcc.Dropdown(
                id='stat-dropdown',
                options=stats_options,
                value='chase_rate',
                clearable=False,
                style={'width': '200px', 'color': 'black'}
            )
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        
        # Contour Lines Dropdown
        html.Div([
            html.Label("Show Contour Lines:", style={'color': 'white'}),
            dcc.Dropdown(
                id='contour-line-dropdown',
                options=[
                    {'label': 'Show Lines', 'value': 'show'},
                    {'label': 'Remove Lines', 'value': 'hide'}
                ],
                value='show', 
                clearable=False,
                style={'width': '150px', 'color': 'black'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        
        # Highlight Peak Checkbox
        html.Div([
            html.Label("Highlight Peak Density:", style={'color': 'white'}),
            dcc.Checklist(
                id='highlight-peak-checkbox',
                options=[{'label': ' Yes', 'value': 'HIGHLIGHT'}],
                value=[], 
                style={'color': 'white'}
            ),
        ], style={'display': 'inline-block'}),
        
    ], style={'textAlign': 'center', 'padding': '10px'}),
    
    # Date Range Slider
    html.Div([
        html.Label("Filter by Date Range:", 
                   style={'color': 'white', 'textAlign': 'left', 'marginBottom': '10px'}),
        dcc.RangeSlider(
            id='date-range-slider',
            min=MIN_DATE_INDEX,
            max=MAX_DATE_INDEX,
            step=1,
            value=[MIN_DATE_INDEX, MAX_DATE_INDEX], 
            marks=format_date_marks(range(MIN_DATE_INDEX, MAX_DATE_INDEX + 1)),
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'width': '90%', 'margin': '20px auto'}),
    
    # Output for Mathematical Spread Factor (Standard Deviation)
    html.Div(
        id='math-spread-factor-output',
        style={
            'textAlign': 'center', 
            'fontSize': '1.2em', 
            'marginTop': '10px', 
            'padding': '10px',
            'backgroundColor': '#333',
            'borderRadius': '5px'
        }
    ),
    
    dcc.Graph(id='stat-heatmap', style={'height': '80vh'})
    
], style={'backgroundColor': 'black', 'color': 'white', 'padding': '20px'})

@app.callback(
    Output('math-spread-factor-output', 'children'), 
    Output('stat-heatmap', 'figure'),
    [Input('pitcher-dropdown', 'value'),
     Input('stat-dropdown', 'value'),
     Input('contour-line-dropdown', 'value'),
     Input('highlight-peak-checkbox', 'value'),
     Input('date-range-slider', 'value')]
)

# Update with new selection
def update_heatmap(pitcher_name, stat_type, contour_option, highlight_peak, date_range):
    # 1. Apply Pitcher Filter
    if pitcher_name == 'ALL':
        player_df = df.copy()
    else:
        player_df = df[df['Pitcher'] == pitcher_name].copy()

    
    # 2. Apply Date Filter
    if inverse_date_map and date_range:
        start_date_index, end_date_index = date_range
        start_date = inverse_date_map[start_date_index]
        end_date = inverse_date_map[end_date_index]
        
        # Filter the DataFrame based on the selected date range (inclusive)
        player_df = player_df[
            (player_df[DATE_COLUMN].dt.date >= start_date) & 
            (player_df[DATE_COLUMN].dt.date <= end_date)
        ]
    
    # 3. Calculate Mathematical Spread Factor (Standard Deviation)
    if len(player_df) > 1:
        spread_x = player_df['PlateLocSide'].std()
        spread_y = player_df['PlateLocHeight'].std()
        
        spread_output_div = html.Span([
            "Pitch Dispersion (Standard Deviation σ): ",
            html.B(f"Horizontal (σ): {spread_x:.3f} ft"),
            " | ",
            html.B(f"Vertical (σ): {spread_y:.3f} ft")
        ])
    else:
        spread_output_div = f"Not enough pitches to calculate dispersion metrics for the selected range. (Pitches: {len(player_df)})"
    
    # 4. Calculate grid
    x_axis, y_axis, stat_grid_smooth = calculate_stat_grid(player_df, stat_type)
    
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
    
    # Define axis ranges 
    x_min, x_max = -2.0, 2.0
    y_min, y_max = 0.0, 5.0
    sz_top = 3.5
    sz_bot = 1.5
    sz_width = 17 / 12 / 2  
    
    # Create figure
    fig = go.Figure()
    # Cap chase rate 60%
    zmax_value = 60 if stat_type == 'chase_rate' else 100
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        x=x_axis,
        y=y_axis,
        z=stat_grid_smooth,
        colorscale='Plasma',
        zmin=0,
        zmax=zmax_value,
        colorbar=dict(
            title=dict(text=f"{stat_name} (%)", font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        # Allows user to hover over heatmap and see specific data
        hovertemplate='X: %{x:.2f} ft<br>Y: %{y:.2f} ft<br>' + stat_name + ': %{z:.1f}%<extra></extra>'
    ))
    
    # Add contour lines if enabled
    if contour_option == 'show':
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
    
    # Highlight Peak Density Area
    if 'HIGHLIGHT' in highlight_peak and len(player_df) > 0:
        hist, xedges, yedges = np.histogram2d(
            player_df['PlateLocSide'], 
            player_df['PlateLocHeight'], 
            bins=50, 
            range=[[x_min, x_max], [y_min, y_max]]
        )
        max_idx = np.unravel_index(hist.argmax(), hist.shape)
        peak_x = (xedges[max_idx[0]] + xedges[max_idx[0]+1]) / 2
        peak_y = (yedges[max_idx[1]] + yedges[max_idx[1]+1]) / 2

        fig.add_trace(go.Scatter(
            x=[peak_x], y=[peak_y],
            mode='markers',
            marker=dict(
                color='gold', 
                size=15, 
                symbol='star',
                line=dict(width=2, color='darkred')
            ),
            name='Peak Density',
            showlegend=True
        ))
    
    # Add strike zone rectangle 
    fig.add_shape(
        type='rect', 
        x0=-sz_width, y0=sz_bot, 
        x1=sz_width, y1=sz_top,
        line=dict(color="white", width=2, dash="dash"),
    )
    
    # Add home plate line 
    fig.add_trace(go.Scatter(
        x=[-0.5, 0.5], y=[0.1, 0.1], mode='lines', 
        line=dict(color='white', width=3), showlegend=False,
        hoverinfo='skip'
    ))
    
    # Update layout 
    fig.update_layout(
        plot_bgcolor='black', 
        paper_bgcolor='black', 
        font_color='white',
        title=f'{stat_name} Map for: {pitcher_name} ({len(player_df)} Pitches)',
        xaxis_title="Plate Location Side (ft)", 
        yaxis_title="Plate Location Height (ft)",
        xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False),  
        yaxis=dict(range=[y_min, y_max], showgrid=False, zeroline=False), 
        margin=dict(l=40, r=40, t=60, b=40), 
        height=800,  
        showlegend=True
    )
    
    return spread_output_div, fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)