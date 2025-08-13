import plotly.graph_objects as go
import pandas as pd
import numpy as np
def create_l_fig(title:str = 'Robot Location', width:int = 1200, height:int = 800):
    location_fig = go.Figure()
    # location_fig.add_trace(go.Scattergeo(lon = l_df['longitude'], lat = l_df['latitude'],mode="markers",
    #     marker=dict(size=5, color="blue"), name = name))
    # location_fig.add_trace(go.Scattergeo(lon = survey.l_sub_df['longitude'], lat = survey.l_sub_df['latitude'],mode="markers",
    #     marker=dict(size=5, color="red"), name = 'Subset Data'))

    # Layout settings
    location_fig.update_layout(
        title=title,
        geo=dict(
            scope="world",  # Use 'europe' for a more focused view
            showland=True,
            
        ),width=width,  # Increase width
        height=height   # Increase height
    )

    return location_fig

def add_point_to_l_fig(location_fig, l_df:pd.DataFrame, size:int, name:str, color:str):
    location_fig.add_trace(go.Scattergeo(lon = l_df['longitude'], lat = l_df['latitude'],mode="markers",
        marker=dict(size=size, color=color), name = name))

    return location_fig
