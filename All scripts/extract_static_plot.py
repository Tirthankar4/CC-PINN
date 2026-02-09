"""
Extract a static 3D plot from an interactive HTML file.
This script reads the Plotly HTML file and extracts data for a specific time and field,
then creates a static image using Plotly's rendering engine.
"""

import os
import re
import json
import numpy as np


def extract_plotly_data(html_path):
    """
    Extract data from Plotly HTML file.
    
    Returns:
        dict with keys: 'x', 'y', 'z', 'frames', 'time_values', 'fields'
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the Plotly.newPlot call - it contains the initial data
    # The pattern: Plotly.newPlot("div-id", [data_array], {layout})
    newplot_match = re.search(
        r'Plotly\.newPlot\(\s*["\'][\w-]+["\']\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*\)',
        content,
        re.DOTALL
    )
    
    if not newplot_match:
        raise ValueError("Could not find Plotly.newPlot in HTML file")
    
    # Extract the data array JSON
    data_json = newplot_match.group(1)
    
    # Find the Plotly.addFrames call - it contains the time series data
    addframes_match = re.search(
        r'Plotly\.addFrames\([^,]+,\s*(\[.*?\])\s*\)',
        content,
        re.DOTALL
    )
    
    # Parse the data
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse data JSON: {e}")
    
    # Extract coordinate arrays (from first trace)
    x = np.array(data[0]['x'])
    y = np.array(data[0]['y'])
    z = np.array(data[0]['z'])
    
    # Extract time values from frame names
    time_values = []
    frames_by_time = {}
    
    if addframes_match:
        frames_json = addframes_match.group(1)
        try:
            frames_data = json.loads(frames_json)
            for frame in frames_data:
                t = float(frame['name'])
                time_values.append(t)
                frames_by_time[t] = frame['data']
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse frames data: {e}")
            print("Using only initial data")
    
    # If no frames found, use initial data
    if not time_values:
        print("No frames found, using initial data at t=0.0")
        time_values = [0.0]
        frames_by_time[0.0] = data
    
    # Field order: density, vx, vy, vz, phi
    fields = ['density', 'vx', 'vy', 'vz', 'phi']
    
    return {
        'x': x,
        'y': y,
        'z': z,
        'frames': frames_by_time,
        'time_values': sorted(time_values),
        'fields': fields,
        'initial_data': data,  # Contains initial frame data
    }


def get_field_at_time(plotly_data, field_name, time_value):
    """
    Get field values at a specific time.
    
    Args:
        plotly_data: Output from extract_plotly_data
        field_name: One of 'density', 'vx', 'vy', 'vz', 'phi'
        time_value: Time value to extract
        
    Returns:
        numpy array of field values
    """
    field_idx = plotly_data['fields'].index(field_name)
    
    # Find closest time
    times = plotly_data['time_values']
    closest_time = min(times, key=lambda t: abs(t - time_value))
    
    if abs(closest_time - time_value) > 1e-3:
        print(f"Warning: Requested time {time_value}, using closest available time {closest_time}")
    
    # Get the frame data
    frame_data = plotly_data['frames'][closest_time]
    values = np.array(frame_data[field_idx]['value'])
    
    return values, closest_time


def plot_3d_volume(x, y, z, values, field_label='rho', title=None, save_path=None, 
                   colorscale='Viridis', cmin=None, cmax=None, opacity=0.1, 
                   surface_count=15, width=1200, height=800, camera=None):
    """
    Create a 3D volume plot using Plotly and export as static image.
    
    Args:
        x, y, z: 1D arrays of coordinates (flattened grid)
        values: 1D array of field values
        field_label: Label for colorbar
        title: Plot title
        save_path: Path to save figure
        colorscale: Plotly colorscale name
        cmin, cmax: Color scale limits (auto if None)
        opacity: Volume opacity
        surface_count: Number of isosurfaces
        width, height: Image dimensions in pixels
        camera: Camera settings dict (optional)
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")
    
    # Set color limits if not provided
    if cmin is None:
        cmin = float(np.min(values))
    if cmax is None:
        cmax = float(np.max(values))
    
    # Create volume trace
    trace = go.Volume(
        x=x,
        y=y,
        z=z,
        value=values,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        colorbar=dict(
            title=field_label,
            orientation='h',  # Horizontal colorbar
            x=0.5,            # Center horizontally
            y=-0.05,          # Position closer to plot
            xanchor='center',
            yanchor='top',
            len=0.6,          # Length of colorbar (60% of width)
            thickness=20,
            lenmode='fraction',
        ),
        showscale=True,
    )
    
    # Create figure
    fig = go.Figure(data=[trace])
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center') if title else None,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode='data',
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=50, t=40 if title else 0),  # Reduced bottom margin
    )
    
    # Set camera angle if provided
    if camera:
        fig.update_layout(scene_camera=camera)
    
    if save_path:
        # Try to export as static image
        try:
            # First try kaleido (newer, recommended)
            pio.write_image(fig, save_path, format='png', width=width, height=height, scale=2)
            print(f"Plot saved to: {save_path}")
        except Exception as e1:
            print(f"Could not use kaleido: {e1}")
            try:
                # Fallback to orca
                pio.write_image(fig, save_path, format='png', width=width, height=height, scale=2)
                print(f"Plot saved to: {save_path}")
            except Exception as e2:
                print(f"Could not use orca: {e2}")
                # Fallback: save as HTML and tell user
                html_path = save_path.replace('.png', '_interactive.html')
                pio.write_html(fig, html_path)
                print(f"\nCould not export static image. Interactive HTML saved to: {html_path}")
                print("\nTo enable static image export, install kaleido:")
                print("  pip install -U kaleido")
                print("\nThen you can open the HTML in a browser and take a screenshot,")
                print("or re-run this script to export directly to PNG.")
                return fig, html_path
    
    return fig, save_path


def main(html_path, time_value=3.00, field='density', output_path=None, 
         width=1200, height=800, camera=None):
    """
    Main function to extract and plot data from HTML.
    
    Args:
        html_path: Path to interactive HTML file
        time_value: Time value to extract (default: 3.00)
        field: Field name to plot (default: 'density')
        output_path: Path to save output image (optional)
        width, height: Image dimensions in pixels
        camera: Camera settings dict (optional)
    """
    print(f"Reading HTML file: {html_path}")
    plotly_data = extract_plotly_data(html_path)
    
    print(f"Available times: {plotly_data['time_values']}")
    print(f"Available fields: {plotly_data['fields']}")
    
    print(f"\nExtracting {field} at t={time_value}...")
    values, actual_time = get_field_at_time(plotly_data, field, time_value)
    
    # Determine colorscale based on field
    colorscales = {
        'density': 'Viridis',
        'vx': 'RdBu',
        'vy': 'RdBu',
        'vz': 'RdBu',
        'phi': 'Plasma',
    }
    colorscale = colorscales.get(field, 'Viridis')
    
    # Get color limits from the initial data
    field_idx = plotly_data['fields'].index(field)
    initial_trace = plotly_data['initial_data'][field_idx]
    cmin = initial_trace.get('cmin', None)
    cmax = initial_trace.get('cmax', None)
    
    # Create title
    title = f"{field} at t = {actual_time:.2f}"
    
    # Determine output path
    if output_path is None:
        base_dir = os.path.dirname(html_path)
        output_path = os.path.join(base_dir, f"{field}_t{actual_time:.2f}.png")
    
    print(f"Creating 3D plot...")
    fig, result_path = plot_3d_volume(
        plotly_data['x'],
        plotly_data['y'],
        plotly_data['z'],
        values,
        field_label=field,
        title=title,
        save_path=output_path,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        width=width,
        height=height,
        camera=camera,
    )
    
    return result_path


if __name__ == "__main__":
    # Configuration
    html_file = r"C:\Users\tirth\Downloads\3D sinusoidal\sinusoidal.html"
    time_to_extract = 3.00
    field_to_plot = "density"  # or 'vx', 'vy', 'vz', 'phi'
    
    # Image settings
    image_width = 1400
    image_height = 900
    
    # Optional: Set camera angle (uncomment and adjust if needed)
    # camera = dict(
    #     eye=dict(x=1.5, y=1.5, z=1.5),  # Camera position
    #     center=dict(x=0, y=0, z=0),      # Look-at point
    #     up=dict(x=0, y=0, z=1)           # Up direction
    # )
    camera = None
    
    # Run extraction
    output = main(html_file, time_value=time_to_extract, field=field_to_plot, 
                  width=image_width, height=image_height, camera=camera)
    print(f"\nDone! Output saved to: {output}")
