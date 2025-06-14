
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
from streamlit_folium import st_folium
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🔌 Least-Cost Electrification Planning Tool")

params = {
    'Grid': {'CAPEX_km': 25000, 'OPEX_percent': 0.05, 'lifetime': 30, 'discount_rate': 0.08, 'connection_rate': 0.3, 'annual_demand_kWh': 700},
    'Mini-grid': {'CAPEX_kW': 8000, 'OPEX_percent': 0.05, 'lifetime': 25, 'discount_rate': 0.08, 'connection_rate': 0.9, 'annual_demand_kWh': 700},
    'SHS': {'CAPEX_unit': 400, 'OPEX_percent': 0.02, 'lifetime': 10, 'discount_rate': 0.08, 'connection_rate': 1, 'annual_demand_kWh': 224}
}

def calculate_lcoe(capex, opex_percent, lifetime, discount_rate, annual_energy):
    total_capex = capex
    total_opex = total_capex * opex_percent
    npv_cost = total_capex + sum(total_opex / (1 + discount_rate)**t for t in range(1, lifetime+1))
    npv_energy = sum(annual_energy / (1 + discount_rate)**t for t in range(1, lifetime+1))
    return npv_cost / npv_energy

uploaded_file = st.file_uploader("📥 Upload a CSV with Latitude, Longitude, and Population", type="csv")

# User-adjustable parameters
n_clusters = st.slider("Number of Clusters", min_value=5, max_value=30, value=10)
grid_lat = st.number_input("Grid Latitude", value=-15.75, format="%.5f")
grid_lon = st.number_input("Grid Longitude", value=49.25, format="%.5f")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))

    coords = gdf[['Longitude', 'Latitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    gdf['Cluster'] = kmeans.labels_

    cluster_centers = gdf.groupby('Cluster')[['Longitude', 'Latitude']].mean().reset_index()
    cluster_centers['distance_to_grid_km'] = np.sqrt(
        (cluster_centers['Longitude'] - grid_lon)**2 +
        (cluster_centers['Latitude'] - grid_lat)**2) * 111

    results = []
    for _, cluster in cluster_centers.iterrows():
        cluster_id = cluster['Cluster']
        n_households = gdf[gdf['Cluster'] == cluster_id].shape[0]
        annual_energy = n_households * params['Grid']['annual_demand_kWh']

        grid_capex = params['Grid']['CAPEX_km'] * cluster['distance_to_grid_km']
        lcoe_grid = calculate_lcoe(grid_capex, params['Grid']['OPEX_percent'],
                                   params['Grid']['lifetime'], params['Grid']['discount_rate'],
                                   annual_energy * params['Grid']['connection_rate'])

        mini_capex = params['Mini-grid']['CAPEX_kW'] * (annual_energy / (24 * 365))
        lcoe_mini = calculate_lcoe(mini_capex, params['Mini-grid']['OPEX_percent'],
                                   params['Mini-grid']['lifetime'], params['Mini-grid']['discount_rate'],
                                   annual_energy * params['Mini-grid']['connection_rate'])

        shs_capex = params['SHS']['CAPEX_unit'] * n_households
        lcoe_shs = calculate_lcoe(shs_capex, params['SHS']['OPEX_percent'],
                                  params['SHS']['lifetime'], params['SHS']['discount_rate'],
                                  annual_energy * params['SHS']['connection_rate'])

        lcoes = {'Grid': lcoe_grid, 'Mini-grid': lcoe_mini, 'SHS': lcoe_shs}
        best = min(lcoes, key=lcoes.get)

        results.append({
            'Cluster': cluster_id,
            'Grid_LCOE': lcoe_grid,
            'MiniGrid_LCOE': lcoe_mini,
            'SHS_LCOE': lcoe_shs,
            'Best_Option': best,
            'Households': n_households,
            'Distance_km': cluster['distance_to_grid_km']
        })

    res_df = pd.DataFrame(results)
    gdf = gdf.merge(res_df[['Cluster', 'Best_Option']], on='Cluster')

    # Show Data
    st.subheader("📊 Cluster-wise Electrification Option")
    st.dataframe(res_df)

    # Folium Map
    st.subheader("🗺️ Interactive Electrification Map")
    center_lat, center_lon = gdf['Latitude'].mean(), gdf['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    colors = {'Grid': 'blue', 'Mini-grid': 'green', 'SHS': 'gray'}

    marker_cluster = MarkerCluster().add_to(m)
    for _, row in gdf.iterrows():
        color = colors[row['Best_Option']]
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(marker_cluster)

    folium.Marker([grid_lat, grid_lon], icon=folium.Icon(color='black'), popup="Existing Grid").add_to(m)

    st_data = st_folium(m, width=900, height=500)

    # Download CSV
    st.subheader("📥 Download Results")
    csv_data = res_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download LCOE Summary CSV", data=csv_data, file_name="lcoe_summary.csv", mime='text/csv')
