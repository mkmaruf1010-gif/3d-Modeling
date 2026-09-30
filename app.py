import os
import cv2
import zipfile
import tempfile
import numpy as np
import streamlit as st
import geopandas as gpd
import plotly.express as px
from shapely.geometry import Point
from PIL import Image
from streamlit_js_eval import get_geolocation

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Georeferenced GIS 3D App",
    page_icon="📍",
    layout="wide"
)

st.title("📍 Georeferenced GIS 3D Point Cloud Generator")
st.caption("লাইভ জিপিএস লোকেশন অথবা ম্যানুয়াল স্থানাঙ্ক ব্যবহার করে সঠিক ভৌগোলিক পজিশনে ৩D মডেল জেনারেট করুন।")

if "captured_images" not in st.session_state:
    st.session_state.captured_images = []

# -------------------------------------------------------------------
# 1. Location & Positioning Settings
# -------------------------------------------------------------------
st.sidebar.header("🗺️ Geographic Reference / Positioning")
loc_option = st.sidebar.radio(
    "লোকেশন নেওয়ার উপায় সিলেক্ট করুন:",
    ["🌐 Auto Browser GPS Location", "✏️ Manual Coordinate Input"]
)

user_lat = 23.8103  # Default Dhaka Latitude
user_lon = 90.4125  # Default Dhaka Longitude
user_alt = 10.0     # Default Altitude in meters
cam_heading = 0.0   # Default North (0 Degrees)

if loc_option == "🌐 Auto Browser GPS Location":
    loc = get_geolocation()
    if loc and 'coords' in loc:
        user_lat = loc['coords']['latitude']
        user_lon = loc['coords']['longitude']
        user_alt = loc['coords'].get('altitude', 10.0) if loc['coords'].get('altitude') is not None else 10.0
        st.sidebar.success(f"📍 জিপিএস সনাক্ত হয়েছে:\nLat: {user_lat:.6f}, Lon: {user_lon:.6f}")
    else:
        st.sidebar.warning("⚠️ ব্রাউজারের লোকেশন অনুমতি দিন অথবা ম্যানুয়াল ইনপুট ব্যবহার করুন।")

else: # Manual Input
    user_lat = st.sidebar.number_input("Latitude (অক্ষাংশ)", value=23.810300, format="%.6f")
    user_lon = st.sidebar.number_input("Longitude (দ্রাঘিমাংশ)", value=90.412500, format="%.6f")
    user_alt = st.sidebar.number_input("Base Altitude (উচ্চতা মিটার)", value=10.0)

cam_heading = st.sidebar.slider("ک্যামেরার দিক / Heading Direction (Degrees)", 0, 360, 0, help="0° = North, 90° = East, 180° = South, 270° = West")

# -------------------------------------------------------------------
# 2. Input Tabs (Upload & Live Camera Capture)
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📁 Device File Upload", "📸 Live Camera Stream"])

with tab1:
    uploaded_files = st.file_uploader("ছবি ফাইল আপলোড করুন", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files and st.button("📥 Load Uploaded Images"):
        st.session_state.captured_images = [np.array(Image.open(f).convert('RGB')) for f in uploaded_files]
        st.success(f"{len(uploaded_files)} টি ছবি লোড হয়েছে!")

with tab2:
    camera_file = st.camera_input("ক্যামেরা দিয়ে সরাসরি ছবি তুলুন")
    if camera_file:
        bytes_data = camera_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        if st.button("📸 Capture Frame"):
            st.session_state.captured_images.append(rgb_img)
            st.toast(f"ছবি সেভ হয়েছে! মোট: {len(st.session_state.captured_images)}")
            st.rerun()

# Gallery
if st.session_state.captured_images:
    st.write("---")
    st.subheader(f"🖼️ ক্যাপচারকৃত ছবি ({len(st.session_state.captured_images)} টি)")
    cols = st.columns(min(len(st.session_state.captured_images), 5))
    for idx, img in enumerate(st.session_state.captured_images):
        with cols[idx % 5]:
            st.image(img, caption=f"Frame #{idx+1}", use_container_width=True)
            
    if st.button("🗑️ Clear Images"):
        st.session_state.captured_images = []
        st.rerun()

# -------------------------------------------------------------------
# 3. Georeferenced 3D Processing & GIS Export
# -------------------------------------------------------------------
st.write("---")
st.subheader("3. Geo-Referenced 3D Reconstruction")

if len(st.session_state.captured_images) >= 2:
    if st.button("🚀 Georeference & Export GIS Shapefile"):
        with st.spinner("ভৌগোলিক স্থানাঙ্ক অনুযায়ী ৩D মডেল প্রসেস হচ্ছে..."):
            
            all_pts_list = []
            all_cols_list = []
            
            rad_heading = np.radians(cam_heading)
            cos_h, sin_h = np.cos(rad_heading), np.sin(rad_heading)
            
            for idx, img_np in enumerate(st.session_state.captured_images):
                h, w, _ = img_np.shape
                step = max(h, w) // 40
                
                sub_img = img_np[::step, ::step]
                sh_h, sh_w, _ = sub_img.shape
                grid_y, grid_x = np.mgrid[0:sh_h, 0:sh_w]
                
                # Camera local 3D offset
                local_x = (grid_x * step - w/2) * 0.02 + (idx * 0.5)
                local_y = (grid_y * step - h/2) * 0.02
                
                r, g, b = sub_img[:, :, 0].astype(float), sub_img[:, :, 1].astype(float), sub_img[:, :, 2].astype(float)
                local_z = (r * 0.299 + g * 0.587 + b * 0.114) * 0.01
                
                # Apply Heading Rotation Matrix
                rot_x = local_x * cos_h - local_y * sin_h
                rot_y = local_x * sin_h + local_y * cos_h
                
                pts = np.column_stack((rot_x.ravel(), rot_y.ravel(), local_z.ravel()))
                colors = sub_img.reshape(-1, 3)
                
                all_pts_list.append(pts)
                all_cols_list.append(colors)

            pts_arr = np.vstack(all_pts_list)
            cols_arr = np.vstack(all_cols_list)
            
            # Convert Relative Meters to Geographic Coordinates (Lat/Lon)
            # 1 degree latitude ~ 111,000 meters
            # 1 degree longitude ~ 111,000 * cos(latitude) meters
            lat_offsets = pts_arr[:, 1] / 111000.0
            lon_offsets = pts_arr[:, 0] / (111000.0 * np.cos(np.radians(user_lat)))
            
            geo_lats = user_lat + lat_offsets
            geo_lons = user_lon + lon_offsets
            geo_alts = user_alt + pts_arr[:, 2]
            
          # 3D Interactive Plot
            fig = px.scatter_3d(
                x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2],
                color=hex_colors, color_discrete_map="identity",
                title="Lightweight 3D Point Cloud Preview"
            )
            fig.update_traces(marker=dict(size=2))
            st.plotly_chart(fig, use_container_width=True)
            # GeoPandas Shapefile Export
            with tempfile.TemporaryDirectory() as temp_dir:
                geometry = [Point(xyz) for xyz in zip(geo_lons, geo_lats, geo_alts)]
                
                gdf = gpd.GeoDataFrame({
                    'Latitude': geo_lats,
                    'Longitude': geo_lons,
                    'Z_Elevation': geo_alts,
                    'geometry': geometry
                }, crs="EPSG:4326")
                
                shp_dir = os.path.join(temp_dir, "georef_shp")
                os.makedirs(shp_dir, exist_ok=True)
                gdf.to_file(os.path.join(shp_dir, "georeferenced_landscape.shp"))
                
                zip_path = os.path.join(temp_dir, "Georeferenced_GIS_Data.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, _, files in os.walk(shp_dir):
                        for f in files:
                            zipf.write(os.path.join(root, f), arcname=f)
                            
                st.success("✅ জিপিএস পজিশনিং ও জিওরেফারেন্সিং সফল হয়েছে!")
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📦 Download Georeferenced GIS Package (.zip)",
                        data=f,
                        file_name="Georeferenced_GIS_Data.zip",
                        mime="application/zip"
                    )
else:
    st.info("💡 অন্তত ২টি ছবি দিয়ে প্রসেসিং স্টার্ট করুন।")
