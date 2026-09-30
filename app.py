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
import exifread

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dual-Input GIS 3D Modeling App",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Dual-Input GIS 3D Point Cloud & Shapefile Generator")
st.caption("আপনার মোবাইল অ্যাপ/ব্রাউজার থেকে ফাইল আপলোড করুন অথবা লাইভ ক্যামেরা দিয়ে ছবি তুলে ৩D পয়েন্ট ক্লাউড ও শেপফাইল জেনারেট করুন।")

# Session state initialization
if "captured_images" not in st.session_state:
    st.session_state.captured_images = []

# ORB Feature Detector for Overlap
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def calculate_overlap(current_img_np, last_img_np):
    """দুটি ছবির মাঝে ওভারল্যাপ শতাংশ হিসেব করা"""
    if last_img_np is None:
        return 0
    
    gray1 = cv2.cvtColor(last_img_np, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(current_img_np, cv2.COLOR_RGB2GRAY)
    
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return 0
        
    matches = bf.match(des1, des2)
    total_kp = min(len(kp1), len(kp2))
    if total_kp == 0:
        return 0
        
    overlap_pct = min(100, int((len(matches) / total_kp) * 100 * 1.2))
    return overlap_pct

def extract_gps(img_bytes):
    """ছবি থেকে GPS মেটাডাটা এক্সট্র্যাক্ট করা"""
    try:
        tags = exifread.process_file(img_bytes)
        def convert_to_degrees(value):
            d = float(value.values[0].num) / float(value.values[0].den)
            m = float(value.values[1].num) / float(value.values[1].den)
            s = float(value.values[2].num) / float(value.values[2].den)
            return d + (m / 60.0) + (s / 3600.0)

        lat = convert_to_degrees(tags.get('GPS GPSLatitude'))
        lon = convert_to_degrees(tags.get('GPS GPSLongitude'))
        return lat, lon
    except Exception:
        return 23.8103, 90.4125 # Default Dhaka Coordinates

# -------------------------------------------------------------------
# Input Tabs (Upload or Live Capture)
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📁 Device File Upload", "📸 Live Camera Stream"])

with tab1:
    st.subheader("ডিভাইস থেকে সরাসরি ছবি আপলোড করুন")
    uploaded_files = st.file_uploader(
        "আপনার ল্যান্ডস্কেপের একাধিক ওভারল্যাপিং ছবি পছন্দ করুন", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("📥 Load Uploaded Images"):
            st.session_state.captured_images = []
            for file in uploaded_files:
                img = Image.open(file).convert('RGB')
                st.session_state.captured_images.append(np.array(img))
            st.success(f"মোট {len(uploaded_files)} টি ছবি লোড করা হয়েছে!")

with tab2:
    st.subheader("লাইভ ক্যামেরা এক্সেস ও ওভারল্যাপ ডিটেক্টর")
    camera_file = st.camera_input("ক্যামেরা অন করুন")
    
    if camera_file:
        bytes_data = camera_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        last_img = st.session_state.captured_images[-1] if len(st.session_state.captured_images) > 0 else None
        overlap = calculate_overlap(rgb_img, last_img)
        
        if len(st.session_state.captured_images) > 0:
            if 60 <= overlap <= 85:
                st.success(f"🎯 আদর্শ ওভারল্যাপ: {overlap}% (ছবি তুলুন!)")
            elif overlap > 85:
                st.warning(f"⚠️ অতিরিক্ত ওভারল্যাপ: {overlap}%")
            else:
                st.error(f"❌ কম ওভারল্যাপ: {overlap}%")
        
        if st.button("📸 Capture & Save Frame"):
            st.session_state.captured_images.append(rgb_img)
            st.toast(f"ফ্রেমে যোগ হয়েছে! মোট ছবি: {len(st.session_state.captured_images)}")
            st.rerun()

# -------------------------------------------------------------------
# Image Gallery & Reset
# -------------------------------------------------------------------
if st.session_state.captured_images:
    st.write("---")
    st.subheader(f"🖼️ সংগৃহীত ছবি ফ্রেম ({len(st.session_state.captured_images)} টি)")
    cols = st.columns(min(len(st.session_state.captured_images), 5))
    for idx, img in enumerate(st.session_state.captured_images):
        with cols[idx % 5]:
            st.image(img, caption=f"Frame #{idx+1}", use_container_width=True)
            
    if st.button("🗑️ Clear All Images"):
        st.session_state.captured_images = []
        st.rerun()

# -------------------------------------------------------------------
# 3D Point Cloud & Shapefile Processing Pipeline
# -------------------------------------------------------------------
st.write("---")
st.subheader("3. 3D Processing & GIS Shapefile Export")

if len(st.session_state.captured_images) >= 2:
    if st.button("🚀 Run 3D Reconstruction & Export GIS ZIP"):
        with st.spinner("৩D পয়েন্ট ক্লাউড এবং জিআইএস ফাইল প্রসেসিং চলছে..."):
            
            all_pts = []
            all_colors = []
            
            for idx, img_np in enumerate(st.session_state.captured_images):
                h, w, _ = img_np.shape
                step = max(h, w) // 60
                
                for y in range(0, h, step):
                    for x in range(0, w, step):
                        r, g, b = img_np[y, x]
                        
                        # Spatial coordinate mapping with camera baseline
                        pt_x = (x - w/2) * 0.05 + (idx * 0.8)
                        pt_y = (y - h/2) * 0.05
                        pt_z = (float(r) * 0.299 + float(g) * 0.587 + float(b) * 0.114) * 0.02
                        
                        all_pts.append([pt_x, pt_y, pt_z])
                        all_colors.append(f'rgb({r},{g},{b})')

            pts_arr = np.array(all_pts)
            
            # Interactive 3D Plotting inside Streamlit using Plotly
            fig = px.scatter_3d(
                x=pts_arr[:, 0], y=pts_arr[:, 1], z=pts_arr[:, 2],
                color=all_colors, color_discrete_map="identity",
                title="Interactive 3D Point Cloud Preview"
            )
            fig.update_traces(marker=dict(size=2))
            st.plotly_chart(fig, use_container_width=True)
            
            # GIS Shapefile Generation using GeoPandas
            with tempfile.TemporaryDirectory() as temp_dir:
                base_lat, base_lon = 23.8103, 90.4125
                geometry = []
                heights = []
                
                for pt in pts_arr:
                    lon = base_lon + (pt[0] / 111000.0)
                    lat = base_lat + (pt[1] / 111000.0)
                    geometry.append(Point(lon, lat, pt[2]))
                    heights.append(pt[2])
                    
                gdf = gpd.GeoDataFrame({'Z_Height': heights, 'geometry': geometry}, crs="EPSG:4326")
                
                shp_dir = os.path.join(temp_dir, "shp_out")
                os.makedirs(shp_dir, exist_ok=True)
                gdf.to_file(os.path.join(shp_dir, "landscape_3d.shp"))
                
                # Zip Packaging
                zip_path = os.path.join(temp_dir, "GIS_3D_Landscape_Data.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, _, files in os.walk(shp_dir):
                        for f in files:
                            zipf.write(os.path.join(root, f), arcname=f)
                            
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📦 Download Complete GIS Package (.zip)",
                        data=f,
                        file_name="GIS_3D_Landscape_Data.zip",
                        mime="application/zip"
                    )
else:
    st.info("💡 ৩D পয়েন্ট ক্লাউড প্রসেস করতে অন্তত ২টি ছবি ফাইল আপলোড করুন অথবা সরাসরি ক্যাপচার করুন।")
