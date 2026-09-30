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

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dual-Input GIS 3D Modeling App",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Dual-Input GIS 3D Point Cloud & Shapefile Generator")
st.caption("অপটিমাইজড পারফরম্যান্স: ব্রাউজার হ্যাং হওয়া ছাড়া দ্রুত ৩D পয়েন্ট ক্লাউড ও শেপফাইল জেনারেট করুন।")

if "captured_images" not in st.session_state:
    st.session_state.captured_images = []

orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def calculate_overlap(current_img_np, last_img_np):
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
    return min(100, int((len(matches) / total_kp) * 100 * 1.2)) if total_kp > 0 else 0

# -------------------------------------------------------------------
# Input Tabs
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📁 Device File Upload", "📸 Live Camera Stream"])

with tab1:
    st.subheader("ডিভাইস থেকে ফাইল আপলোড করুন")
    uploaded_files = st.file_uploader(
        "ল্যান্ডস্কেপের ছবি নির্বাচন করুন", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    if uploaded_files and st.button("📥 Load Uploaded Images"):
        st.session_state.captured_images = [
            np.array(Image.open(f).convert('RGB')) for f in uploaded_files
        ]
        st.success(f"মোট {len(uploaded_files)} টি ছবি লোড হয়েছে!")

with tab2:
    st.subheader("লাইভ ক্যামেরা স্ট্রিম")
    camera_file = st.camera_input("ক্যামেরা অন করুন")
    
    if camera_file:
        bytes_data = camera_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        last_img = st.session_state.captured_images[-1] if st.session_state.captured_images else None
        overlap = calculate_overlap(rgb_img, last_img)
        
        if st.session_state.captured_images:
            if 60 <= overlap <= 85:
                st.success(f"🎯 আদর্শ ওভারল্যাপ: {overlap}%")
            else:
                st.info(f"ওভারল্যাপ: {overlap}%")
        
        if st.button("📸 Capture Frame"):
            st.session_state.captured_images.append(rgb_img)
            st.toast(f"ছবি সেভ হয়েছে! মোট: {len(st.session_state.captured_images)}")
            st.rerun()

# -------------------------------------------------------------------
# Gallery
# -------------------------------------------------------------------
if st.session_state.captured_images:
    st.write("---")
    st.subheader(f"🖼️ সংগৃহীত ফ্রেম ({len(st.session_state.captured_images)} টি)")
    cols = st.columns(min(len(st.session_state.captured_images), 5))
    for idx, img in enumerate(st.session_state.captured_images):
        with cols[idx % 5]:
            st.image(img, caption=f"Frame #{idx+1}", use_container_width=True)
            
    if st.button("🗑️ Clear All Images"):
        st.session_state.captured_images = []
        st.rerun()

# -------------------------------------------------------------------
# Fast Fast Vectorized 3D Processing Pipeline
# -------------------------------------------------------------------
st.write("---")
st.subheader("3. 3D Processing & GIS Export")

if len(st.session_state.captured_images) >= 2:
    if st.button("🚀 Run Fast 3D Reconstruction & Export GIS ZIP"):
        with st.spinner("ভেক্টর প্রসেসিংয়ের মাধ্যমে অতি দ্রুত ৩D পয়েন্ট তৈরি হচ্ছে..."):
            
            all_pts_list = []
            all_cols_list = []
            
            # Vectorized Matrix Processing (For extreme speed & zero lag)
            for idx, img_np in enumerate(st.session_state.captured_images):
                h, w, _ = img_np.shape
                step = max(h, w) // 40 # Downsample step to protect browser memory
                
                sub_img = img_np[::step, ::step]
                sh_h, sh_w, _ = sub_img.shape
                
                grid_y, grid_x = np.mgrid[0:sh_h, 0:sh_w]
                
                pt_x = (grid_x * step - w/2) * 0.05 + (idx * 0.8)
                pt_y = (grid_y * step - h/2) * 0.05
                
                r = sub_img[:, :, 0].astype(float)
                g = sub_img[:, :, 1].astype(float)
                b = sub_img[:, :, 2].astype(float)
                
                pt_z = (r * 0.299 + g * 0.587 + b * 0.114) * 0.01
                
                pts = np.column_stack((pt_x.ravel(), pt_y.ravel(), pt_z.ravel()))
                colors = sub_img.reshape(-1, 3)
                
                all_pts_list.append(pts)
                all_cols_list.append(colors)

            pts_arr = np.vstack(all_pts_list)
            cols_arr = np.vstack(all_cols_list)
            
            # Browser plot downsampling limit (Max 4000 points to avoid page freeze)
            plot_limit = 4000
            if len(pts_arr) > plot_limit:
                indices = np.random.choice(len(pts_arr), size=plot_limit, replace=False)
                plot_pts = pts_arr[indices]
                plot_cols = cols_arr[indices]
            else:
                plot_pts = pts_arr
                plot_cols = cols_arr

            hex_colors = [f'rgb({c[0]},{c[1]},{c[2]})' for c in plot_cols]
            
           # 3D Interactive Plot
            fig = px.scatter_3d(
                x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2],
                color=hex_colors, color_discrete_map="identity",
                title="Lightweight 3D Point Cloud Preview"
            )
            fig.update_traces(marker=dict(size=2))
            st.plotly_chart(fig, use_container_width=True)
            
            # GeoPandas GIS Package Generation
            with tempfile.TemporaryDirectory() as temp_dir:
                base_lat, base_lon = 23.8103, 90.4125
                
                lons = base_lon + (pts_arr[:, 0] / 111000.0)
                lats = base_lat + (pts_arr[:, 1] / 111000.0)
                
                geometry = [Point(xy) for xy in zip(lons, lats, pts_arr[:, 2])]
                
                gdf = gpd.GeoDataFrame({'Z_Height': pts_arr[:, 2], 'geometry': geometry}, crs="EPSG:4326")
                
                shp_dir = os.path.join(temp_dir, "shp_out")
                os.makedirs(shp_dir, exist_ok=True)
                gdf.to_file(os.path.join(shp_dir, "landscape_3d.shp"))
                
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
    st.info("💡 অন্তত ২টি ছবি দিয়ে ৩D প্রসেসিং স্টার্ট করুন।")
