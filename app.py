import streamlit as st
import numpy as np
import torch
import rasterio
import tempfile
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_opening

@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    state_dict = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

st.set_page_config(layout="wide")
st.title("Feature Extraction")
st.write("Upload a satellite image (.tif)")

uploaded_file = st.file_uploader("Upload Image", type=["tif"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    st.success("Image uploaded!")

    with rasterio.open(img_path) as src:
        image = src.read([1,2,3])
        image = np.transpose(image, (1,2,0))

    img_show = image.astype(np.float32)
    img_show = (img_show - img_show.min()) / (img_show.max() + 1e-8)

    st.subheader("Original Image")
    st.image(img_show, width="stretch")

    if st.button("Run Prediction"):
        img = image.astype(np.float32)
        valid_mask = np.all(img > 5, axis=2)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.transpose(img, (2,0,1))
        img_tensor = torch.tensor(img).unsqueeze(0).float()

        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.9).float().numpy()[0,0]

        pred[~valid_mask] = 0
        pred = binary_opening(pred, structure=np.ones((3,3)))

        st.subheader("Predicted Mask")
        st.image((pred * 255).astype(np.uint8))

        img_disp = image.astype(np.float32)
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() + 1e-8)
        img_disp = (img_disp * 255).astype(np.uint8)

        overlay = img_disp.copy()
        overlay[(pred == 1) & valid_mask] = [255, 0, 0]

        st.subheader("Overlay")
        st.image(overlay, width="stretch")

        with rasterio.open(img_path) as src:
            transform = src.transform
            crs = src.crs

        polygons = []

        for geom, val in shapes(pred.astype("uint8"), transform=transform):
            if val == 1:
                polygons.append(shape(geom))

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        gdf = gdf[gdf.area > 5]

        shp_path = "output.shp"
        gdf.to_file(shp_path)

        st.success("Shapefile generated!")

        with open(shp_path, "rb") as f:
            st.download_button("Download Shapefile", f, file_name="output.shp")