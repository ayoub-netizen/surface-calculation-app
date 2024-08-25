import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st

# Fonction pour convertir une couleur RGB en nom de couleur approximatif
def rgb_to_name(rgb):
    names = {
        'Red': (255, 0, 0),
        'Green': (0, 255, 0),
        'Blue': (0, 0, 255),
        'Yellow': (255, 255, 0),
        'Cyan': (0, 255, 255),
        'Magenta': (255, 0, 255),
        'White': (255, 255, 255),
        'Black': (0, 0, 0),
        'Gray': (128, 128, 128),
        'Orange': (255, 165, 0),
        'Pink': (255, 192, 203),
        'Purple': (128, 0, 128),
        'Brown': (165, 42, 42),
        'Beige': (245, 245, 220),
        'Olive': (128, 128, 0),
        'Maroon': (128, 0, 0),
        'Navy': (0, 0, 128),
        'Teal': (0, 128, 128),
        'Lime': (0, 255, 0),
        'Violet': (238, 130, 238),
    }
    min_dist = float('inf')
    closest_name = None
    for name, color in names.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(color))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

# Fonction pour traiter l'image
def process_image(image, num_clusters=20):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    color_names = [rgb_to_name(center) for center in centers]
    masks = {}
    for i, color_name in enumerate(color_names):
        mask = (labels == i).astype(np.uint8) * 255
        mask = mask.reshape(image_rgb.shape[:2])
        masks[color_name] = mask
    return masks, color_names

# Fonction pour calculer les surfaces
def calculate_surfaces(masks, color_names, reference_color_name, reference_surface):
    area_pixels_reference = np.sum(masks[reference_color_name] > 0)
    surfaces_m2 = {}
    for color_name, mask in masks.items():
        area_pixels = np.sum(mask > 0)
        if area_pixels_reference > 0 and (area_pixels / area_pixels_reference) * reference_surface >= 1:
            area_m2 = (area_pixels / area_pixels_reference) * reference_surface
            surfaces_m2[color_name] = area_m2
    return surfaces_m2

# Interface Streamlit
st.title("Surface Calculation from Colored Plan")

uploaded_file = st.file_uploader("Upload a plan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    masks, color_names = process_image(image)

    reference_color_name = st.selectbox("Select reference color", color_names)
    reference_surface = st.number_input("Enter the surface of the reference color in m²", min_value=0.0, step=0.1)

    if st.button("Calculate Surfaces"):
        surfaces = calculate_surfaces(masks, color_names, reference_color_name, reference_surface)
        st.write("Calculated Surfaces:")
        for color, surface in surfaces.items():
            st.write(f"{color}: {surface:.2f} m²")

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)