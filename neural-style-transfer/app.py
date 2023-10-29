import streamlit as st
import os
from train import neural_style_transfer


# Streamlit UI
# Streamlit UI
st.title("Neural Style Transfer")

uploaded_original = st.file_uploader("Upload the original image:")
uploaded_style = st.file_uploader("Upload the style image:")

total_steps_slider = st.slider("Total Steps", min_value=100, max_value=5000, value=2000)

output_path = st.text_input("Output Path", "output")


alpha = st.slider("Content Weight (alpha)", min_value=0.1, max_value=10.0, value=1.0)
beta = st.slider("Style Weight (beta)", min_value=0.001, max_value=0.1, value=0.01)

if st.button("Generate Style Transfer"):
    if uploaded_original is not None and uploaded_style is not None:
        st.text("Generating...")
        

        output_image_path = neural_style_transfer(
            uploaded_original, uploaded_style, output_path, total_steps_slider,
            alpha, beta
        )


        st.image(output_image_path, caption="Generated Image", use_column_width=True)

        st.success("Style transfer completed!")
