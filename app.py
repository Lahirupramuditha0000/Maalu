
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from PIL import Image

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="maalu.",
    page_icon="🐟",
    layout="centered"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0B1C22;
    color: #E4EEEE;
}

/* Main content width */
.block-container {
    max-width: 950px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Brand */
.brand {
    color: #79C9B9;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -2px;
    margin-bottom: 0;
}

.subtitle {
    color: #8CA6A8;
    font-size: 0.95rem;
    margin-top: -5px;
    margin-bottom: 35px;
}

/* Section headings */
.section-heading {
    color: #79C9B9;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Result cards */
.metric-card {
    background: #142A31;
    border: 1px solid #25434A;
    border-radius: 14px;
    padding: 22px;
    min-height: 125px;
}

.metric-label {
    color: #8CA6A8;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    font-weight: 600;
}

.metric-value {
    color: #79C9B9;
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 10px;
    overflow-wrap: anywhere;
}

/* Prediction result */
.prediction-card {
    background: #142A31;
    border: 1px solid #32665F;
    border-radius: 14px;
    padding: 25px;
    margin-top: 15px;
}

.prediction-title {
    color: #8CA6A8;
    font-size: 0.75rem;
    letter-spacing: 2px;
}

.prediction-species {
    color: #79C9B9;
    font-size: 2rem;
    font-weight: 700;
    margin-top: 8px;
}

/* General text */
.muted {
    color: #8CA6A8;
    font-size: 0.85rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background-color: #142A31;
    border: 1px dashed #32665F;
    border-radius: 14px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    background-color: #236D64;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background-color: #32877A;
    color: white;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background-color: #79C9B9;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #25434A;
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #607A7D;
    font-size: 0.75rem;
    margin-top: 55px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL AND CLASS NAMES
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fish_classifier.keras")


@st.cache_data
def load_classes():
    with open("class_names.json", "r") as file:
        return json.load(file)


try:
    model = load_model()
    class_names = load_classes()

except Exception as e:
    st.error(f"Could not load the model or class names: {e}")
    st.stop()


# Check model and class compatibility
if model.output_shape[-1] != len(class_names):
    st.error("The number of model outputs does not match the class names.")
    st.stop()


# =====================================================
# 4. HEADER
# =====================================================

st.markdown('<div class="brand">🐟 maalu.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">A simple look into the underwater world.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-heading">FISH SPECIES IDENTIFICATION</div>',
    unsafe_allow_html=True
)


# =====================================================
# 5. IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a fish image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    st.markdown("")

    identify_button = st.button("Identify Species")

    if identify_button:

        with st.spinner("Analyzing image..."):

            # Resize image
            resized_image = image.resize((224, 224))

            # Convert image to NumPy array
            image_array = np.array(resized_image, dtype=np.float32)

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # IMPORTANT:
            # Do not apply preprocess_input() here.
            # MobileNetV2 preprocessing is already inside the model.

            # Model prediction
            predictions = model.predict(image_array, verbose=0)[0]


        # =================================================
        # 6. PREDICTION ANALYSIS
        # =================================================

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]

        confidence = float(predictions[predicted_index]) * 100

        # Get top 5 predictions
        top_5 = np.argsort(predictions)[-5:][::-1]

        top_5_names = [class_names[i] for i in top_5]
        top_5_values = [float(predictions[i]) * 100 for i in top_5]


        st.markdown("---")

        st.markdown(
            '<div class="section-heading">MODEL OUTPUT / PREDICTION ANALYSIS</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # 7. MAIN RESULT CARDS
        # =================================================

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PREDICTED SPECIES</div>
                <div class="metric-value">{predicted_class}</div>
            </div>
            """, unsafe_allow_html=True)


        with col2:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MODEL CONFIDENCE</div>
                <div class="metric-value">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)


        # =================================================
        # 8. PREDICTION DISTRIBUTION
        # =================================================

        st.markdown(
            '<div class="section-heading">CLASS PROBABILITY DISTRIBUTION</div>',
            unsafe_allow_html=True
        )

        for rank, index in enumerate(top_5, start=1):

            species = class_names[index]
            probability = float(predictions[index])

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(
                    f"**{rank}. {species}**"
                )

            with col2:
                st.markdown(
                    f"<div style='text-align:right;color:#79C9B9;font-weight:600;'>"
                    f"{probability * 100:.2f}%</div>",
                    unsafe_allow_html=True
                )

            st.progress(probability)


        # =================================================
        # 9. TOP 5 COMPARISON CHART
        # =================================================

        st.markdown(
            '<div class="section-heading">PROBABILITY COMPARISON</div>',
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({
            "Species": top_5_names,
            "Probability (%)": top_5_values
        })

        st.bar_chart(
            chart_data,
            x="Species",
            y="Probability (%)",
            horizontal=True
        )


        # =================================================
        # 10. DETAILED RESULTS TABLE
        # =================================================

        st.markdown(
            '<div class="section-heading">DETAILED PREDICTION RESULTS</div>',
            unsafe_allow_html=True
        )

        result_table = pd.DataFrame({
            "Rank": range(1, len(top_5) + 1),
            "Species": top_5_names,
            "Probability (%)": [
                round(value, 2) for value in top_5_values
            ]
        })

        st.dataframe(
            result_table,
            hide_index=True,
            use_container_width=True
        )


        # =================================================
        # 11. MODEL OUTPUT INFORMATION
        # =================================================

        st.markdown(
            '<div class="section-heading">ANALYSIS INFORMATION</div>',
            unsafe_allow_html=True
        )

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.metric(
                label="Total Classes",
                value=len(class_names)
            )

        with info_col2:
            st.metric(
                label="Image Input Size",
                value="224 × 224"
            )

        st.caption(
            "Confidence scores represent the model's softmax outputs. "
            "They are not guaranteed measures of real-world accuracy."
        )


# =====================================================
# 12. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    maalu. · Exploring the underwater world, one image at a time.
</div>
""", unsafe_allow_html=True)
