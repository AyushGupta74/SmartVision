import streamlit as st
import matplotlib.pyplot as plt
import cv2

from utils import preprocess_image
from predict import predict


st.set_page_config(
    page_title="SmartVision",
    layout="wide"
)

st.title("🔍 SmartVision")
st.subheader("Industrial Image Classification System")

st.info("""

**Supported Classes**

🏢 Buildings  
🌲 Forest  
🧊 Glacier  
⛰️ Mountain  
🌊 Sea  
🛣️ Street

Upload a JPG, JPEG, or PNG image belonging to one of the above classes.

""")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    original, processed = preprocess_image(uploaded_file)
    print("Processed Shape:", processed.shape)
    print("Min:", processed.min())
    print("Max:", processed.max())

    prediction, confidence, probabilities, top3, classes = predict(processed)
    print("Input Shape:", processed.shape)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original Image")

        st.image(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    with col2:

        st.markdown("### Processed Image")

        st.image(
            (processed[0]*255).astype("uint8"),
            use_container_width=True
        )

    st.success(f"Prediction : **{prediction}**")

    st.info(f"Confidence : **{confidence:.2f}%**")

    st.markdown("## Top Predictions")

    for idx in top3:

        st.write(
            f"{classes[idx]} : {probabilities[idx]*100:.2f}%"
        )

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(classes, probabilities)

    plt.xticks(rotation=45)

    plt.ylabel("Probability")

    plt.tight_layout()

    st.pyplot(fig)
