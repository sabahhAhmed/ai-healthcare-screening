
import streamlit as st
import hashlib
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ==============================
# Page Settings
# ==============================

st.set_page_config(
    page_title="AI Healthcare Screening",
    page_icon="🎀",
    layout="wide"
)


# ==============================
# Load Saved Models
# ==============================

@st.cache_resource
def load_models():

    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    dl_model = tf.keras.models.load_model("efficientnet_model.keras")

    return rf_model, scaler, dl_model


rf_model, scaler, dl_model = load_models()


# ==============================
# Login
# ==============================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


USERNAME = "admin"
PASSWORD_HASH = hash_password("admin123")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.title("🎀 AI-Based Healthcare Screening System")
    st.subheader("Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        entered_hash = hash_password(password)

        if username == USERNAME and entered_hash == PASSWORD_HASH:

            st.session_state.logged_in = True
            st.rerun()

        else:

            st.error("Invalid username or password")


# ==============================
# Main Application
# ==============================

else:

    st.title("🎀 AI-Based Healthcare Screening System")

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.rerun()


    vital_tab, mammogram_tab = st.tabs(
        ["🫀 Vital Signs", "🩻 Mammogram"]
    )


    # ==============================
    # Vital Signs
    # ==============================

    with vital_tab:

        st.header("Vital Signs Prediction")

        col1, col2 = st.columns(2)


        with col1:

            hr = st.number_input(
                "Heart Rate (BPM)",
                min_value=0.0,
                max_value=250.0,
                value=70.0
            )

            resp = st.number_input(
                "Respiratory Rate (BPM)",
                min_value=0.0,
                max_value=100.0,
                value=18.0
            )


        with col2:

            spo2 = st.number_input(
                "SpO₂ (%)",
                min_value=0.0,
                max_value=100.0,
                value=98.0
            )

            temp = st.number_input(
                "Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=37.0
            )


        if st.button("Predict Vital Signs"):

            input_data = np.array(
                [[hr, resp, spo2, temp]]
            )

            input_scaled = scaler.transform(input_data)

            prediction = rf_model.predict(input_scaled)[0]

            probabilities = rf_model.predict_proba(input_scaled)[0]

            class_index = list(rf_model.classes_).index(prediction)

            confidence = probabilities[class_index] * 100

            if str(prediction).lower() in ["1", "abnormal"]:

                st.error(
                    f"Abnormal — Confidence: {confidence:.2f}%"
                )

            else:

                st.success(
                    f"Normal — Confidence: {confidence:.2f}%"
                )


    # ==============================
    # Mammogram
    # ==============================

    with mammogram_tab:

        st.header("Breast Cancer Detection")

        uploaded_file = st.file_uploader(
            "Upload a mammogram image",
            type=["png", "jpg", "jpeg"]
        )


        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            image_resized = image.resize(
                (224, 224)
            )


            image_array = np.array(
                image_resized
            )


            image_input = np.expand_dims(
                image_array,
                axis=0
            )


            prediction = dl_model.predict(
                image_input,
                verbose=0
            )[0][0]


            if prediction >= 0.5:

                result = "Cancer"
                confidence = prediction * 100

            else:

                result = "No Cancer"
                confidence = (1 - prediction) * 100


            if result == "Cancer":

                st.error(
                    f"{result} — Confidence: {confidence:.2f}%"
                )

            else:

                st.success(
                    f"{result} — Confidence: {confidence:.2f}%"
                )


            # ==============================
            # Grad-CAM
            # ==============================

            efficientnet = dl_model.layers[0]

            last_conv_layer = efficientnet.get_layer(
                "top_conv"
            )


            grad_model = tf.keras.models.Model(
                inputs=efficientnet.input,
                outputs=[
                    last_conv_layer.output,
                    dl_model.layers[-1](
                        dl_model.layers[-2](
                            efficientnet.output
                        )
                    )
                ]
            )


            input_tensor = tf.convert_to_tensor(
                image_input,
                dtype=tf.float32
            )


            with tf.GradientTape() as tape:

                conv_outputs, predictions = grad_model(
                    input_tensor,
                    training=False
                )

                loss = predictions[:, 0]


            grads = tape.gradient(
                loss,
                conv_outputs
            )


            pooled_grads = tf.reduce_mean(
                grads,
                axis=(1, 2)
            )


            heatmap = (
                conv_outputs[0] *
                pooled_grads[0]
            )


            heatmap = tf.reduce_sum(
                heatmap,
                axis=-1
            )


            heatmap = tf.maximum(
                heatmap,
                0
            )


            heatmap = heatmap / (
                tf.reduce_max(heatmap) + 1e-8
            )


            heatmap_resized = tf.image.resize(
                heatmap[..., tf.newaxis],
                [224, 224]
            ).numpy().squeeze()


            # ==============================
            # Display Images
            # ==============================

            col1, col2 = st.columns(2)


            with col1:

                st.image(
                    image_resized,
                    caption="Original Mammogram",
                    use_container_width=True
                )


            with col2:

                fig, ax = plt.subplots()

                ax.imshow(image_resized)

                ax.imshow(
                    heatmap_resized,
                    cmap="jet",
                    alpha=0.5
                )

                ax.axis("off")

                ax.set_title(
                    "Grad-CAM — Important Region"
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)
```
