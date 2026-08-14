import io
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification

# Page config
st.set_page_config(
    page_title="ViT Image Classification Inspector",
    page_icon="🛡️",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .verdict-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .verdict-real {
        background-color: rgba(46, 125, 50, 0.15);
        border: 2px solid #2e7d32;
        color: #2e7d32;
    }
    .verdict-fake {
        background-color: rgba(198, 40, 40, 0.15);
        border: 2px solid #c62828;
        color: #c62828;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ ViT Model Inspector")
st.write("Upload an image to inspect model prediction and confidence.")

MODEL_PATH = "./usmfe_vit_ultimate_90_model"

@st.cache_resource
def load_vit_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return processor, model, device

# Load model
try:
    with st.spinner("Loading ViT model..."):
        processor, model, device = load_vit_model(MODEL_PATH)
    st.success(f"Model loaded successfully on **{device.type.upper()}**!")
except Exception as e:
    st.error(f"Failed to load model from `{MODEL_PATH}`: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file is not None:
    try:
        # Read image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Inference
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probabilities = F.softmax(logits, dim=-1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
        
        pred_idx = pred_idx.item()
        confidence_pct = confidence.item() * 100
        
        # Explicit class mapping: Index 0 / LABEL_0 -> FAKE, Index 1 / LABEL_1 -> REAL
        LABEL_MAP = {
            0: "FAKE",
            1: "REAL",
            "0": "FAKE",
            "1": "REAL",
            "LABEL_0": "FAKE",
            "LABEL_1": "REAL"
        }
        
        raw_label = model.config.id2label.get(pred_idx, pred_idx) if hasattr(model.config, 'id2label') and model.config.id2label else pred_idx
        label_name = LABEL_MAP.get(raw_label, LABEL_MAP.get(pred_idx, "REAL" if pred_idx == 1 else "FAKE"))
        
        # Styling condition
        if label_name == "REAL":
            verdict_class = "verdict-real"
        else:
            verdict_class = "verdict-fake"


            
        st.markdown(
            f"""
            <div class="verdict-box {verdict_class}">
                <h2 style="margin:0;">Verdict: {label_name}</h2>
                <h3 style="margin:5px 0 0 0;">Confidence: {confidence_pct:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display preview image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
