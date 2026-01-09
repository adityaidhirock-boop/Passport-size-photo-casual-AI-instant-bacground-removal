import streamlit as st
import numpy as np
from PIL import Image
import cv2
import albumentations as A
import os
import io
import gc

# ========== EXACT MODEL ARCHITECTURE ==========
class ResNetEncoder(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=False):
        super().__init__()
        # pretrained=False prevents the app from hanging while trying to download weights
        resnet = models.resnet34(pretrained=pretrained)
        self.channels = [64, 64, 128, 256, 512]
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features

class HybridU2Net(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=False, num_classes=1):
        super().__init__()
        self.encoder = ResNetEncoder(encoder_name, pretrained)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
            nn.Sigmoid()
        )
        # Side outputs kept for state_dict compatibility
        self.side4 = nn.Conv2d(256, num_classes, 1)
        self.side3 = nn.Conv2d(128, num_classes, 1)
        self.side2 = nn.Conv2d(64, num_classes, 1)
        self.side1 = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        dec4 = self.dec4(enc_features[4])
        dec3 = self.dec3(dec4_skip)
        dec2 = self.dec2(dec3_skip)
        dec1 = self.dec1(dec2_skip)
        dec1_up = F.interpolate(dec1, size=enc_features[0].shape[2:], mode='bilinear', align_corners=False)
        main_output = self.final_conv(dec1_skip)
        return F.interpolate(main_output, size=input_size, mode='bilinear', align_corners=False), None

# ========== CONFIGURATION ==========
PASSPORT_SIZES = {
    "US Passport (2x2 inch)": {"width": 600, "height": 600},
    "India/EU Passport (35x45mm)": {"width": 413, "height": 531},
    "Custom Size": {"width": 600, "height": 600}
}

BACKGROUND_COLORS = {
    "white": (255, 255, 255),
    "blue": (70, 130, 180),
    "light_blue": (173, 216, 230),
    "transparent": None
}

# ========== CACHED MODEL LOADING ==========
@st.cache_resource
def load_ai_model(model_path):
    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found."
    try:
        model = HybridU2Net(encoder_name='resnet34', pretrained=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        # Freeze parameters to save memory
        for param in model.parameters():
            param.requires_grad = False
        return model, "success"
    except Exception as e:
        return None, str(e)

def preprocess_image(image_np):
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image_resized = cv2.resize(image_np, (320, 320))
    transformed = transform(image=image_resized)
    return transformed['image'].unsqueeze(0)

# ========== APP INTERFACE ==========
def main():
    st.set_page_config(page_title="AI Passport Photo", page_icon="📸")
    st.title("📸 AI Passport Photo Generator")

    model_path = "best_model.pth"
    model, status = load_ai_model(model_path)

    if model is None:
        st.error(f"Model Error: {status}")
        st.info("Check if 'best_model.pth' is in the root directory.")
        return

    st.sidebar.header("Settings")
    size_choice = st.sidebar.selectbox("Passport Format", list(PASSPORT_SIZES.keys()))
    bg_choice = st.sidebar.selectbox("Background Color", list(BACKGROUND_COLORS.keys()))
    
    uploaded_file = st.file_uploader("Upload a face photo", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="Original", width=300)

        if st.button("Generate Passport Photo"):
            with st.spinner("Processing..."):
                # Inference
                input_tensor = preprocess_image(img_np)
                    output, _ = model(input_tensor)
                    mask = output.cpu().numpy()[0, 0]
                
                # Post-process mask
                mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
                mask = (mask > 0.5).astype(np.float32)

                # Simple Crop & Background
                target = PASSPORT_SIZES[size_choice]
                res_img = cv2.resize(img_np, (target['width'], target['height']))
                res_mask = cv2.resize(mask, (target['width'], target['height']))
                
                color = BACKGROUND_COLORS[bg_choice]
                if color:
                    mask_3ch = np.stack([res_mask] * 3, axis=2)
                    bg = np.full_like(res_img, color, dtype=np.uint8)
                    final_img = (res_img * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
                    result_pil = Image.fromarray(final_img)
                else:
                    # Transparent
                    rgba = np.zeros((res_img.shape[0], res_img.shape[1], 4), dtype=np.uint8)
                    rgba[:,:,:3] = res_img
                    rgba[:,:,3] = (res_mask * 255).astype(np.uint8)
                    result_pil = Image.fromarray(rgba, 'RGBA')

                st.image(result_pil, caption="Result")
                
                # Download
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button("Download Image", buf.getvalue(), "passport.png", "image/png")
                
                # Clear memory
                del input_tensor, output, mask
                gc.collect()

if __name__ == "__main__":
    main()
