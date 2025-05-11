import streamlit as st
import numpy as np
from scipy.fftpack import dctn, idctn
from PIL import Image
import matplotlib.pyplot as plt
import io
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🔍 DCT + Quantization Compression Visualizer")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Standard JPEG quantization matrix
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def real_compress(image, quality=50):
    """Perform actual compression by discarding coefficients"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ycbcr = image.convert('YCbCr')
        y, cb, cr = ycbcr.split()
    
    channels = {
        'Y': np.array(y, dtype=np.float32),
        'Cb': np.array(cb, dtype=np.float32), 
        'Cr': np.array(cr, dtype=np.float32)
    }
    compressed_channels = {}
    
    # Scale quantization matrix
    quality = max(1, min(100, quality))
    scale = 5000/quality if quality < 50 else 200 - 2*quality
    Q_scaled = np.floor((Q * scale + 50) / 100).clip(1, 255).astype(np.float32)
    
    for name, channel in channels.items():
        h, w = channel.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
        
        compressed = np.zeros_like(padded, dtype=np.float32)
        nonzero_coeffs = 0
        total_coeffs = 0
        
        for i in range(0, padded.shape[0], 8):
            for j in range(0, padded.shape[1], 8):
                block = padded[i:i+8, j:j+8]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dct_block = dctn(block, norm='ortho')
                
                quantized = np.round(dct_block / Q_scaled)
                quantized[np.abs(quantized) < 0.5] = 0
                
                nonzero_coeffs += np.count_nonzero(quantized)
                total_coeffs += 64
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    compressed[i:i+8, j:j+8] = idctn(quantized * Q_scaled, norm='ortho')
        
        compressed = np.clip(compressed[:h, :w], 0, 255).astype(np.uint8)
        compressed_channels[name] = compressed
        st.sidebar.info(f"{name} channel: {nonzero_coeffs/total_coeffs:.1%} coefficients kept")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_img = Image.fromarray(compressed_channels['Y'], 'L')
        cb_img = Image.fromarray(compressed_channels['Cb'], 'L')
        cr_img = Image.fromarray(compressed_channels['Cr'], 'L')
        return Image.merge('YCbCr', (y_img, cb_img, cr_img)).convert('RGB')

# Streamlit UI
with st.sidebar:
    st.title("⚙ Settings")
    quality = st.slider("Compression Quality", 1, 100, 50)

col1, col2 = st.columns(2)

if uploaded_file:
    original_bytes = uploaded_file.getbuffer().nbytes
    
    with st.spinner("Compressing..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original_img = Image.open(uploaded_file)
            compressed_img = real_compress(original_img, quality)
            
            buf = io.BytesIO()
            compressed_img.save(buf, format="JPEG", quality=quality, subsampling=0)
            compressed_bytes = buf.getbuffer().nbytes
    
    # Original image - using use_container_width instead of use_column_width
    with col1:
        st.subheader("Original Image")
        st.image(original_img, use_container_width=True)
        st.caption(f"Size: {original_bytes/1024:.1f} KB")
        st.markdown(f"<h3 style='text-align: left;'>Compression Ratio: <b>{original_bytes/compressed_bytes:.1f}x</b></h3>", 
                   unsafe_allow_html=True)
    
    # Compressed image - using use_container_width instead of use_column_width
    with col2:
        st.subheader(f"Compressed (Quality: {quality})")
        st.image(compressed_img, use_container_width=True)
        st.caption(f"Size: {compressed_bytes/1024:.1f} KB")
    
    # Download button
    st.download_button(
        label="Download Compressed Image",
        data=buf.getvalue(),
        file_name="compressed.jpg",
        mime="image/jpeg"
    )
else:
    st.info("☝ Upload an image to get started")

plt.close('all')