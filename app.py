"""
CNN Assignment Demonstration — Streamlit App
=============================================
Interactive demonstration of key CNN concepts including:
- Part 1: Convolution matrix operations (manual 2D convolution, kernel effects,
  stride/padding calculations)
- Part 2: CNN architecture, feature maps, pooling, and classification
  (RGB vs grayscale structure, activation functions, softmax output,
  and importing pre-trained PyTorch data)

Dependencies: streamlit, numpy, matplotlib
Training script (train_cnn.py) requires: torch (PyTorch)
Run with: streamlit run app.py

Author: Brogan McKenzie and Adonijah Farner
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')                          # Non-interactive backend required for Streamlit rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches          # Used for drawing labeled boxes in diagrams
from matplotlib.gridspec import GridSpec        # Flexible subplot layout for side-by-side comparisons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 3D polygon rendering for RGB channel viz

st.set_page_config(page_title="CNN Demo", page_icon="CNN", layout="wide")

# ----------------------------------------------
# Sidebar Navigation
# ----------------------------------------------
st.sidebar.title("CNN Demo")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "1. Convolution Operation",
        "2. Kernel Effects",
        "3. Stride & Padding",
        "4. RGB vs Grayscale",
        "5. CNN Architecture",
        "6. Activation Functions",
        "7. Pooling Layers",
        "8. CNN Steps Flowchart",
        "9. Softmax Output",
        "10. Train a CNN (Live)",
    ],
)

# ----------------------------------------------
# Helper: show a matplotlib figure in Streamlit
# ----------------------------------------------
def show(fig):
    """Render a matplotlib figure in the Streamlit UI and free memory."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)  # Prevent memory leaks from accumulating open figures


# ============================================================
# OVERVIEW
# ============================================================
if section == "Overview":
    st.title("Convolutional Neural Network — Interactive Demo")
    st.markdown("""
    This app walks through the core concepts behind **Convolutional Neural Networks (CNNs)**
    in two parts:

    **Part 1 — Convolution Math**
    - Manual 2-D convolution with step-by-step arithmetic
    - Effects of different kernels (blur, sharpen, edge detect, Sobel)
    - How stride and padding change the output size

    **Part 2 — CNN Concepts & Live Training**
    - RGB vs grayscale image structure
    - End-to-end CNN architecture diagram
    - Activation functions, pooling, softmax
    - **Live training** of a CNN on a synthetic dataset (runs in your browser!)

    Use the sidebar to jump to any section.
    """)


# ============================================================
# 1 - CONVOLUTION OPERATION
# ============================================================
elif section == "1. Convolution Operation":
    st.header("Part 1: Manual Convolution Operation")

    st.markdown("""
    A 2-D convolution slides a small **kernel** (filter) across an input matrix,
    computing the element-wise product and summing at each position to produce a
    feature map.
    """)

    col_input, col_kernel = st.columns(2)
    with col_input:
        st.subheader("Input Matrix (5×5)")
        default_input = "1 2 3 0 1\n0 1 2 3 1\n1 0 1 2 0\n2 1 0 1 3\n1 2 1 0 2"
        input_text = st.text_area("Edit input values (space-separated rows):", value=default_input, height=140)
    with col_kernel:
        st.subheader("Kernel (3×3)")
        kernel_choice = st.selectbox("Preset kernel:", ["Edge Detection", "Identity", "Blur (Box)", "Sharpen", "Sobel X", "Sobel Y", "Custom"])
        # Predefined kernels for common convolution operations
        # Edge Detection: highlights boundaries by subtracting neighbors from center
        # Identity: passes input through unchanged - useful as a baseline
        # Blur (Box): averages a 3x3 neighborhood to smooth noise
        # Sharpen: amplifies center pixel relative to neighbors for detail
        # Sobel X/Y: gradient approximations for horizontal/vertical edges
        preset_kernels = {
            "Edge Detection": [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
            "Identity":       [[0,0,0],[0,1,0],[0,0,0]],
            "Blur (Box)":     [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
            "Sharpen":        [[0,-1,0],[-1,5,-1],[0,-1,0]],
            "Sobel X":        [[-1,0,1],[-2,0,2],[-1,0,1]],
            "Sobel Y":        [[-1,-2,-1],[0,0,0],[1,2,1]],
        }
        if kernel_choice != "Custom":
            kernel = np.array(preset_kernels[kernel_choice])
            st.text(f"{kernel}")
        else:
            kernel_text = st.text_area("Enter 3×3 kernel:", value="-1 -1 -1\n-1 8 -1\n-1 -1 -1", height=100)
            kernel = np.array([[float(v) for v in row.split()] for row in kernel_text.strip().split("\n")])

    # Parse input matrix from user-editable text area
    try:
        input_matrix = np.array([[float(v) for v in row.split()] for row in input_text.strip().split("\n")])
    except Exception:
        st.error("Could not parse input matrix. Use space-separated numbers, one row per line.")
        st.stop()

    # Compute valid convolution output dimensions - output shrinks without padding
    # Formula: out_dim = input_dim - kernel_dim + 1
    kh, kw = kernel.shape
    out_h = input_matrix.shape[0] - kh + 1
    out_w = input_matrix.shape[1] - kw + 1

    if out_h < 1 or out_w < 1:
        st.error("Input is too small for this kernel size.")
        st.stop()

    # Manual 2D cross-correlation - what deep learning frameworks call "convolution"
    # Slide kernel across every valid position, compute element-wise product and sum
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            # Extract region overlapping with kernel at position (i, j) and dot product
            output[i, j] = np.sum(input_matrix[i:i+kh, j:j+kw] * kernel)

    # Walk through convolution arithmetic at position (0,0) for the user
    with st.expander("Step-by-step at position (0, 0)", expanded=True):
        region = input_matrix[:kh, :kw]  # Top-left patch matching kernel size
        st.markdown(f"**Region extracted:** `{region.tolist()}`")
        st.markdown(f"**Element-wise product:** `{(region * kernel).tolist()}`")
        st.markdown(f"**Sum → {np.sum(region * kernel):.2f}**")

    # Side-by-side visualization: input matrix, kernel, and resulting feature map
    # Blues for input, diverging RdBu for kernel weights, coolwarm for output
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("2D Convolution Operation", fontsize=14, fontweight='bold')

    im0 = axes[0].imshow(input_matrix, cmap='Blues', aspect='equal')
    axes[0].set_title(f"Input ({input_matrix.shape[0]}×{input_matrix.shape[1]})", fontsize=11)
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            axes[0].text(j, i, f"{input_matrix[i,j]:.0f}", ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Diverging colormap centered at zero so negative/positive weights are distinct
    im1 = axes[1].imshow(kernel, cmap='RdBu_r', aspect='equal', vmin=-max(abs(kernel.min()), abs(kernel.max())), vmax=max(abs(kernel.min()), abs(kernel.max())))
    axes[1].set_title(f"Kernel ({kh}×{kw})", fontsize=11)
    for i in range(kh):
        for j in range(kw):
            axes[1].text(j, i, f"{kernel[i,j]:.1f}", ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(output, cmap='coolwarm', aspect='equal')
    axes[2].set_title(f"Output ({out_h}×{out_w})", fontsize=11)
    for i in range(out_h):
        for j in range(out_w):
            axes[2].text(j, i, f"{output[i,j]:.1f}", ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    show(fig)


# ============================================================
# 2 - KERNEL EFFECTS
# ============================================================
elif section == "2. Kernel Effects":
    st.header("Effect of Different Convolution Kernels")

    st.markdown("Each kernel extracts different features from the same input image.")

    # Simple synthetic 8x8 test image with a bright square and inner region
    # Clear edges and flat areas make each kernel effect easy to see
    np.random.seed(42)
    img = np.zeros((8, 8))
    img[2:6, 2:6] = 1.0   # Outer bright square
    img[3:5, 3:5] = 0.5   # Inner dimmer region to create internal edges

    # Dictionary of standard 3x3 kernels used in image processing and CNNs
    kernels = {
        "Identity": np.array([[0,0,0],[0,1,0],[0,0,0]]),
        "Blur (Box)": np.ones((3,3)) / 9.0,
        "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        "Edge Detect": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
        "Sobel X": np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
        "Sobel Y": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Effect of Different Convolution Kernels", fontsize=14, fontweight='bold')

    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("Original Image", fontsize=10)

    # Apply each kernel to the test image using manual convolution then plot results
    for idx, (name, k) in enumerate(kernels.items()):
        row = (idx + 1) // 4
        col = (idx + 1) % 4
        # Valid convolution: output shrinks by (kernel_size - 1) in each dimension
        oh = img.shape[0] - k.shape[0] + 1
        ow = img.shape[1] - k.shape[1] + 1
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[i, j] = np.sum(img[i:i+3, j:j+3] * k)  # Element-wise multiply and sum
        axes[row, col].imshow(out, cmap='gray')
        axes[row, col].set_title(name, fontsize=10)

    axes[1, 3].axis('off')
    plt.tight_layout()
    show(fig)

    with st.expander("Kernel matrices"):
        for name, k in kernels.items():
            st.markdown(f"**{name}**")
            st.text(f"{k}")


# ============================================================
# 3 - STRIDE & PADDING
# ============================================================
elif section == "3. Stride & Padding":
    st.header("Stride & Padding Effects")

    st.markdown("Stride and padding control the spatial dimensions of the output feature map.")

    col1, col2, col3 = st.columns(3)
    with col1:
        input_size = st.slider("Input size", 4, 12, 6)
    with col2:
        kernel_size = st.slider("Kernel size", 2, 5, 3)
    with col3:
        st.write("")  # spacer

    # Four common stride/padding combos to show how each affects output size
    # Padding adds zeros around the border; stride controls step size
    configs = [
        ("Stride=1, Pad=0", 1, 0),   # Standard valid convolution - output shrinks
        ("Stride=1, Pad=1", 1, 1),   # "Same" padding - output matches input size for 3x3
        ("Stride=2, Pad=0", 2, 0),   # Strided convolution - aggressive downsampling
        ("Stride=2, Pad=1", 2, 1),   # Strided with padding - moderate downsampling
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Output Dimensions: {input_size}×{input_size} Input, {kernel_size}×{kernel_size} Kernel",
        fontsize=12, fontweight='bold',
    )

    for idx, (title, stride, pad) in enumerate(configs):
        # Output size formula: out = floor((input + 2*pad - kernel) / stride) + 1
        padded = input_size + 2 * pad
        out_size = (padded - kernel_size) // stride + 1
        ax = axes[idx]

        # Draw grid lines for the padded input dimensions
        for i in range(padded + 1):
            ax.axhline(y=i, color='lightblue', linewidth=0.5)
            ax.axvline(x=i, color='lightblue', linewidth=0.5)

        # Show zero-padding cells in yellow so they stand out from real data
        if pad > 0:
            for i in range(padded):
                for j in range(padded):
                    if i < pad or i >= padded - pad or j < pad or j >= padded - pad:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightyellow', edgecolor='gray', linewidth=0.5))

        for i in range(pad, pad + input_size):
            for j in range(pad, pad + input_size):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightblue', edgecolor='gray', linewidth=0.5))

        ax.add_patch(plt.Rectangle((0, 0), kernel_size, kernel_size, fill=False, edgecolor='red', linewidth=2.5))
        ax.set_xlim(0, padded)
        ax.set_ylim(padded, 0)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\nOutput: {out_size}×{out_size}", fontsize=9)
        formula = f"({padded}−{kernel_size})/{stride}+1 = {out_size}"
        ax.text(padded / 2, padded + 0.5, formula, ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    show(fig)

    st.info(
        f"**Formula:** output = (input + 2×padding − kernel) / stride + 1"
    )


# ============================================================
# 4 - RGB vs GRAYSCALE
# ============================================================
elif section == "4. RGB vs Grayscale":
    st.header("RGB vs Grayscale Image Structure")

    st.markdown("""
    - **RGB images** have shape **(H × W × 3)** — three channels for red, green, and blue.
    - **Grayscale images** have shape **(H × W × 1)** — a single intensity channel.
    """)

    tab1, tab2 = st.tabs(["3-D Visualization", "Block Diagram"])

    with tab1:
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(1, 2, figure=fig, wspace=0.4)

        # 3D plot: stack three semi-transparent planes for R, G, B channels
        # Each plane at a different z-level, color-coded by channel
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax1.set_title("RGB Color Image\n(H × W × 3 Channels)", fontsize=12, fontweight='bold', pad=15)
        for idx, (color, label) in enumerate(zip(['red','green','blue'], ['Red','Green','Blue'])):
            z = idx * 2  # Offset each channel along z-axis for visual separation
            verts = [[(0,0,z),(4,0,z),(4,4,z),(0,4,z)]]
            ax1.add_collection3d(Poly3DCollection(verts, alpha=0.3, facecolor=color, edgecolor=color, linewidth=2))
            ax1.text(4.5, 2, z, f"{label} Channel", fontsize=8, color=color)
        ax1.set_xlim(0,5); ax1.set_ylim(0,5); ax1.set_zlim(-1,6)
        ax1.set_xlabel('Width'); ax1.set_ylabel('Height'); ax1.set_zlabel('Channels')
        ax1.view_init(elev=20, azim=-60)

        # Grayscale comparison: single 2D matrix with annotated H and W dimensions
        # Random noise as placeholder pixel data to show single-channel structure
        np.random.seed(0)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Grayscale Image\n(H × W × 1 Channel)", fontsize=12, fontweight='bold')
        ax2.imshow(np.random.rand(6,6), cmap='gray', aspect='equal')
        ax2.set_xlabel("Width"); ax2.set_ylabel("Height")
        ax2.annotate('', xy=(6.3,0), xytext=(6.3,6), arrowprops=dict(arrowstyle='<->', lw=1.5))
        ax2.text(6.8, 3, 'H', fontsize=12, ha='center', fontweight='bold')
        ax2.annotate('', xy=(0,-0.8), xytext=(6,-0.8), arrowprops=dict(arrowstyle='<->', lw=1.5))
        ax2.text(3, -1.5, 'W', fontsize=12, ha='center', fontweight='bold')

        plt.tight_layout()
        show(fig)

    with tab2:
        # Block diagram contrasting RGB tensor shape (H,W,3) vs grayscale (H,W,1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Block Diagram: RGB vs Grayscale Image Structure", fontsize=14, fontweight='bold')

        ax = axes[0]
        ax.set_xlim(0,10); ax.set_ylim(0,8); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("RGB Image (3 Channels)", fontsize=12)
        for x_off, y_off, label, color, alpha in [
            (0.5,0.5,'Blue','#4444FF',0.4),(1.0,1.0,'Green','#44BB44',0.5),(1.5,1.5,'Red','#FF4444',0.5)
        ]:
            ax.add_patch(mpatches.FancyBboxPatch((x_off+1,y_off+1),4,4,boxstyle="round,pad=0.1",facecolor=color,alpha=alpha,edgecolor='black',linewidth=1.5))
        ax.text(3.5,3.5,"R",fontsize=20,ha='center',va='center',fontweight='bold',color='white')
        ax.text(3.0,3.0,"G",fontsize=20,ha='center',va='center',fontweight='bold',color='white')
        ax.text(2.5,2.5,"B",fontsize=20,ha='center',va='center',fontweight='bold',color='white')
        ax.text(5,0.3,"Shape: (H, W, 3)",fontsize=11,ha='center',fontweight='bold',bbox=dict(boxstyle='round,pad=0.3',facecolor='lightyellow',edgecolor='orange'))
        ax.annotate('',xy=(7,5.5),xytext=(7,1.5),arrowprops=dict(arrowstyle='<->',lw=2))
        ax.text(7.5,3.5,'3\nChannels',fontsize=10,ha='left',va='center',fontweight='bold')

        ax = axes[1]
        ax.set_xlim(0,10); ax.set_ylim(0,8); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Grayscale Image (1 Channel)", fontsize=12)
        ax.add_patch(mpatches.FancyBboxPatch((2,1.5),4,4,boxstyle="round,pad=0.1",facecolor='gray',alpha=0.5,edgecolor='black',linewidth=2))
        ax.text(4,3.5,"Intensity\nValues\n(0-255)",fontsize=12,ha='center',va='center',fontweight='bold')
        ax.text(5,0.3,"Shape: (H, W, 1)",fontsize=11,ha='center',fontweight='bold',bbox=dict(boxstyle='round,pad=0.3',facecolor='lightyellow',edgecolor='orange'))
        ax.annotate('',xy=(6.5,1.5),xytext=(6.5,5.5),arrowprops=dict(arrowstyle='<->',lw=2))
        ax.text(7,3.5,'H',fontsize=12,ha='left',va='center',fontweight='bold')
        ax.annotate('',xy=(2,6.2),xytext=(6,6.2),arrowprops=dict(arrowstyle='<->',lw=2))
        ax.text(4,6.8,'W',fontsize=12,ha='center',va='center',fontweight='bold')

        plt.tight_layout()
        show(fig)


# ============================================================
# 5 - CNN ARCHITECTURE
# ============================================================
elif section == "5. CNN Architecture":
    st.header("CNN Architecture Diagram")

    st.markdown("""
    A typical CNN alternates **convolutional layers** (feature extraction) with **pooling layers**
    (dimensionality reduction), then passes the flattened features through **fully connected layers**
    for classification.
    """)

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.set_xlim(0, 20); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title("Convolutional Neural Network Architecture", fontsize=16, fontweight='bold', pad=20)

    # Each tuple: x_pos, label, fill_color, box_width
    # Diagram flows left-to-right from raw input through conv+pool to FC+softmax
    layers_info = [
        (1, "Input\nImage\n(32×32×3)", "#E8F5E9", 1.2),
        (3.5, "Conv Layer 1\n+ ReLU\n(32 filters)", "#BBDEFB", 1.2),
        (6, "Max\nPooling\n(2×2)", "#F8BBD0", 0.9),
        (8.2, "Conv Layer 2\n+ ReLU\n(64 filters)", "#BBDEFB", 1.2),
        (10.7, "Max\nPooling\n(2×2)", "#F8BBD0", 0.9),
        (13, "Flatten", "#FFF9C4", 0.8),
        (15, "Fully\nConnected\n(128 units)", "#D1C4E9", 1.2),
        (17.2, "Dropout\n(0.5)", "#FFCCBC", 0.8),
        (19, "Output\n(Softmax)\n10 classes", "#C8E6C9", 1.0),
    ]

    for x, label, color, width in layers_info:
        rect = mpatches.FancyBboxPatch((x - width/2, 1.5), width, 3.5,
                                        boxstyle="round,pad=0.15", facecolor=color,
                                        edgecolor='#333333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 3.25, label, fontsize=8, ha='center', va='center', fontweight='bold')

    # Arrows between consecutive layer blocks to show data flow
    arrow_xs = [1.6, 4.1, 6.45, 8.8, 11.15, 13.4, 15.6, 17.6]
    for xs in arrow_xs:
        ax.annotate('', xy=(xs + 0.7, 3.25), xytext=(xs, 3.25),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    # Dashed boxes group layers into Feature Extraction and Classification stages
    ax.add_patch(mpatches.FancyBboxPatch((0.2, 0.2), 11.5, 0.8, boxstyle="round,pad=0.1",
                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1.5, linestyle='--'))
    ax.text(6, 0.6, "Feature Extraction", fontsize=10, ha='center', color='#1976D2', fontweight='bold')

    ax.add_patch(mpatches.FancyBboxPatch((12.2, 0.2), 7.5, 0.8, boxstyle="round,pad=0.1",
                 facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.5, linestyle='--'))
    ax.text(16, 0.6, "Classification", fontsize=10, ha='center', color='#7B1FA2', fontweight='bold')

    plt.tight_layout()
    show(fig)


# ============================================================
# 6 - ACTIVATION FUNCTIONS
# ============================================================
elif section == "6. Activation Functions":
    st.header("Common Activation Functions")

    st.markdown("Activation functions introduce **non-linearity**, allowing the network to learn complex patterns.")

    # X values across a symmetric range so behavior around zero is visible
    x = np.linspace(-5, 5, 200)

    # Compute each activation function for plotting
    # ReLU: zeroes out negatives, passes positives - most common in CNNs
    # Leaky ReLU: small slope for negatives prevents "dying neuron" problem
    # Sigmoid: squashes to (0,1) but suffers vanishing gradients
    # Tanh: zero-centered version of sigmoid, squashes to (-1,1)
    activations = {
        "ReLU": np.maximum(0, x),
        "Leaky ReLU (α=0.1)": np.where(x > 0, x, 0.1 * x),
        "Sigmoid": 1 / (1 + np.exp(-x)),
        "Tanh": np.tanh(x),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.suptitle("Common Activation Functions in CNNs", fontsize=14, fontweight='bold')
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for idx, (name, y) in enumerate(activations.items()):
        ax = axes[idx]
        ax.plot(x, y, color=colors[idx], linewidth=2.5)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel("x"); ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    show(fig)

    st.markdown("""
    | Function | Formula | Key Property |
    |---|---|---|
    | **ReLU** | max(0, x) | Fast, sparse activation; can suffer "dying ReLU" |
    | **Leaky ReLU** | max(αx, x) | Avoids dying neurons with small negative slope |
    | **Sigmoid** | 1 / (1 + e⁻ˣ) | Squashes to (0, 1); used in binary classification |
    | **Tanh** | (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ) | Zero-centered; squashes to (−1, 1) |
    """)


# ============================================================
# 7 - POOLING LAYERS
# ============================================================
elif section == "7. Pooling Layers":
    st.header("Pooling Layer Demonstration")

    st.markdown("Pooling reduces spatial dimensions while retaining the most important information.")

    # Example 4x4 feature map to demonstrate pooling
    # Values chosen to make the difference between max and average pooling obvious
    feature_map = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [3, 2, 7, 8],
        [4, 1, 3, 5]
    ])

    # Apply 2x2 pooling with stride 2 - non-overlapping windows
    # Max pooling: keeps only the largest value in each 2x2 region
    # Avg pooling: computes mean of each region - smoother but loses sharp features
    max_pool = np.zeros((2, 2))
    avg_pool = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            region = feature_map[i*2:i*2+2, j*2:j*2+2]  # 2x2 non-overlapping block
            max_pool[i, j] = np.max(region)   # Retain strongest signal
            avg_pool[i, j] = np.mean(region)  # Smooth by averaging

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Pooling Operations (2×2 Pool Size, Stride 2)", fontsize=14, fontweight='bold')

    im0 = axes[0].imshow(feature_map, cmap='YlOrRd', aspect='equal')
    axes[0].set_title("Input Feature Map (4×4)", fontsize=11)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, str(feature_map[i,j]), ha='center', va='center', fontsize=14, fontweight='bold')
    # Dashed colored rectangles outline the four 2x2 pooling windows on the input
    pool_colors = ['blue','green','red','purple']
    for i in range(2):
        for j in range(2):
            axes[0].add_patch(plt.Rectangle((j*2-0.5, i*2-0.5), 2, 2, fill=False,
                              edgecolor=pool_colors[i*2+j], linewidth=2.5, linestyle='--'))

    im1 = axes[1].imshow(max_pool, cmap='YlOrRd', aspect='equal')
    axes[1].set_title("Max Pooling (2×2)", fontsize=11)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(int(max_pool[i,j])), ha='center', va='center', fontsize=16, fontweight='bold')

    im2 = axes[2].imshow(avg_pool, cmap='YlOrRd', aspect='equal')
    axes[2].set_title("Average Pooling (2×2)", fontsize=11)
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f"{avg_pool[i,j]:.1f}", ha='center', va='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    show(fig)


# ============================================================
# 8 - CNN STEPS FLOWCHART
# ============================================================
elif section == "8. CNN Steps Flowchart":
    st.header("Steps in a Convolutional Neural Network")

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis('off')
    ax.set_title("Steps in a Convolutional Neural Network", fontsize=16, fontweight='bold', pad=20)

    # Steps ordered top-to-bottom representing the forward pass through a CNN
    steps = [
        (8, 11,  "1. Input Image",           "Raw pixel data\n(e.g., 224×224×3 RGB)",       "#E8F5E9", "#388E3C"),
        (8, 9.5, "2. Convolution Layer",      "Apply learnable filters\nto extract features", "#BBDEFB", "#1565C0"),
        (8, 8,   "3. Activation (ReLU)",       "Introduce nonlinearity\nf(x) = max(0, x)",    "#FFF9C4", "#F57F17"),
        (8, 6.5, "4. Pooling Layer",           "Reduce spatial dimensions\n(Max/Avg Pooling)", "#F8BBD0", "#C2185B"),
        (8, 5,   "5. Repeat Steps 2-4",        "Stack multiple Conv+ReLU+Pool\nblocks",       "#E1BEE7", "#7B1FA2"),
        (8, 3.5, "6. Flatten",                 "Convert 2D feature maps\nto 1D vector",       "#FFCCBC", "#E64A19"),
        (8, 2,   "7. Fully Connected Layer",   "Learn non-linear combinations\nof features",  "#D1C4E9", "#512DA8"),
        (8, 0.5, "8. Output Layer (Softmax)",  "Produce class probabilities",                 "#C8E6C9", "#2E7D32"),
    ]

    for x, y, title, desc, fill, border in steps:
        rect = mpatches.FancyBboxPatch((x - 3.5, y - 0.55), 7, 1.1,
                                        boxstyle="round,pad=0.15", facecolor=fill,
                                        edgecolor=border, linewidth=2)
        ax.add_patch(rect)
        ax.text(x - 3, y + 0.1, title, fontsize=10, fontweight='bold', va='center', color=border)
        ax.text(x + 0.5, y - 0.1, desc, fontsize=8, va='center', ha='center', color='#333333')

    # Connect consecutive steps with downward arrows to show sequential data flow
    for i in range(len(steps) - 1):
        y_start = steps[i][1] - 0.55
        y_end = steps[i+1][1] + 0.55
        ax.annotate('', xy=(8, y_end), xytext=(8, y_start),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=2))

    plt.tight_layout()
    show(fig)


# ============================================================
# 9 - SOFTMAX OUTPUT
# ============================================================
elif section == "9. Softmax Output":
    st.header("Softmax Output Layer")

    st.markdown("""
    The **softmax** function converts raw logits from the final layer into a probability
    distribution over classes. Each output sums to 1.0.
    """)

    # Example raw logits - unnormalized scores from a CNN final dense layer
    # Higher logits correspond to stronger predictions for that class
    logits = np.array([2.1, 0.5, -1.2, 0.8, 3.5, -0.3, 1.1, -2.0, 0.3, 1.7])
    class_names = ['Airplane','Auto','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

    # Numerically stable softmax: subtract max before exponentiating to prevent overflow
    # Then normalize so all probabilities sum to 1.0
    exp_logits = np.exp(logits - np.max(logits))
    softmax = exp_logits / np.sum(exp_logits)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Output Layer: From Logits to Softmax Probabilities", fontsize=14, fontweight='bold')

    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    ax1.barh(class_names, logits, color=colors)
    ax1.set_title("Raw Logits (Before Softmax)", fontsize=11)
    ax1.set_xlabel("Logit Value"); ax1.axvline(x=0, color='gray', linewidth=0.5)

    bars = ax2.barh(class_names, softmax * 100, color=colors)
    ax2.set_title("Softmax Probabilities", fontsize=11)
    ax2.set_xlabel("Probability (%)")
    for bar, prob in zip(bars, softmax):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{prob*100:.1f}%', va='center', fontsize=9)

    predicted = np.argmax(softmax)
    ax2.text(0.95, 0.05, f"Prediction: {class_names[predicted]}\n({softmax[predicted]*100:.1f}%)",
             transform=ax2.transAxes, fontsize=11, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='green'))

    plt.tight_layout()
    show(fig)

    with st.expander("Softmax formula"):
        st.latex(r"\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}")


# ============================================================
# ============================================================
# 10 - CNN TRAINING RESULTS
# ============================================================
elif section == "10. Pre-Trained CNN":
    st.header("CNN Training Results on Synthetic Data")

    st.markdown("""
    A CNN was trained on a synthetic 10-class geometric pattern dataset (5000 training images,
    32x32 RGB). The model uses two Conv-BN-Conv-Pool-Dropout blocks followed by a dense
    classification head. Results below were generated with PyTorch.
    """)

    # Load pre-computed results from local training run
    import os
    results_path = os.path.join(os.path.dirname(__file__), 'cnn_results.npz')
    if not os.path.exists(results_path):
        st.error("cnn_results.npz not found. Run `python train_cnn.py` first to generate training results.")
    else:
        data = np.load(results_path, allow_pickle=True)
        class_names = list(data['class_names'])
        test_acc = float(data['test_acc'])
        test_loss = float(data['test_loss'])

        st.success(f"**Test Accuracy: {test_acc:.2%}** | Test Loss: {test_loss:.4f}")

        # -- Sample images --
        st.subheader("Dataset Samples")
        sample_images = data['sample_images']
        sample_labels = data['sample_labels']
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle("Synthetic Dataset Samples", fontsize=14, fontweight='bold')
        for i in range(10):
            ax = axes[i // 5, i % 5]
            ax.imshow(sample_images[i])
            ax.set_title(f"Class {sample_labels[i]}: {class_names[sample_labels[i]]}", fontsize=9)
            ax.axis('off')
        plt.tight_layout()
        show(fig)

        # -- Training curves --
        st.subheader("Training History")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Training History", fontsize=14, fontweight='bold')
        ax1.plot(data['epoch_acc'], label='Train', linewidth=2)
        ax1.plot(data['epoch_val_acc'], label='Val', linewidth=2)
        ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(data['epoch_loss'], label='Train', linewidth=2)
        ax2.plot(data['epoch_val_loss'], label='Val', linewidth=2)
        ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        show(fig)

        # -- Predictions --
        st.subheader("Predictions")
        pred_images = data['pred_images']
        pred_labels = data['pred_labels']
        pred_outputs = data['pred_outputs']
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        fig.suptitle(f"Predictions (Test Accuracy: {test_acc:.2%})", fontsize=14, fontweight='bold')
        for i in range(10):
            ax = axes[i // 5, i % 5]
            ax.imshow(pred_images[i])
            pred_class = np.argmax(pred_outputs[i])
            true_class = pred_labels[i]
            confidence = pred_outputs[i][pred_class] * 100
            color = 'green' if pred_class == true_class else 'red'
            ax.set_title(f"Pred: {class_names[pred_class]}\n({confidence:.1f}%)\nTrue: {class_names[true_class]}",
                         fontsize=8, color=color, fontweight='bold')
            ax.axis('off')
        plt.tight_layout()
        show(fig)

        # -- Feature maps --
        st.subheader("Feature Maps - First Conv Layer")
        feature_maps = data['feature_maps']
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle("Feature Maps - First Conv Layer (32 filters)", fontsize=14, fontweight='bold')
        for i in range(32):
            ax = axes[i // 8, i % 8]
            ax.imshow(feature_maps[0, i, :, :], cmap='viridis')
            ax.set_title(f"F{i+1}", fontsize=7)
            ax.axis('off')
        plt.tight_layout()
        show(fig)

        # -- Learned filters --
        st.subheader("Learned Filters")
        filters_display = data['filters_display']
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle("Learned Filters (First Conv Layer)", fontsize=14, fontweight='bold')
        for i in range(32):
            ax = axes[i // 8, i % 8]
            ax.imshow(filters_display[i])
            ax.set_title(f"F{i+1}", fontsize=7)
            ax.axis('off')
        plt.tight_layout()
        show(fig)
