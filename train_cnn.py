"""
Train CNN locally and save all results for the Streamlit app.

Run once:  python train_cnn.py
Produces:  cnn_results.npz (training curves, predictions, feature maps, filters, samples)

The Streamlit app loads this file instead of training live.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Use GPU if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -- Config --
n_epochs = 15
batch_size = 64
n_train = 5000
n_test = max(n_train // 5, 200)  # 20% of training size, at least 200
img_size = 32                     # 32x32 RGB images
n_classes = 10

# 10 geometric pattern classes - each draws a unique shape on noisy background
class_names = ['H-Lines','V-Lines','Diagonal','Circle','Block',
               'Corners','Cross','Triangle','Border','Checker']

# -- Generate synthetic data --
np.random.seed(42)  # Fixed seed for reproducible dataset

def generate_class_images(n, class_id, img_size=32):
    """Generate n synthetic images for a given class.

    Heavy augmentation makes classification genuinely difficult:
    - High background noise (0.0-0.5) with strong Gaussian overlay
    - Pattern intensity barely above background (0.45-0.65)
    - Large position jitter (+/-6 pixels)
    - Random pixel dropout masks 30-50% of pattern pixels
    - All grayscale (no color shortcuts)
    - Variable sizes and thicknesses
    """
    # Bright noisy background - pattern must compete with this
    imgs = np.random.rand(n, img_size, img_size, 3).astype('float32') * 0.5
    for i in range(n):
        # Large positional jitter
        dr = np.random.randint(-6, 7)
        dc = np.random.randint(-6, 7)
        # Low intensity barely above background mean of ~0.25
        intensity = np.random.uniform(0.45, 0.65)
        # Pixel dropout rate - randomly erase 30-50% of pattern pixels
        drop_rate = np.random.uniform(0.30, 0.50)

        def safe(r1, r2, c1, c2):
            """Clamp slice bounds to valid image coordinates."""
            return max(0, r1), min(img_size, r2), max(0, c1), min(img_size, c2)

        def draw_with_dropout(img, r1, r2, c1, c2, val):
            """Draw a filled rectangle but randomly drop pixels."""
            r1, r2, c1, c2 = safe(r1, r2, c1, c2)
            if r2 <= r1 or c2 <= c1:
                return
            mask = np.random.rand(r2-r1, c2-c1) > drop_rate
            for ch in range(3):
                region = img[r1:r2, c1:c2, ch]
                region[mask] = val

        if class_id == 0:     # Horizontal lines (2-4 thin lines)
            n_lines = np.random.randint(2, 5)
            for ln in range(n_lines):
                row = 5 + ln * 6 + dr + np.random.randint(-2, 3)
                draw_with_dropout(imgs[i], row, row+1, 3+dc, 29+dc, intensity)

        elif class_id == 1:   # Vertical lines (2-4 thin lines)
            n_lines = np.random.randint(2, 5)
            for ln in range(n_lines):
                col = 5 + ln * 6 + dc + np.random.randint(-2, 3)
                draw_with_dropout(imgs[i], 3+dr, 29+dr, col, col+1, intensity)

        elif class_id == 2:   # Diagonal line (thin, 1px)
            for k in range(24):
                r, c = 4 + k + dr, 4 + k + dc
                if 0 <= r < 32 and 0 <= c < 32:
                    if np.random.rand() > drop_rate:
                        imgs[i, r, c, :] = intensity

        elif class_id == 3:   # Circle (thin outline, random radius)
            radius = np.random.randint(5, 12)
            cx, cy = 16 + dr, 16 + dc
            for angle in np.linspace(0, 2*np.pi, 36):
                r = int(cx + radius * np.sin(angle))
                c = int(cy + radius * np.cos(angle))
                if 0 <= r < 32 and 0 <= c < 32:
                    if np.random.rand() > drop_rate:
                        imgs[i, r, c, :] = intensity

        elif class_id == 4:   # Solid block (small, random size 6-12)
            sz = np.random.randint(6, 13)
            top = 16 - sz // 2 + dr
            left = 16 - sz // 2 + dc
            draw_with_dropout(imgs[i], top, top+sz, left, left+sz, intensity)

        elif class_id == 5:   # Corner dots (small, random size 3-6)
            sz = np.random.randint(3, 7)
            draw_with_dropout(imgs[i], 2+dr, 2+sz+dr, 2+dc, 2+sz+dc, intensity)
            draw_with_dropout(imgs[i], 30-sz+dr, 30+dr, 30-sz+dc, 30+dc, intensity)

        elif class_id == 6:   # Cross (thin arms, width 1-2)
            w = np.random.randint(1, 3)
            draw_with_dropout(imgs[i], 16-w//2+dr, 16+w//2+1+dr, 3+dc, 29+dc, intensity)
            draw_with_dropout(imgs[i], 3+dr, 29+dr, 16-w//2+dc, 16+w//2+1+dc, intensity)

        elif class_id == 7:   # Triangle (thin outline only, not filled)
            h = np.random.randint(8, 16)
            for row in range(h):
                spread = int(row * 0.9)
                r = 4 + row + dr
                cl = 16 - spread + dc
                cr = 16 + spread + dc
                if 0 <= r < 32:
                    # Left edge pixel
                    if 0 <= cl < 32 and np.random.rand() > drop_rate:
                        imgs[i, r, cl, :] = intensity
                    # Right edge pixel
                    if 0 <= cr < 32 and np.random.rand() > drop_rate:
                        imgs[i, r, cr, :] = intensity
                # Bottom edge
                if row == h - 1 and 0 <= r < 32:
                    for cc in range(max(0, cl), min(32, cr+1)):
                        if np.random.rand() > drop_rate:
                            imgs[i, r, cc, :] = intensity

        elif class_id == 8:   # Border frame (thin, 1px)
            m = 3 + np.random.randint(0, 3)
            draw_with_dropout(imgs[i], m+dr, m+1+dr, m+dc, 31-m+dc, intensity)
            draw_with_dropout(imgs[i], 31-m+dr, 31-m+1+dr, m+dc, 31-m+dc, intensity)
            draw_with_dropout(imgs[i], m+dr, 31-m+dr, m+dc, m+1+dc, intensity)
            draw_with_dropout(imgs[i], m+dr, 31-m+dr, 31-m+dc, 31-m+1+dc, intensity)

        elif class_id == 9:   # Checkerboard (small cells 3-5)
            cell = np.random.randint(3, 6)
            for r in range(0, 32, cell):
                for c in range(0, 32, cell):
                    if ((r // cell) + (c // cell)) % 2 == 0:
                        cr_pos = r + dr
                        cc_pos = c + dc
                        half = max(1, cell // 2)
                        draw_with_dropout(imgs[i], cr_pos, cr_pos+half,
                                          cc_pos, cc_pos+half, intensity)

        # Very heavy Gaussian noise (sigma=0.25) buries the signal
        imgs[i] += np.random.randn(img_size, img_size, 3).astype('float32') * 0.25
    return np.clip(imgs, 0, 1)  # Clamp to valid pixel range

print("Generating dataset...")
# Generate equal samples per class for balanced training
x_train_list, y_train_list = [], []
x_test_list, y_test_list = [], []
for c in range(n_classes):
    x_train_list.append(generate_class_images(n_train // n_classes, c))
    y_train_list.append(np.full(n_train // n_classes, c))
    x_test_list.append(generate_class_images(n_test // n_classes, c))
    y_test_list.append(np.full(n_test // n_classes, c))

# Concatenate per-class arrays into single train/test sets
x_train = np.concatenate(x_train_list)
y_train = np.concatenate(y_train_list)
x_test = np.concatenate(x_test_list)
y_test = np.concatenate(y_test_list)

# Shuffle so the model does not see all of one class in sequence
idx = np.random.permutation(len(x_train))
x_train, y_train = x_train[idx], y_train[idx]
idx = np.random.permutation(len(x_test))
x_test, y_test = x_test[idx], y_test[idx]

# Grab one example per class for the sample grid in the app
sample_images = np.zeros((10, img_size, img_size, 3), dtype='float32')
sample_labels = np.zeros(10, dtype='int64')
shown = set()
for i in range(len(x_train)):
    c = y_train[i]
    if c not in shown:
        sample_images[len(shown)] = x_train[i]
        sample_labels[len(shown)] = c
        shown.add(c)
        if len(shown) >= 10:
            break

# -- Build model --
class CNN(nn.Module):
    """Two-block CNN matching the architecture described in sections 1-9."""
    def __init__(self):
        super(CNN, self).__init__()
        # Block 1: 32 filters for low-level features - edges, textures
        self.conv1a = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)       # 32x32 -> 16x16
        self.drop1 = nn.Dropout(0.25)

        # Block 2: 64 filters for higher-level features - shapes, patterns
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)        # 16x16 -> 8x8
        self.drop2 = nn.Dropout(0.25)

        # Classification head
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Flatten 8x8x64 = 4096 -> 128
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.5)            # Aggressive dropout before output
        self.fc2 = nn.Linear(128, 10)           # 10-class scores

    def forward(self, x):
        # Block 1
        x = torch.relu(self.conv1a(x))
        x = self.bn1a(x)
        x = torch.relu(self.conv1b(x))
        x = self.drop1(self.pool1(x))
        # Block 2
        x = torch.relu(self.conv2a(x))
        x = self.bn2a(x)
        x = torch.relu(self.conv2b(x))
        x = self.drop2(self.pool2(x))
        # Classification head
        x = x.reshape(x.size(0), -1)  # Flatten - reshape handles non-contiguous
        x = torch.relu(self.fc1(x))
        x = self.bn_fc(x)
        x = self.drop3(x)
        x = self.fc2(x)               # Raw logits - softmax applied at inference
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # Combines log-softmax and NLL loss
optimizer = optim.Adam(model.parameters())

# -- Prepare DataLoaders --
# Convert numpy HWC arrays to PyTorch NCHW format - channels first
x_train_t = torch.tensor(x_train).permute(0, 3, 1, 2)
y_train_t = torch.tensor(y_train, dtype=torch.long)
x_test_t = torch.tensor(x_test).permute(0, 3, 1, 2)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Hold out 20% of training data for validation
val_size = int(0.2 * len(x_train_t))
x_val_t, y_val_t = x_train_t[:val_size], y_train_t[:val_size]
x_train_t, y_train_t = x_train_t[val_size:], y_train_t[val_size:]

train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size)

# -- Train --
epoch_acc, epoch_val_acc, epoch_loss, epoch_val_loss = [], [], [], []

for epoch in range(n_epochs):
    # Training pass - gradient updates enabled
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()          # Clear gradients from previous step
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()                # Compute gradients via backpropagation
        optimizer.step()               # Update weights
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation pass - no gradients needed
    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            vl += criterion(outputs, labels).item() * images.size(0)
            vc += (outputs.argmax(1) == labels).sum().item()
            vt += labels.size(0)

    val_loss, val_acc = vl / vt, vc / vt
    epoch_acc.append(train_acc)
    epoch_val_acc.append(val_acc)
    epoch_loss.append(train_loss)
    epoch_val_loss.append(val_loss)
    print(f"Epoch {epoch+1}/{n_epochs} - acc: {train_acc:.3f} val_acc: {val_acc:.3f} loss: {train_loss:.4f}")

# -- Evaluate on held-out test set --
model.eval()
tc, tt, tl = 0, 0, 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        tl += criterion(outputs, labels).item() * images.size(0)
        tc += (outputs.argmax(1) == labels).sum().item()
        tt += labels.size(0)

test_acc = tc / tt
test_loss = tl / tt
print(f"\nTest Accuracy: {test_acc:.2%} | Test Loss: {test_loss:.4f}")

# -- Predictions on 10 test images --
# Apply softmax to convert raw logits into probability distributions
with torch.no_grad():
    pred_inputs = torch.tensor(x_test[:10]).permute(0, 3, 1, 2).to(device)
    pred_outputs = torch.softmax(model(pred_inputs), dim=1).cpu().numpy()
pred_images = x_test[:10]
pred_labels = y_test[:10]

# -- Feature maps from first conv layer --
# Register a forward hook to capture intermediate activations
hook_output = [None]  # Use list since closures cannot rebind outer variables
def hook_fn(module, input, output):
    hook_output[0] = output.detach().cpu().numpy()
hook = model.conv1a.register_forward_hook(hook_fn)
with torch.no_grad():
    model(torch.tensor(x_test[0:1]).permute(0, 3, 1, 2).to(device))
hook.remove()
feature_maps = hook_output[0]  # Shape: (1, 32, 32, 32) - 32 filter activations

# -- Learned filters --
# PyTorch stores weights as (out_ch, in_ch, H, W) - transpose to HWC for display
raw_filters = model.conv1a.weight.detach().cpu().numpy()
filters_display = np.zeros((32, 3, 3, 3), dtype='float32')
for i in range(32):
    f = raw_filters[i].transpose(1, 2, 0)       # CHW -> HWC for RGB display
    f = (f - f.min()) / (f.max() - f.min())      # Min-max normalize to [0, 1]
    filters_display[i] = f

# -- Save all results to a single compressed file --
# The Streamlit app loads this instead of running training at deploy time
np.savez_compressed('cnn_results.npz',
    # Training history - one value per epoch
    epoch_acc=np.array(epoch_acc),
    epoch_val_acc=np.array(epoch_val_acc),
    epoch_loss=np.array(epoch_loss),
    epoch_val_loss=np.array(epoch_val_loss),
    # Final test set metrics
    test_acc=np.array(test_acc),
    test_loss=np.array(test_loss),
    # One sample image per class for the dataset grid
    sample_images=sample_images,
    sample_labels=sample_labels,
    # 10 test predictions with softmax probabilities
    pred_images=pred_images,
    pred_labels=pred_labels,
    pred_outputs=pred_outputs,
    # First conv layer outputs and weights for visualization
    feature_maps=feature_maps,
    filters_display=filters_display,
    # Class name strings for plot labels
    class_names=np.array(class_names),
)

print(f"\nSaved cnn_results.npz")
