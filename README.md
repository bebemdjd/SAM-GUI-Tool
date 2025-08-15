# SAM GUI Tool - Interactive Image Segmentation Interface

A user-friendly graphical interface for the Segment Anything Model (SAM) that enables interactive image segmentation with real-time visualization and multiple operation modes.
<img width="1195" height="826" alt="image" src="https://github.com/user-attachments/assets/9cabf412-c522-43d1-a6c9-44e6021ec19e" />

![SAM GUI Interface](https://img.shields.io/badge/GUI-Tkinter-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![SAM](https://img.shields.io/badge/Model-SAM-orange)

## 🚀 Quick Start

### 1. Launch the Application
```bash
python test_gui.py
```

### 2. Load a SAM Model
- Click **"选择模型文件夹"** (Choose Model Folder)
- Navigate to your SAM model weights directory
- The tool will automatically detect and match model files:
  - `sam_vit_b_*.pth` → ViT-B model
  - `sam_vit_l_*.pth` → ViT-L model  
  - `sam_vit_h_*.pth` → ViT-H model
- Select your preferred model architecture and click **"加载模型"** (Load Model)

### 3. Load an Image
- Click **"加载图像"** (Load Image)
- Select your target image (supports JPG, PNG, BMP, TIFF)
- The image will appear in the main canvas

### 4. Start Segmenting
Choose from three operation modes and start segmenting!

## 🎯 Operation Modes

### 📍 Point Mode (点击模式)
**Best for**: Precise object segmentation with user guidance

**How to use**:
1. Select **"点击模式"** (Point Mode)
2. Choose point type:
   - **前景点** (Foreground): Click on the object you want to segment
   - **背景点** (Background): Click on areas you want to exclude
3. Click on the image to add points
4. Click **"生成mask"** (Generate Mask) to create segmentation

**Tips**:
- Red circles = Foreground points
- Blue circles = Background points  
- Use **"撤销"** (Undo) to remove the last point
- Multiple points improve accuracy

### 📦 Box Mode (框选模式)
**Best for**: Quick segmentation of rectangular regions

**How to use**:
1. Select **"框选模式"** (Box Mode)
2. Click and drag to draw a bounding box around your target object
3. Release mouse button - the mask will be generated automatically

**Tips**:
- Yellow rectangle shows your selection
- Drag from corner to corner for best results
- The mask will be generated immediately upon release

### 🤖 Auto Mode (自动分割)
**Best for**: Discovering all objects in an image

**How to use**:
1. Select **"自动分割"** (Auto Mode)
2. Click **"生成mask"** (Generate Mask)
3. Wait for processing (may take 10-30 seconds)
4. The largest detected object will be displayed

**Tips**:
- Processes the entire image automatically
- Shows the largest segmented region by default
- Great for exploration and object discovery

## 🎛️ Interface Overview

### Model Configuration Panel (模型配置)
- **Model Architecture**: Choose between ViT-B (fastest), ViT-L (balanced), ViT-H (best quality)
- **Model Files**: Automatic detection and loading of SAM weights
- **Status Indicator**: Real-time model loading status with color codes:
  - 🟠 Orange: Loading/Ready to load
  - 🟢 Green: Successfully loaded
  - 🔴 Red: Error occurred

### File Operations (文件操作)
- **加载图像** (Load Image): Import images for segmentation
- **保存mask** (Save Mask): Export binary mask as PNG
- **保存叠加图** (Save Overlay): Export image with colored mask overlay

### Action Controls (操作)
- **清除** (Clear): Remove all annotations and masks
- **生成mask** (Generate Mask): Create segmentation based on current inputs
- **撤销** (Undo): Remove the last point in Point Mode

### Real-time Log Panel (日志信息)
- **Live Logging**: View detailed operation progress
- **Auto-scroll**: Automatically scroll to latest messages
- **Log Controls**: 
  - **清空日志** (Clear Log): Remove all log entries
  - **保存日志** (Save Log): Export log to text file

## 🎨 Visual Feedback

### Mask Visualization
- **Green overlay**: Segmented regions (70% image + 30% green)
- **Real-time preview**: Instant visual feedback during interaction

### Point Annotations
- **Red circles**: Foreground points (include in mask)
- **Blue circles**: Background points (exclude from mask)
- **White borders**: Enhanced visibility on any background

### Bounding Box
- **Yellow rectangle**: Current selection area in Box Mode

## 💡 Tips for Best Results

### Point Mode Strategy
- Start with 1-2 foreground points on your target object
- Add background points near object boundaries for precision
- Use multiple foreground points for complex shapes
- Background points help separate similar-colored objects

### Image Quality Tips
- Use high-resolution images when possible
- Ensure good contrast between objects
- Avoid heavily compressed or blurry images

### Performance Optimization
- **ViT-B**: Choose for real-time interaction (fastest)
- **ViT-L**: Good balance of speed and quality
- **ViT-H**: Best quality for final results (slower)

## 🔧 Model Status Indicators

| Status | Color | Meaning |
|--------|-------|---------|
| 检查模块中... | 🟠 Orange | Checking SAM modules |
| 模块已加载 | 🟠 Orange | Modules loaded, ready for model |
| 已加载 | 🟢 Green | Model successfully loaded |
| 加载失败 | 🔴 Red | Model loading failed |

## 📁 Supported File Formats

### Input Images
- **JPEG** (.jpg, .jpeg)
- **PNG** (.png) 
- **BMP** (.bmp)
- **TIFF** (.tiff, .tif)

### Output Files
- **Masks**: PNG binary images (black/white)
- **Overlays**: PNG/JPEG with colored mask overlay
- **Logs**: UTF-8 text files

## ⚡ Keyboard Shortcuts

- **Ctrl+O**: Load image (when available)
- **Ctrl+S**: Save current mask (when available)
- **Escape**: Clear current annotations
- **Space**: Generate mask (when inputs are ready)

## 🐛 Troubleshooting

### Common Issues

**"请先加载模型" (Please load model first)**
- Ensure you've selected a model folder with valid .pth files
- Check that the model architecture matches your files

**"SAM模块不可用" (SAM modules unavailable)**  
- Install required dependencies: `pip install torch opencv-python pillow`
- Ensure SAM model files are in the correct directory

**Slow performance**
- Try ViT-B model for faster processing
- Reduce image resolution if very large
- Close other applications to free memory

**Auto segmentation fails**
- Check if your model supports automatic mask generation
- Try Point or Box mode instead
- Ensure sufficient system memory

### Getting Help

1. **Check the log panel** for detailed error messages
2. **Verify model files** are correctly named and not corrupted
3. **Try different operation modes** if one isn't working
4. **Restart the application** if interface becomes unresponsive

---

## 📋 Quick Reference

| 操作 | 步骤 |
|--------|-------|
| **基础分割** | 加载模型 → 加载图像 → 点击模式 → 点击对象 → 生成mask |
| **快速框选** | 加载模型 → 加载图像 → 框选模式 → 拖拽矩形框 |
| **自动发现** | 加载模型 → 加载图像 → 自动分割 → 生成mask |
| **保存结果** | 生成mask → 保存mask（二进制）或 保存叠加图（可视化） |
| **重新开始** | 点击清除 → 添加新的点/框 → 生成mask |

**💡 专业提示**: 日志信息面板会显示后台具体操作过程 - 如果遇到问题请检查日志信息！
