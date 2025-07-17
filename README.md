# 🚗 Smart Parking Detection System

A real-time parking space detection system powered by YOLO models, optimized for Hugging Face Spaces.

## Features

- **🧠 AI-Powered Detection**: Uses YOLOv8 models for accurate vehicle detection
- **⚙️ Configurable Spaces**: Easy JSON-based parking space configuration
- **📱 Real-time Processing**: Instant image analysis with visual feedback
- **📊 Analytics Dashboard**: Performance metrics and occupancy patterns
- **🤗 HF Spaces Optimized**: Memory-efficient deployment on Hugging Face Spaces

## How to Use

1. **Setup**: Upload a reference image of your parking lot
2. **Configure**: Define parking spaces using the JSON editor
3. **Detect**: Upload images to detect vehicle occupancy
4. **Analyze**: View real-time analytics and space utilization

## Model Options

- **YOLOv8n**: Fastest, best for real-time (6MB)
- **YOLOv8s**: Balanced speed/accuracy (22MB)
- **YOLOv8m**: Higher accuracy (50MB)

## Configuration Format

```json
{
  "spaces": [
    {
      "id": 1,
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "label_position": [x, y]
    }
  ]
}
```

## Tech Stack

- **Streamlit**: Web interface
- **Ultralytics YOLO**: Object detection
- **OpenCV**: Image processing
- **NumPy/Pandas**: Data handling

## Performance

- **Memory Usage**: Optimized for 16GB RAM
- **Processing Speed**: 1-3 seconds per image
- **Accuracy**: 85-95% vehicle detection

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Access at `http://localhost:8501`

Built with ❤️ for the AI community
