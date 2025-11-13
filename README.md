# CADVISION

AI-powered CAD drawing comparison and analysis tool. Compare CAD drawings, detect changes, and highlight differences with automated component matching.

## Features

- **CAD Drawing Comparison**: Upload two CAD files (PDF or images) and automatically detect differences
- **Component Matching**: AI-powered matching of components between drawings using CLIP embeddings
- **Visual Highlighting**: Automatically highlight matched components and differences
- **Modern Web Interface**: Clean, Oracle-inspired UI for easy file upload and result visualization

## Project Structure

```
CAD-Detector/
├── backend/          # FastAPI backend server
├── frontend/         # React frontend application
├── functions/       # Core processing functions
│   ├── detector.py  # Object detection
│   ├── clip_compare.py  # CLIP-based component matching
│   ├── highlight.py     # Difference highlighting
│   └── pipeline.py      # Main processing pipeline
├── input/           # Input files directory
└── output_api/      # Generated output files
```

## Setup

### Backend

1. Install Python dependencies:
```bash
pip install fastapi uvicorn opencv-python pdf2image torch torchvision transformers
```

2. Run the backend server:
```bash
cd backend
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### Frontend

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in the terminal)

## Usage

1. Start both backend and frontend servers
2. Open the web interface in your browser
3. Upload two CAD files (PDF or image formats: PNG, JPG, JPEG)
4. Click "Run Comparison" to analyze the drawings
5. View the results showing:
   - Highlighted differences
   - Original input images
   - Matched components with similarity scores

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /compare` - Compare two CAD files
  - Accepts: `file1` and `file2` (multipart/form-data)
  - Returns: Comparison results with images and matches

## Technologies

- **Backend**: FastAPI, OpenCV, PyTorch, CLIP
- **Frontend**: React, Vite
- **AI/ML**: CLIP embeddings for semantic component matching

## License

MIT

