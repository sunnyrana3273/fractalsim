# Hand Tracker - Web Application

A modern web application built with **Vite + TypeScript** frontend and **Flask** backend that tracks hand movements and detects when you open and close your hand using your camera. Real-time visual feedback and notifications are displayed in a beautiful web interface.

## Features

- 🎥 **Live Camera Feed**: Stream your camera feed directly in the browser
- 👋 **Hand Gesture Detection**: Automatically detects when you open or close your hand
- 🔔 **Real-time Notifications**: Get instant visual feedback for each gesture
- 📊 **Statistics Tracking**: See total gesture count and current hand status
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 📐 **Square Manipulation**: Create and resize squares by pulling corners with your finger
- 🔒 **Lock Mechanism**: Close your fist to lock squares in place

## Tech Stack

- **Frontend**: Vite + TypeScript
- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV + MediaPipe

## Setup

### Prerequisites

- Python 3.12 (MediaPipe requires Python ≤ 3.12)
- Node.js 18+ and npm

### Backend Setup

1. Create and activate virtual environment:

```bash
./setup.sh
source venv/bin/activate
```

Or manually:

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

1. Install Node dependencies:

```bash
npm install
```

## Running the Application

The application requires two servers to run:
1. **Flask backend** (API + video streaming)
2. **Vite dev server** (frontend)

### Option 1: Run Both Servers Manually

**Terminal 1 - Flask Backend:**
```bash
source venv/bin/activate
python app.py
```

The Flask server will run on `http://localhost:5001`

**Terminal 2 - Vite Frontend:**
```bash
npm run dev
```

The Vite dev server will run on `http://localhost:5173`

Open your browser to: `http://localhost:5173`

### Option 2: Use npm scripts (Recommended)

You can run both servers using npm scripts:

```bash
# In one terminal, start both servers
npm run dev:all
```

This will start both the Flask backend and Vite frontend concurrently.

## Project Structure

```
fractalsim/
├── app.py                 # Flask API server with hand tracking
├── src/                   # TypeScript frontend source
│   ├── components/        # React-style components
│   ├── types/            # TypeScript type definitions
│   ├── utils/            # Utility functions (API calls)
│   ├── main.ts           # Entry point
│   └── styles.css        # Global styles
├── index.html            # Vite entry HTML
├── package.json          # Node.js dependencies and scripts
├── vite.config.ts        # Vite configuration
├── tsconfig.json         # TypeScript configuration
├── requirements.txt      # Python dependencies
└── setup.sh              # Setup script for Python environment
```

## Development

### Backend Development

- Flask runs in debug mode by default
- Backend API endpoints:
  - `GET /api/notifications` - Get recent notifications
  - `GET /api/stats` - Get current statistics
  - `POST /api/clear_squares` - Clear all squares
  - `GET /video_feed` - Video stream (MJPEG)

### Frontend Development

- Vite provides hot module replacement (HMR)
- TypeScript for type safety
- Modern ES modules

### Building for Production

```bash
# Build frontend
npm run build

# The built files will be in the `dist/` directory
```

## Usage

1. Allow camera access when prompted by your browser
2. Show your hand to the camera
3. Open your hand to create a square
4. Move your index finger near any corner to resize the square
5. Pull your finger away from the center to make it bigger, toward the center to make it smaller
6. Close your fist to lock the square in place
7. Use the "Clear Squares" button to remove all squares

## API Endpoints

### GET `/api/notifications`
Returns recent gesture notifications.

**Response:**
```json
{
  "notifications": [
    {
      "id": 1,
      "type": "OPEN",
      "timestamp": 1234567890,
      "message": "Hand open!"
    }
  ],
  "total_count": 1
}
```

### GET `/api/stats`
Returns current statistics.

**Response:**
```json
{
  "total_gestures": 5,
  "hand_detected": true
}
```

### POST `/api/clear_squares`
Clears all squares from the video feed.

**Response:**
```json
{
  "success": true,
  "message": "Squares cleared"
}
```

### GET `/video_feed`
Returns an MJPEG video stream of the camera feed with hand tracking overlay.

## Notes

- Port 5001 is used for Flask (instead of 5000) because macOS uses port 5000 for AirPlay Receiver
- The application uses MediaPipe for hand tracking, which requires Python ≤ 3.12
- Hand gestures are detected when 4-5 fingers are extended (open) or 0-2 fingers (closed/fist)
- Squares are created only once when you first open your hand
- You can manipulate squares by grabbing corners with your index finger

## Troubleshooting

### Port Already in Use
If port 5001 is in use, change it in `app.py`:
```python
port = 5001  # Change to your preferred port
```

### Camera Not Working
- Make sure you've granted camera permissions in your browser
- Check that no other application is using the camera
- On macOS, you may need to grant camera access to your terminal application

### MediaPipe Installation Issues
If you have Python 3.14+, you'll need to use Python 3.12:
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
