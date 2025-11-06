from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import math
from collections import deque

app = Flask(__name__)
CORS(app)

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # State tracking
        self.prev_hand_open = False
        self.hand_open_count = 0
        self.notifications = deque(maxlen=10)
        self.lock = threading.Lock()
        
        # Shapes to draw (store position, width, height, shape type, and timestamp)
        self.shapes = []  # List of (x, y, width, height, shape_type, timestamp, alpha)
        self.shape_mode = "square"  # "square" or "circle"
        
        # Resize interaction state
        self.active_resize = None  # {"shape_idx": int, "handle": str, "finger_pos": (x, y), "initial_width": int, "initial_height": int}
        self.handle_proximity_threshold = 80  # pixels
        
        # Drag/move interaction state
        self.active_drag = None  # {"shape_idx": int, "finger_pos": (x, y), "initial_center": (x, y)}
        self.drag_proximity_threshold = 100  # pixels - distance from center to start dragging
        
    def count_extended_fingers(self, landmarks, hand_label="Right"):
        """Count how many fingers are extended based on landmark positions."""
        finger_tips = [4, 8, 12, 16, 20]
        finger_mcp = [2, 5, 9, 13, 17]
        
        extended_count = 0
        
        # Check thumb (depends on hand orientation)
        thumb_extended = False
        if hand_label == "Right":
            thumb_extended = landmarks[finger_tips[0]].x > landmarks[finger_mcp[0]].x
        else:
            thumb_extended = landmarks[finger_tips[0]].x < landmarks[finger_mcp[0]].x
        
        if thumb_extended:
            extended_count += 1
        
        # Check other four fingers
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_mcp[i]].y:
                extended_count += 1
        
        return extended_count
    
    def is_hand_open(self, landmarks, hand_label="Right"):
        """Determine if hand is open (4-5 fingers extended) or closed."""
        extended_fingers = self.count_extended_fingers(landmarks, hand_label)
        return extended_fingers >= 4
    
    def is_fist_closed(self, landmarks, hand_label="Right"):
        """Determine if fist is completely closed (0-1 fingers extended)."""
        extended_fingers = self.count_extended_fingers(landmarks, hand_label)
        return extended_fingers <= 1
    
    def get_hand_bounding_box(self, landmarks, frame_width, frame_height):
        """Calculate bounding box of hand from landmarks."""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(min_x * frame_width)
        y1 = int(min_y * frame_height)
        x2 = int(max_x * frame_width)
        y2 = int(max_y * frame_height)
        
        # Calculate center and size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        size = max(width, height)  # Use larger dimension for square
        
        return center_x, center_y, size
    
    def get_finger_tip_position(self, landmarks, finger_idx, frame_width, frame_height):
        """Get pixel position of a finger tip.
        finger_idx: 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
        """
        if finger_idx >= len(landmarks):
            return None
        
        lm = landmarks[finger_idx]
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        return (x, y)
    
    def get_shape_handles(self, shape):
        """Get all handles for a shape (square or circle)."""
        width = shape["width"]
        height = shape["height"]
        x = shape["x"]
        y = shape["y"]
        shape_type = shape.get("shape_type", "square")
        
        if shape_type == "circle":
            # For circles, use radius and create 8 handles around the perimeter
            radius = max(width, height) // 2
            handles = {}
            for i in range(8):
                angle = (i * 2 * math.pi) / 8
                handle_x = int(x + radius * math.cos(angle))
                handle_y = int(y + radius * math.sin(angle))
                handles[f"handle_{i}"] = (handle_x, handle_y)
            return handles
        else:
            # Rectangle/square handles
            half_width = width // 2
            half_height = height // 2
            handles = {
                # Corners
                "top_left": (x - half_width, y - half_height),
                "top_right": (x + half_width, y - half_height),
                "bottom_left": (x - half_width, y + half_height),
                "bottom_right": (x + half_width, y + half_height),
                # Edges (middle points)
                "top": (x, y - half_height),
                "bottom": (x, y + half_height),
                "left": (x - half_width, y),
                "right": (x + half_width, y)
            }
            return handles
    
    def get_shape_bounds(self, shape):
        """Get the bounding box for drawing the shape."""
        width = shape["width"]
        height = shape["height"]
        x = shape["x"]
        y = shape["y"]
        shape_type = shape.get("shape_type", "square")
        
        if shape_type == "circle":
            radius = max(width, height) // 2
            return (x, y, radius)
        else:
            half_width = width // 2
            half_height = height // 2
            top_left = (x - half_width, y - half_height)
            bottom_right = (x + half_width, y + half_height)
            return (top_left, bottom_right, None)
    
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    def find_nearest_handle(self, finger_pos, shapes, threshold):
        """Find the nearest handle (corner or edge) to the finger position."""
        nearest = None
        min_distance = threshold
        
        for idx, shape in enumerate(shapes):
            handles = self.get_shape_handles(shape)
            for handle_name, handle_pos in handles.items():
                dist = self.distance(finger_pos, handle_pos)
                if dist < min_distance:
                    min_distance = dist
                    nearest = {
                        "shape_idx": idx,
                        "handle": handle_name,
                        "handle_pos": handle_pos,
                        "distance": dist
                    }
        
        return nearest
    
    def is_point_inside_shape(self, point, shape):
        """Check if a point is inside a shape."""
        x, y = point
        shape_x = shape["x"]
        shape_y = shape["y"]
        shape_type = shape.get("shape_type", "square")
        
        if shape_type == "circle":
            radius = max(shape["width"], shape["height"]) // 2
            dist_from_center = self.distance(point, (shape_x, shape_y))
            return dist_from_center <= radius
        else:
            # Rectangle/square
            half_width = shape["width"] // 2
            half_height = shape["height"] // 2
            return (shape_x - half_width <= x <= shape_x + half_width and
                    shape_y - half_height <= y <= shape_y + half_height)
    
    def find_shape_to_drag(self, finger_pos, shapes, handle_threshold):
        """Find a shape that the finger is over (not on a handle)."""
        # First check if finger is near any handle (don't drag if near handle)
        nearest_handle = self.find_nearest_handle(finger_pos, shapes, handle_threshold)
        if nearest_handle:
            return None  # Finger is near a handle, don't drag
        
        # Check if finger is inside any shape (can drag from anywhere inside)
        for idx, shape in enumerate(shapes):
            if self.is_point_inside_shape(finger_pos, shape):
                return {
                    "shape_idx": idx,
                    "distance": 0  # Always allow drag from inside shape
                }
        
        return None
    
    def process_frame(self, frame):
        """Process a single frame and detect hand gestures."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_detected = False
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Detect hand state
                current_hand_open = self.is_hand_open(hand_landmarks.landmark, hand_label)
                fist_closed = self.is_fist_closed(hand_landmarks.landmark, hand_label)
                hand_detected = True
                
                # Get hand position for drawing square
                frame_height, frame_width = frame.shape[:2]
                hand_x, hand_y, hand_size = self.get_hand_bounding_box(
                    hand_landmarks.landmark, frame_width, frame_height
                )
                
                # Get index finger tip for resize interaction
                index_finger_tip = self.get_finger_tip_position(
                    hand_landmarks.landmark, 8, frame_width, frame_height
                )
                
                # Handle resize and drag interactions
                with self.lock:
                    # Exit resize mode if fist is closed while resizing
                    if self.active_resize is not None and fist_closed:
                        # Lock square in place by exiting resize mode
                        self.active_resize = None
                        print("Fist closed - resize locked in place")
                    
                    # Resize interaction - works when hand is open
                    if not fist_closed and current_hand_open and index_finger_tip:
                        # Check for resize handles first
                        if self.active_resize is None:
                            # Look for nearby handle (corner or edge) to start resize
                            nearest = self.find_nearest_handle(
                                index_finger_tip, self.shapes, self.handle_proximity_threshold
                            )
                            if nearest:
                                # Start resize
                                shape = self.shapes[nearest["shape_idx"]]
                                handles = self.get_shape_handles(shape)
                                self.active_resize = {
                                    "shape_idx": nearest["shape_idx"],
                                    "handle": nearest["handle"],
                                    "finger_pos": index_finger_tip,
                                    "initial_finger_pos": index_finger_tip,
                                    "initial_width": shape["width"],
                                    "initial_height": shape["height"],
                                    "initial_center": (shape["x"], shape["y"]),
                                    "handle_pos": handles[nearest["handle"]]
                                }
                        else:
                            # Continue resize if finger is moving
                            shape_idx = self.active_resize["shape_idx"]
                            handle_name = self.active_resize["handle"]
                            shape = self.shapes[shape_idx]
                            shape_center = (shape["x"], shape["y"])
                            shape_type = shape.get("shape_type", "square")
                            
                            # Get initial dimensions
                            initial_width = self.active_resize["initial_width"]
                            initial_height = self.active_resize["initial_height"]
                            
                            if shape_type == "circle":
                                # For circles, resize based on distance from center
                                finger_to_center_dist = self.distance(index_finger_tip, shape_center)
                                target_radius = int(finger_to_center_dist)
                                current_radius = max(shape["width"], shape["height"]) // 2
                                new_radius = int(current_radius * 0.80 + target_radius * 0.20)
                                new_radius = max(new_radius, 25)
                                new_radius = min(new_radius, min(frame_width, frame_height) // 2 - 10)
                                
                                # Update both width and height to maintain circle
                                shape["width"] = new_radius * 2
                                shape["height"] = new_radius * 2
                            else:
                                # Rectangle/square resize logic
                                if handle_name in ["top", "bottom"]:
                                    # Vertical edge - change height only
                                    finger_y_dist = abs(index_finger_tip[1] - shape_center[1])
                                    target_height = int(finger_y_dist * 2)
                                    current_height = shape["height"]
                                    new_height = int(current_height * 0.80 + target_height * 0.20)
                                    new_height = max(new_height, 50)
                                    new_height = min(new_height, frame_height - 20)
                                    shape["height"] = new_height
                                    
                                elif handle_name in ["left", "right"]:
                                    # Horizontal edge - change width only
                                    finger_x_dist = abs(index_finger_tip[0] - shape_center[0])
                                    target_width = int(finger_x_dist * 2)
                                    current_width = shape["width"]
                                    new_width = int(current_width * 0.80 + target_width * 0.20)
                                    new_width = max(new_width, 50)
                                    new_width = min(new_width, frame_width - 20)
                                    shape["width"] = new_width
                                    
                                else:
                                    # Corner handle - change both width and height proportionally
                                    finger_to_center_dist = self.distance(index_finger_tip, shape_center)
                                    
                                    # Distance from center determines size
                                    target_size = int(finger_to_center_dist * 1.414)  # sqrt(2) ≈ 1.414
                                    
                                    # Maintain aspect ratio or scale both dimensions
                                    current_size_avg = (shape["width"] + shape["height"]) // 2
                                    new_size_avg = int(current_size_avg * 0.80 + target_size * 0.20)
                                    
                                    # Scale both dimensions proportionally
                                    scale_factor = new_size_avg / max(current_size_avg, 1)
                                    new_width = int(shape["width"] * scale_factor)
                                    new_height = int(shape["height"] * scale_factor)
                                    
                                    # Constraints
                                    new_width = max(new_width, 50)
                                    new_width = min(new_width, frame_width - 20)
                                    new_height = max(new_height, 50)
                                    new_height = min(new_height, frame_height - 20)
                                    
                                    shape["width"] = new_width
                                    shape["height"] = new_height
                            
                            self.active_resize["finger_pos"] = index_finger_tip
                    
                    # Drag interaction - works when hand is closed (fist)
                    if fist_closed and len(self.shapes) > 0:
                        # Use hand center position for dragging (wrist landmark)
                        wrist_pos = self.get_finger_tip_position(
                            hand_landmarks.landmark, 0, frame_width, frame_height  # Wrist is landmark 0
                        )
                        
                        if wrist_pos:
                            # Check if we can drag the shape
                            if self.active_drag is None:
                                # Check if wrist is inside any shape
                                drag_target = self.find_shape_to_drag(
                                    wrist_pos, self.shapes, self.handle_proximity_threshold
                                )
                                if drag_target:
                                    # Start drag
                                    shape = self.shapes[drag_target["shape_idx"]]
                                    self.active_drag = {
                                        "shape_idx": drag_target["shape_idx"],
                                        "hand_pos": wrist_pos,
                                        "initial_hand_pos": wrist_pos,
                                        "initial_center": (shape["x"], shape["y"])
                                    }
                            else:
                                # Continue dragging
                                shape_idx = self.active_drag["shape_idx"]
                                shape = self.shapes[shape_idx]
                                
                                # Get previous hand position for smooth tracking
                                prev_hand_pos = self.active_drag["hand_pos"]
                                
                                # Calculate movement delta from previous position
                                dx = wrist_pos[0] - prev_hand_pos[0]
                                dy = wrist_pos[1] - prev_hand_pos[1]
                                
                                # Update shape center position smoothly
                                new_x = shape["x"] + dx
                                new_y = shape["y"] + dy
                                
                                # Keep shape within frame bounds
                                if shape.get("shape_type", "square") == "circle":
                                    radius = max(shape["width"], shape["height"]) // 2
                                    new_x = max(radius, min(new_x, frame_width - radius))
                                    new_y = max(radius, min(new_y, frame_height - radius))
                                else:
                                    half_width = shape["width"] // 2
                                    half_height = shape["height"] // 2
                                    new_x = max(half_width, min(new_x, frame_width - half_width))
                                    new_y = max(half_height, min(new_y, frame_height - half_height))
                                
                                shape["x"] = new_x
                                shape["y"] = new_y
                                
                                # Update hand position for next frame
                                self.active_drag["hand_pos"] = wrist_pos
                    
                    # Release resize if finger not detected
                    if self.active_resize is not None and not index_finger_tip:
                        self.active_resize = None
                    
                    # Release if finger moved away from handle (for resize)
                    if self.active_resize is not None and index_finger_tip and not fist_closed:
                        shape_idx = self.active_resize["shape_idx"]
                        shape = self.shapes[shape_idx]
                        handles = self.get_shape_handles(shape)
                        handle_pos = handles[self.active_resize["handle"]]
                        dist = self.distance(index_finger_tip, handle_pos)
                        if dist > self.handle_proximity_threshold * 1.5:
                            self.active_resize = None
                    
                    # Release drag if hand is no longer closed or moved away from shape
                    if self.active_drag is not None:
                        if not fist_closed:
                            # Release if hand opens
                            self.active_drag = None
                        else:
                            # Check if wrist moved outside shape (with tolerance)
                            wrist_pos = self.get_finger_tip_position(
                                hand_landmarks.landmark, 0, frame_width, frame_height
                            )
                            if wrist_pos:
                                shape_idx = self.active_drag["shape_idx"]
                                shape = self.shapes[shape_idx]
                                
                                # Add padding to allow some movement outside before releasing
                                padding = 50  # pixels
                                if shape.get("shape_type", "square") == "circle":
                                    radius = max(shape["width"], shape["height"]) // 2
                                    expanded_shape = {
                                        "x": shape["x"],
                                        "y": shape["y"],
                                        "width": (radius + padding) * 2,
                                        "height": (radius + padding) * 2,
                                        "shape_type": "circle"
                                    }
                                else:
                                    expanded_shape = {
                                        "x": shape["x"],
                                        "y": shape["y"],
                                        "width": shape["width"] + padding * 2,
                                        "height": shape["height"] + padding * 2
                                    }
                                
                                if not self.is_point_inside_shape(wrist_pos, expanded_shape):
                                    self.active_drag = None
                            else:
                                # No wrist detected, release drag
                                self.active_drag = None
                
                # Check for state change
                with self.lock:
                    if current_hand_open != self.prev_hand_open:
                        self.hand_open_count += 1
                        timestamp = time.time()
                        gesture_type = "OPEN" if current_hand_open else "CLOSED"
                        notification = {
                            "id": self.hand_open_count,
                            "type": gesture_type,
                            "timestamp": timestamp,
                            "message": f"Hand {gesture_type.lower()}!"
                        }
                        self.notifications.append(notification)
                        
                        # Add shape when hand opens (only if no shape exists)
                        if current_hand_open and len(self.shapes) == 0:
                            # Size should be proportional to hand size, with some padding
                            shape_size = int(hand_size * 1.5)
                            self.shapes.append({
                                "x": hand_x,
                                "y": hand_y,
                                "width": shape_size,
                                "height": shape_size,
                                "shape_type": self.shape_mode,
                                "timestamp": timestamp,
                                "alpha": 255  # Start fully opaque
                            })
                            shape_name = "Circle" if self.shape_mode == "circle" else "Square"
                            print(f"Hand gesture detected! {shape_name} drawn at ({hand_x}, {hand_y})")
                        elif not current_hand_open:
                            print(f"Hand gesture detected! Count: {self.hand_open_count}")
                    
                    self.prev_hand_open = current_hand_open
                    
                    # Show current state on frame
                    state_text = "OPEN" if current_hand_open else "CLOSED"
                    cv2.putText(frame, f"Hand: {state_text}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show finger count
                    finger_count = self.count_extended_fingers(hand_landmarks.landmark, hand_label)
                    cv2.putText(frame, f"Fingers: {finger_count}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not hand_detected:
            with self.lock:
                self.prev_hand_open = False
        
        # Draw shapes
        with self.lock:
            frame_height, frame_width = frame.shape[:2]
            
            # Draw all shapes permanently (no expiration)
            for idx, shape in enumerate(self.shapes):
                shape_type = shape.get("shape_type", "square")
                color = (200, 230, 255)  # Light pastel blue (BGR format)
                thickness = 8
                
                if shape_type == "circle":
                    # Draw circle
                    center = (shape["x"], shape["y"])
                    radius = max(shape["width"], shape["height"]) // 2
                    cv2.circle(frame, center, radius, color, thickness)
                else:
                    # Draw rectangle/square
                    bounds = self.get_shape_bounds(shape)
                    top_left, bottom_right, _ = bounds
                    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                
                # Draw interactive handles with visual feedback
                handles = self.get_shape_handles(shape)
                is_active_shape = self.active_resize and self.active_resize["shape_idx"] == idx
                
                for handle_name, handle_pos in handles.items():
                    # Check if this handle is being manipulated
                    is_active_handle = (is_active_shape and 
                                       self.active_resize["handle"] == handle_name)
                    
                    # Determine handle size (corners are larger for rectangles)
                    if shape_type == "circle":
                        handle_size = 12 if is_active_handle else 8
                    else:
                        is_corner = handle_name in ["top_left", "top_right", "bottom_left", "bottom_right"]
                        handle_size = 12 if is_active_handle else (10 if is_corner else 8)
                    
                    handle_color = (255, 255, 0) if is_active_handle else (200, 230, 255)  # Yellow for active, light pastel blue for inactive
                    
                    # Draw filled circle for handle
                    cv2.circle(frame, handle_pos, handle_size, handle_color, -1)
                    # Draw outline
                    cv2.circle(frame, handle_pos, handle_size, (255, 255, 255), 3)
                    
                    # Draw pulsing effect for active handle
                    if is_active_handle:
                        pulse_size = int(handle_size * 1.8)
                        pulse_alpha = 100
                        overlay = frame.copy()
                        cv2.circle(overlay, handle_pos, pulse_size, handle_color, 4)
                        cv2.addWeighted(overlay, pulse_alpha / 255.0, frame, 
                                      1 - pulse_alpha / 255.0, 0, frame)
        
        return frame

# Global tracker instance
tracker = HandTracker()
camera = None
camera_lock = threading.Lock()

def get_camera():
    """Get or create camera instance."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError("Could not open camera")
        return camera

def generate_frames():
    """Generate video frames from camera."""
    cam = get_camera()
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame with hand tracking
        frame = tracker.process_frame(frame)
        
        # Display gesture count
        with tracker.lock:
            count = tracker.hand_open_count
        cv2.putText(frame, f"Gesture Count: {count}", 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/notifications')
def get_notifications():
    """Get recent notifications."""
    with tracker.lock:
        notifications = list(tracker.notifications)
        count = tracker.hand_open_count
    return jsonify({
        "notifications": notifications,
        "total_count": count
    })

@app.route('/api/stats')
def get_stats():
    """Get current stats."""
    with tracker.lock:
        return jsonify({
            "total_gestures": tracker.hand_open_count,
            "hand_detected": tracker.prev_hand_open is not None
        })

@app.route('/api/clear_squares', methods=['POST'])
def clear_squares():
    """Clear all shapes."""
    with tracker.lock:
        tracker.shapes.clear()
    return jsonify({"success": True, "message": "Shapes cleared"})

@app.route('/api/toggle_shape_mode', methods=['POST'])
def toggle_shape_mode():
    """Toggle between square and circle mode."""
    with tracker.lock:
        tracker.shape_mode = "circle" if tracker.shape_mode == "square" else "square"
    return jsonify({
        "success": True, 
        "shape_mode": tracker.shape_mode,
        "message": f"Shape mode set to {tracker.shape_mode}"
    })

@app.route('/api/get_shape_mode', methods=['GET'])
def get_shape_mode():
    """Get current shape mode."""
    with tracker.lock:
        return jsonify({"shape_mode": tracker.shape_mode})

@app.route('/api/clear_notifications', methods=['POST'])
def clear_notifications():
    """Clear all notifications."""
    with tracker.lock:
        tracker.notifications.clear()
    return jsonify({"success": True, "message": "Notifications cleared"})

if __name__ == '__main__':
    port = 5001
    print("Starting jorkin it Web Server...")
    print(f"Open your browser to: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)

