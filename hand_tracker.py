import cv2
import mediapipe as mp
import numpy as np

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
        
        # Visual feedback
        self.show_feedback = False
        self.feedback_timer = 0
        self.feedback_duration = 30  # frames to show feedback
        
    def count_extended_fingers(self, landmarks, hand_label="Right"):
        """
        Count how many fingers are extended based on landmark positions.
        Returns: number of extended fingers (0-5)
        """
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_mcp = [2, 5, 9, 13, 17]    # base joints
        
        extended_count = 0
        
        # Check thumb (special case - compare x coordinates, depends on hand)
        thumb_extended = False
        if hand_label == "Right":
            # Right hand: thumb extended if tip is to the right of mcp
            thumb_extended = landmarks[finger_tips[0]].x > landmarks[finger_mcp[0]].x
        else:
            # Left hand: thumb extended if tip is to the left of mcp
            thumb_extended = landmarks[finger_tips[0]].x < landmarks[finger_mcp[0]].x
        
        if thumb_extended:
            extended_count += 1
        
        # Check other four fingers (compare y coordinates)
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_mcp[i]].y:
                extended_count += 1
        
        return extended_count
    
    def is_hand_open(self, landmarks, hand_label="Right"):
        """Determine if hand is open (4-5 fingers extended) or closed (0-1 fingers)"""
        extended_fingers = self.count_extended_fingers(landmarks, hand_label)
        # Consider hand open if 4-5 fingers extended, closed if 0-2
        return extended_fingers >= 4
    
    def draw_feedback(self, frame):
        """Draw visual feedback on the frame"""
        if self.show_feedback:
            height, width = frame.shape[:2]
            
            # Draw a colorful circle that pulses
            center = (width // 2, height // 2)
            radius = int(50 + 20 * np.sin(self.feedback_timer * 0.2))
            color = (0, 255, 0) if self.prev_hand_open else (0, 0, 255)
            
            cv2.circle(frame, center, radius, color, -1)
            cv2.circle(frame, center, radius + 10, color, 3)
            
            # Add text
            text = "HAND OPEN!" if self.prev_hand_open else "HAND CLOSED!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2 + 100
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            self.feedback_timer += 1
            if self.feedback_timer >= self.feedback_duration:
                self.show_feedback = False
                self.feedback_timer = 0
    
    def process_frame(self, frame):
        """Process a single frame and detect hand gestures"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_detected = False
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand label (Left or Right)
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Detect hand state
                current_hand_open = self.is_hand_open(hand_landmarks.landmark, hand_label)
                hand_detected = True
                
                # Check for state change
                if current_hand_open != self.prev_hand_open:
                    self.show_feedback = True
                    self.feedback_timer = 0
                    self.hand_open_count += 1
                    print(f"Hand gesture detected! Count: {self.hand_open_count}")
                
                self.prev_hand_open = current_hand_open
                
                # Show current state on frame
                state_text = "OPEN" if current_hand_open else "CLOSED"
                cv2.putText(frame, f"Hand: {state_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show finger count
                finger_count = self.count_extended_fingers(hand_landmarks.landmark, hand_label)
                cv2.putText(frame, f"Fingers: {finger_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if not hand_detected:
            self.prev_hand_open = False
        
        # Draw feedback
        self.draw_feedback(frame)
        
        return frame

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Hand Tracker Started!")
    print("Open and close your hand to see the feedback")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        frame = tracker.process_frame(frame)
        
        # Display count
        cv2.putText(frame, f"Gesture Count: {tracker.hand_open_count}", 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Hand Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total gestures detected: {tracker.hand_open_count}")

if __name__ == "__main__":
    main()

