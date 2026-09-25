import cv2
import math
from ultralytics import YOLO

class FireDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        Initializes the YOLOv8 model for fire detection.
        Requires a fine-tuned model path (e.g., trained on D-Fire dataset).
        """
        # Load the Ultralytics YOLO model (CUDA is automatically used if available)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_fire(self, frame):
        """
        Processes a single frame to detect fire.
        Returns the bounding box, confidence, and calculated centroid.
        """
        # Run inference on the frame
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        # Iterate over detected bounding boxes
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Assuming class 0 is 'fire' in the custom trained model
            if cls_id == 0 and conf >= self.conf_threshold:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate the centroid of the fire bounding box
                u_c = (x1 + x2) / 2.0
                v_c = (y1 + y2) / 2.0
                
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "centroid": (u_c, v_c)
                })
                
        return detections

# ==========================================================
# Integration Example with MQTT Transmitter (from main.py)
# ==========================================================
if __name__ == "__main__":
    import time
    import paho.mqtt.client as mqtt

    # Setup MQTT Client based on main.py configuration
    MQTT_BROKER = "broker.hivemq.com"
    TOPIC_CMD = "my_stepper_esp32/cmd"
    
    mqtt_client = mqtt.Client(client_id="Python-Fire-Detector-Controller")
    mqtt_client.connect(MQTT_BROKER, 1883, 60)
    mqtt_client.loop_start()

    # Initialize the camera (Resolution target: 1280x720)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize Detector
    # Note: Replace 'yolov8n.pt' with your specific fire-trained weights (e.g., 'best.pt')
    detector = FireDetector(model_path="yolov8n.pt", conf_threshold=0.5)

    print("Starting real-time fire detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Run Detection
            fires = detector.detect_fire(frame)

            for fire in fires:
                cx, cy = fire["centroid"]
                conf = fire["confidence"]
                
                # Draw bounding box and centroid on frame for monitoring
                x1, y1, x2, y2 = map(int, fire["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                
                # 2. Homography placeholder: Map (cx, cy) to floor coordinates (Xw, Yw)
                # In the full system, this feeds into cv2.perspectiveTransform
                target_motor_x = int(cx * 1.5) # Mock conversion multiplier
                
                # 3. Transmit driver action to ESP32 (receiver.ino)
                # Using the absolute position command (G <pos>) as defined in the receiver
                command = f"G {target_motor_x}"
                mqtt_client.publish(TOPIC_CMD, command)
                print(f"[FIRE DETECTED] Conf: {conf:.2f} | Centroid: ({cx:.1f}, {cy:.1f}) | TX: {command}")
                
                # Throttle commands to prevent flooding the ESP32 while it moves
                time.sleep(1)

            # Display the monitoring interface
            cv2.imshow("Fire Monitoring - Camera 1", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Graceful shutdown
        mqtt_client.publish(TOPIC_CMD, "STOP") # Send halt command to ESP32
        mqtt_client.loop_stop()
        cap.release()
        cv2.destroyAllWindows()