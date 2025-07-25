import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Allowed objects
object_labels = [
    'cell phone', 'microphone', 'remote', 'laptop',
    'keyboard', 'mouse', 'camera', 'glasses', 'tv', 'bottle'
]

# Allowed animals
animal_labels = ['cat', 'dog', 'bird', 'horse', 'cow', 'sheep']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally

    # Run YOLO detection
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf < 0.5:
            continue

        if label in object_labels:
            category = "OBJECT"
            color = (0, 255, 255)
        elif label in animal_labels:
            category = "ANIMAL"
            color = (255, 0, 0)
        else:
            continue

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw center dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Draw label with confidence
        label_text = f'{category} ({label} {conf:.2f})'
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Only Objects + Animals", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
