import cv2
from ultralytics import YOLO

# Tải mô hình YOLOv8
model = YOLO("yolov8n.pt")

# Mở luồng video từ FFmpeg
cap = cv2.VideoCapture('udp://127.0.0.1:12345')

while True:
    # Đọc khung hình từ luồng video
    ret, frame = cap.read()
    if not ret:
        break

    # Thực hiện nhận diện đối tượng
    results = model(frame)

    # Vẽ các kết quả nhận diện lên khung hình
    annotated_frame = results[0].plot()

    # Hiển thị khung hình
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
