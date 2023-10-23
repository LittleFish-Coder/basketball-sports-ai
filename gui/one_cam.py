from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import sys, cv2, threading
from ultralytics import YOLO
import os

app = QtWidgets.QApplication(sys.argv)
window_w, window_h = 300, 200    # 定義預設長寬尺寸

Form = QtWidgets.QWidget()
Form.setWindowTitle('SportsAI')
Form.resize(window_w, window_h)  # 使用變數

def windowResize(self):
    global window_w, window_h    # 定義使用全域變數
    window_w = Form.width()      # 讀取視窗寬度
    window_h = Form.height()     # 讀取視窗高度
    label.setGeometry(0,0,window_w,window_h)  # 設定 QLabel 長寬

Form.resizeEvent = windowResize  # 定義視窗尺寸改變時的要執行的函式

label = QtWidgets.QLabel(Form)
label.setGeometry(0,0,window_w,window_h)  # 使用變數

# Load model
pt = os.path.join(os.getcwd(), 'model_pt/yolov8n.pt')
model = YOLO(pt)

def opencv():
    global window_w, window_h    # 定義使用全域變數
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        frame = cv2.resize(frame, (window_w, window_h))  # 使用變數
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference on the frame
        results = model(frame, show_conf=False, show_labels=True)
        annotated_frame = results[0].plot()

        annotated_frame = annotated_frame
        height, width, channel = annotated_frame.shape
        # height, width, channel = frame.shape
        bytesPerline = channel * width
        img = QImage(annotated_frame, width, height, bytesPerline, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))

video = threading.Thread(target=opencv)
video.start()

Form.show()
sys.exit(app.exec_())