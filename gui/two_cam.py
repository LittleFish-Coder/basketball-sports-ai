from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import sys, cv2, threading
from ultralytics import YOLO
from PyQt5 import QtCore
import os

app = QtWidgets.QApplication(sys.argv)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('SportsAI')
        self.resize(400, 400)  # 设置窗口的大小
        self.setGeometry(100, 100, 800, 400)  # 设置窗口的位置和大小

        layout = QtWidgets.QHBoxLayout(self)

        # Create a splitter for resizable sub-windows
        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter)

        # Left sub-window for displaying the webcam's result
        left_widget = QtWidgets.QWidget()
        layout.addWidget(left_widget, 1)  # Equal proportion for both sub-windows

        # Right sub-window for your GUI components
        right_widget = QtWidgets.QWidget()
        layout.addWidget(right_widget, 1)


        # Create layouts for sub-windows
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        # In the left sub-window, display webcam's result
        self.left_label = QtWidgets.QLabel()
        self.left_label.setFixedSize(400, 400)
        # self.left_label.setScaledContents(True)  # Scale the image to fit QLabel
        left_layout.addWidget(self.left_label)

        # In the right sub-window, add your GUI components
        self.right_label = QtWidgets.QLabel()
        self.right_label.setFixedSize(400, 400)
        # self.right_label.setScaledContents(True)  # Scale the image to fit QLabel
        right_layout.addWidget(self.right_label)

        # Load model
        pt = os.path.join(os.getcwd(), 'model_pt/yolov8n-pose.pt')
        self.model_pose = YOLO(pt)

        pt = os.path.join(os.getcwd(), 'model_pt/yolov8n.pt')
        self.model_shot = YOLO(pt)

        self.video_thread = threading.Thread(target=self.opencv)
        self.video_thread.start()

        self.shot_thread = threading.Thread(target=self.shot_detection, args=(self.right_label,))
        self.shot_thread.start()

    def opencv(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLOv8 inference on the frame
            results = self.model_pose(frame, show_conf=False, show_labels=True)
            pose_frame = results[0].plot()

            results = self.model_shot(frame, show_conf=False, show_labels=True)
            shot_frame = results[0].plot()

            pose_frame = cv2.resize(pose_frame, (self.left_label.width(), self.left_label.height()))
            shot_frame = cv2.resize(shot_frame, (self.right_label.width(), self.right_label.height()))

            height, width, channel = pose_frame.shape
            bytesPerline = channel * width
            
            pose_img = QImage(pose_frame, width, height, bytesPerline, QImage.Format_RGB888)
            shot_img = QImage(shot_frame, width, height, bytesPerline, QImage.Format_RGB888)

            self.left_label.setPixmap(QPixmap.fromImage(pose_img))
            # self.right_label.setPixmap(QPixmap.fromImage(shot_img))

    def shot_detection(self, sub_window):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLOv8 inference on the frame
            results = self.model_shot(frame, show_conf=False, show_labels=True)
            frame = results[0].plot()

            frame = cv2.resize(frame, (sub_window.width(), sub_window.height()))

            height, width, channel = frame.shape
            bytesPerline = channel * width
            
            img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)

            sub_window.setPixmap(QPixmap.fromImage(img))
    

if __name__ == '__main__':
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())