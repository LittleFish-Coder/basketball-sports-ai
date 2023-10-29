from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import sys, cv2, threading
from ultralytics import YOLO
from PyQt5 import QtCore
import cvndi
import os


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.source = cvndi.get_sources()  # whether to use ndi camera

        self.setWindowTitle("SportsAI")
        self.resize(1920, 1080)
        self.setGeometry(0, 0, 1920, 1080)  # 设置窗口的位置和大小
        # screen_height = self.availableGeometry().height()

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
        self.left_label.setFixedSize(720, 480)
        # self.left_label.setFixedSize(desired_width, desired_height)
        # self.left_label.setScaledContents(True)  # Scale the image to fit QLabel
        left_layout.addWidget(self.left_label)

        # In the right sub-window, add your GUI components
        self.right_label = QtWidgets.QLabel()
        self.right_label.setFixedSize(720, 480)
        # self.right_label.setScaledContents(True)  # Scale the image to fit QLabel
        right_layout.addWidget(self.right_label)

        # Load model
        pt = os.path.join(os.getcwd(), "model_pt/yolov8n-pose.pt")
        self.model_pose = YOLO(pt)

        pt = os.path.join(os.getcwd(), "model_pt/yolov8n.pt")
        self.model_shot = YOLO(pt)

        self.pose_thread = threading.Thread(target=self.pose_estimation)
        self.pose_thread.start()

        self.shot_thread = threading.Thread(target=self.shot_detection)
        self.shot_thread.start()

    def pose_estimation(self):
        # cap = cvndi.VideoCapture(cvndi.ip_source(self.source, "102"))  # for ndi camera
        cap = cv2.VideoCapture(0)  # for webcam
        # if not cap.isOpened():
        #     print("Cannot open camera")
        #     return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLOv8 inference on the frame
            results = self.model_pose(frame)
            frame = results[0].plot(boxes=False)

            frame = cv2.resize(frame, (self.left_label.width(), self.left_label.height()))
            height, width, channel = frame.shape
            bytesPerline = channel * width

            img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
            self.left_label.setPixmap(QPixmap.fromImage(img))

    def shot_detection(self):
        # cap = cvndi.VideoCapture(cvndi.ip_source(self.source, "102"))  # for ndi camera
        cap = cv2.VideoCapture(0)  # for webcam
        # if not cap.isOpened():
        #     print("Cannot open camera")
        #     return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLOv8 inference on the frame
            results = self.model_shot(frame)
            frame = results[0].plot(conf=False)

            frame = cv2.resize(frame, (self.right_label.width(), self.right_label.height()))
            height, width, channel = frame.shape
            bytesPerline = channel * width

            img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
            self.right_label.setPixmap(QPixmap.fromImage(img))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
