import sys
import typing
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSplitter
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import os
import threading
import cvndi

window_width, window_height = 1280, 720  # define the default width and height


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # initialize parameters
        self.params()

        self.setWindowTitle("SportsAI")
        self.resize(self.window_width, self.window_height)
        self.setGeometry(0, 0, self.window_width, self.window_height)

        self.layout = QHBoxLayout(self)
        # set alignment
        self.layout.setAlignment(Qt.AlignCenter)
        # create a QLabel for displaying the webcam's result
        self.label = QLabel()
        # add the QLabel to the layout
        self.layout.addWidget(self.label)
        # set resize event
        self.resizeEvent = resizeEvent

        # start the pose estimation thread
        pose_estimation_thread = threading.Thread(target=self.pose_estimation)
        pose_estimation_thread.start()

    def params(self):
        global window_width, window_height
        self.window_width, self.window_height = window_width, window_height

    def load_model(self):
        # Load model
        pt = os.path.join(os.getcwd(), "model_pt/yolov8n-pose.pt")
        self.model = YOLO(pt)

    def pose_estimation(self):
        # initialize the model
        self.load_model()

        # get the window's width and height !!! important !!!
        global window_width, window_height
        # window_width, window_height = self.width(), self.height()

        cap = cv2.VideoCapture(0)
        # cap = cvndi.VideoCapture(cvndi.ip_source(self.source, "102"))  # for ndi camera
        # if not cap.isOpened():
        #     print("Cannot open camera")
        #     exit()
        while cap:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                continue
            frame = cv2.resize(frame, (window_width, window_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # inference the frame
            results = self.model(frame)
            annotated_frame = results[0].plot(boxes=False)

            height, width, channel = annotated_frame.shape
            bytesPerline = channel * width
            img = QImage(annotated_frame, width, height, bytesPerline, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(img))


# overwrite the resize event
def resizeEvent(self):
    global window_width, window_height
    # window_width, window_height = mainWindow.width(), mainWindow.height()
    # mainWindow.label.setGeometry(0, 0, window_width, window_height)


if __name__ == "__main__":
    app = QApplication(sys.argv)  # initialize the application
    mainWindow = MainWindow()  # create a new instance of the main window
    mainWindow.resizeEvent = resizeEvent
    mainWindow.show()  # make the main window visible
    sys.exit(app.exec_())  # start the event loop
