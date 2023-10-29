from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import sys, cv2, threading

window_w, window_h = 960, 540


class Widget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # self.setObjectName("SportsAI")
        self.setWindowTitle("SportsAI")
        self.resize(640, 360)  # initial size of the window

        # Create a horizontal layout for the split window
        split_layout = QtWidgets.QHBoxLayout(self)
        # Create a QSplitter to divide the split window
        splitter = QtWidgets.QSplitter()

        self.left_label = QtWidgets.QLabel(self)
        self.right_label = QtWidgets.QLabel(self)

        # Add labels to the QSplitter
        splitter.addWidget(self.left_label)
        splitter.addWidget(self.right_label)

        # Add the QSplitter to the split_layout
        split_layout.addWidget(splitter)

        # splitter.setSizes([window_w // 2, window_w // 2])

        # Set minimum sizes
        # self.left_label.setMinimumWidth(0)
        # self.right_label.setMinimumWidth(0)

        # detection thread
        webcam = threading.Thread(target=self.webcam)
        webcam.start()

    def webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            width, height = self.width(), self.height()
            height = width * 9 // 16  # set 16:9 ratio
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            frame = cv2.resize(frame, (width, height))  # 使用變數
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerline = channel * width
            img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
            self.left_label.setPixmap(QPixmap.fromImage(img))
            self.right_label.setPixmap(QPixmap.fromImage(img))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)  # Initialize the application
    # Create the main window
    Window = Widget()
    Window.show()
    # End

    sys.exit(app.exec_())  # Start the event loop
