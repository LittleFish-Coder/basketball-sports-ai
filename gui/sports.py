from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import sys, cv2, threading

app = QtWidgets.QApplication(sys.argv)
window_w, window_h = 300, 200    # 定義預設長寬尺寸

Form = QtWidgets.QWidget()
Form.setWindowTitle('oxxo.studio')
Form.resize(window_w, window_h)  # 使用變數

def windowResize(self):
    global window_w, window_h    # 定義使用全域變數
    window_w = Form.width()      # 讀取視窗寬度
    window_h = Form.height()     # 讀取視窗高度
    label.setGeometry(0,0,window_w,window_h)  # 設定 QLabel 長寬

Form.resizeEvent = windowResize  # 定義視窗尺寸改變時的要執行的函式

label = QtWidgets.QLabel(Form)
label.setGeometry(0,0,window_w,window_h)  # 使用變數

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
        height, width, channel = frame.shape
        bytesPerline = channel * width
        img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))

video = threading.Thread(target=opencv)
video.start()

Form.show()
sys.exit(app.exec_())

# class Widget(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#         # self.setObjectName("SportsAI")
#         self.setWindowTitle('SportsAI')
#         self.resize(800, 400)
#         self.ui()

#     def ui(self):
#         label = QtWidgets.QLabel(self)
#         label.move(50,50)
#         label.setText('hello world')
#         # label.setStyleSheet('font-size:30px; color:#00c')


# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)  # Initialize the application

#     # Create the main window
#     Window = Widget()
#     Window.show()
#     # End

#     sys.exit(app.exec_())   # Start the event loop