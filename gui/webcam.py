from PyQt5 import QtWidgets, QtCore
import sys

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.setWindowTitle('SportsAI')
        self.resize(300, 200)
        self.ui()

    def ui(self):
        pushButton = QtWidgets.QPushButton(self)
        pushButton.setGeometry(QtCore.QRect(100, 70, 113, 32))
        pushButton.setObjectName("pushButton")
        pushButton.setText("PushButton")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    SecondWindow = MyWidget()
    SecondWindow.show()
    sys.exit(app.exec_())