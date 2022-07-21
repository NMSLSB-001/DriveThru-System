import imghdr
import sys

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtGui
from MainWindow import Ui_MainWindow

import ssd_live


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.pushButton.clicked.connect(self.controlTimer)

        # start/stop timer

    def dataChange(self):
        word = "Your Parent is arriving"
        cursor = self.textBrowser.textCursor()
        cursor.insertHtml('''<p><span style="color: red; font-weight: bold; font-size: 20px">{} </span>'''.format(word))
        self.textBrowser.setText(word)

    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            self.dataChange()
            # create video capture
            self.cap = cv2.VideoCapture(r"..//1.mp4")
            self.cap.set(3, 1280)
            self.cap.set(4, 720)
            # start timer
            # self.image = self.ssd_live.getData()
            self.timer.start(2)
            # update control_bt text
            self.pushButton.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.pushButton.setText("Start")

    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        # image = self.image
        image = cv2.resize(image, None, fx=0.65, fy=0.65)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.label_image.setPixmap(QPixmap.fromImage(qImg))


if __name__ == '__main__':
    ssd_live.start()
    app = QApplication(sys.argv)
    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
