# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui2.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QMovie, QIcon
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import numpy as np

from detect_traffic import detectTraffic, detectLabel
import cv2


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(663, 479)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 641, 452))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.originalPic1 = QtWidgets.QLabel(self.tab_3)
        self.originalPic1.setFrameShape(QtWidgets.QFrame.Box)
        self.originalPic1.setText("")
        self.originalPic1.setObjectName("originalPic1")
        self.verticalLayout_5.addWidget(self.originalPic1)
        self.labelTraffic = QtWidgets.QLabel(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelTraffic.sizePolicy().hasHeightForWidth())
        self.labelTraffic.setSizePolicy(sizePolicy)
        self.labelTraffic.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelTraffic.setTextFormat(QtCore.Qt.RichText)
        self.labelTraffic.setAlignment(QtCore.Qt.AlignCenter)
        self.labelTraffic.setObjectName("labelTraffic")
        self.verticalLayout_5.addWidget(self.labelTraffic)
        self.selectImage1 = QtWidgets.QPushButton(self.tab_3)
        self.selectImage1.setObjectName("selectImage1")
        self.selectImage1.clicked.connect(self.selectImageForClassifier)
        self.verticalLayout_5.addWidget(self.selectImage1)
        self.verticalLayout_6.addLayout(self.verticalLayout_5)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.labelOriginal = QtWidgets.QLabel(self.tab)
        self.labelOriginal.setMinimumSize(QtCore.QSize(0, 374))
        self.labelOriginal.setFrameShape(QtWidgets.QFrame.Box)
        self.labelOriginal.setText("")
        self.labelOriginal.setObjectName("labelOriginal")
        self.horizontalLayout.addWidget(self.labelOriginal)
        self.labelResult = QtWidgets.QLabel(self.tab)
        self.labelResult.setFrameShape(QtWidgets.QFrame.Box)
        self.labelResult.setText("")
        self.labelResult.setObjectName("labelResult")
        self.horizontalLayout.addWidget(self.labelResult)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.selectImage2 = QtWidgets.QPushButton(self.tab)
        self.selectImage2.setObjectName("selectImage2")
        self.selectImage2.clicked.connect(self.selectImage)
        self.verticalLayout.addWidget(self.selectImage2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.tabWidget.addTab(self.tab, "")
        # self.tab_2 = QtWidgets.QWidget()
        # self.tab_2.setObjectName("tab_2")
        # self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_2)
        # self.verticalLayout_4.setObjectName("verticalLayout_4")
        # self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        # self.verticalLayout_3.setObjectName("verticalLayout_3")
        # self.view = QtWidgets.QLabel(self.tab_2)
        # self.view.setFrameShape(QtWidgets.QFrame.Box)
        # self.view.setText("")
        # self.view.setObjectName("view")
        # self.verticalLayout_3.addWidget(self.view)
        # self.selectVideo = QtWidgets.QPushButton(self.tab_2)
        # self.selectVideo.setObjectName("selectVideo")
        # self.verticalLayout_3.addWidget(self.selectVideo)
        # self.selectVideo.clicked.connect(self.selectVideoPath)
        # self.stopVideo = QtWidgets.QPushButton(self.tab_2)
        # self.stopVideo.setObjectName("stopVideo")
        # self.selectVideo.clicked.connect(self.stop)
        # self.verticalLayout_3.addWidget(self.stopVideo)
        # self.verticalLayout_4.addLayout(self.verticalLayout_3)
        # self.tabWidget.addTab(self.tab_2, "")

        self.movie = QMovie("loading.gif")
        self.thread = {}
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Detect Traffic Light"))
        self.labelTraffic.setText(_translate("MainWindow", ""))
        self.selectImage1.setText(_translate("MainWindow", "Chọn ảnh"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Phân loại nhãn"))
        self.selectImage2.setText(_translate("MainWindow", "Chọn ảnh"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Ảnh"))
        # self.selectVideo.setText(_translate("MainWindow", "Chọn Video"))
        # self.stopVideo.setText(_translate("MainWindow", "Dừng"))
        # self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Video"))

    def selectImage(self):
        file = QFileDialog.getOpenFileName(None, 'Open File', 'c\\', 'Image files (*.jpg)', 'Image files (*.jpg)')
        if file:
            imagePath = file[0]
            pixmap = QPixmap(imagePath)
            self.labelOriginal.setPixmap(QPixmap(pixmap))
            self.labelOriginal.setScaledContents(True)

            self.labelResult.setMovie(self.movie)
            self.labelResult.setScaledContents(True)
            self.movie.start()

            self.thread[0] = detect_image(index=1, imagePathh=imagePath)
            self.thread[0].start()
            self.thread[0].signal.connect(self.showResultImg)

    def showResultImg(self, imagePath):
        self.labelResult.setPixmap(QPixmap(imagePath))
        self.movie.stop()
        self.labelResult.setScaledContents(True)

    def selectVideoPath(self):
        file = QFileDialog.getOpenFileName(None, 'Open Video', 'c\\', 'Video files (*.mp4)', 'Video files(*.mp4)')

    def show_frame(self, label):
        """Updates the image_label with a new opencv image"""
        print(label)
        # ori_qt_img = self.convert_cv_qt(output)
        # self.view.setPixmap(ori_qt_img)
        # self.view.setScaledContents(True)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def stop(self):
        self.thread[2].terminate()

    def selectImageForClassifier(self):
        file = QFileDialog.getOpenFileName(None, 'Open File', 'c\\', 'Image files (*.jpg)', 'Image files (*.jpg)')
        if file:
            imagePath = file[0]
            pixmap = QPixmap(imagePath)
            self.originalPic1.setPixmap(QPixmap(pixmap))
            self.originalPic1.setScaledContents(True)

            self.thread[1] = label_classified(index=2, imagePath=imagePath)
            self.thread[1].start()
            self.thread[1].signal.connect(self.showLabelTraffic)

    def showLabelTraffic(self, label):
        self.labelTraffic.setText(label)

class detect_video(QThread):
    signal = pyqtSignal(str)
    def __init__(self, index, videoPath):
        self.index = index
        self.videoPath = videoPath
        print("start threading", self.index)
        super(detect_video, self).__init__()

    def run(self):
        print(self.videoPath)
        # cap = cv2.VideoCapture(self.videoPath)  # 'D:/8.Record video/My Video.mp4'
        # while True:
        #     ret, cv_img = cap.read()
        #     if ret:
        #         label = detectLabelFrame(cv_img)
        #         print(label)
        #         if(label != ""):
        #             self.signal.emit(label)
                    

class detect_image(QThread):
    signal = pyqtSignal(str)
    def __init__(self, index, imagePathh):
        self.index = index
        self.imagePath = imagePathh
        print("start threading", self.index)
        super(detect_image, self).__init__()
        
    def run(self):
        file_name = detectTraffic(self.imagePath)
        while True:
            if(file_name != ""):
                self.signal.emit(file_name)
                break

class label_classified(QThread):
    signal = pyqtSignal(str)
    def __init__(self, index, imagePath):
        self.index = index
        self.imagePath = imagePath
        print("start threading", self.index)
        super(label_classified, self).__init__()

    def run(self):
        label = detectLabel(self.imagePath)
        while True:
            if(label != ""):
                self.signal.emit(label)
                break

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())