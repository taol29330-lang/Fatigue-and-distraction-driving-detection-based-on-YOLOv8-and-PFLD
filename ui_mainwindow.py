# -*- coding: utf-8 -*-

################################################################################
## Form converted to PySide6-compatible version from original PySide2 UI
################################################################################
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from PySide6.QtCore import QCoreApplication, QRect, QSize, QMetaObject, Qt
from PySide6.QtGui import QAction, QFont
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QMenu, QMenuBar, QSizePolicy,
    QStatusBar, QTextBrowser, QVBoxLayout, QHBoxLayout, QWidget
)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1060, 578)

        self.actionOpen_camera = QAction(MainWindow)
        self.actionOpen_camera.setObjectName("actionOpen_camera")
        self.actionBaseline = QAction(MainWindow)
        self.actionBaseline.setObjectName("actionBaseline")

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # ===== White theme (global) =====
        MainWindow.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
            }
            QWidget#centralwidget {
                background-color: #FFFFFF;
            }
            QLabel {
                color: #111111;
                background: transparent;
            }
            QMenuBar {
                background-color: #FFFFFF;
                color: #111111;
            }
            QMenu {
                background-color: #FFFFFF;
                color: #111111;
            }
            QStatusBar {
                background-color: #FFFFFF;
                color: #111111;
            }
            QTextBrowser {
                background-color: #FFFFFF;
                color: #111111;
                border: 1px solid #E5E5E5;
                border-radius: 8px;
            }
        """)

        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.label.setMinimumSize(QSize(720, 480))
        self.label.setMaximumSize(QSize(720, 480))

        # welcome label centered + wrap
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)

        # welcome label style (pure white)
        self.label.setStyleSheet("""
            QLabel#label {
                background-color: #FFFFFF;
                border: 2px solid #E5E5E5;
                border-radius: 12px;
                color: #111111;
                padding: 18px;
            }
        """)

        f = QFont()
        f.setFamily("Microsoft YaHei")
        f.setPointSize(20)
        f.setBold(True)
        self.label.setFont(f)

        self.horizontalLayout.addWidget(self.label)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.label_2.setMaximumSize(QSize(120, 30))
        self.horizontalLayout_5.addWidget(self.label_2)
        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName("label_10")
        self.label_10.setMaximumSize(QSize(180, 30))
        self.horizontalLayout_5.addWidget(self.label_10)
        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.label_3.setMaximumSize(QSize(150, 30))
        self.horizontalLayout_2.addWidget(self.label_3)
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.label_4.setMaximumSize(QSize(150, 30))
        self.horizontalLayout_2.addWidget(self.label_4)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.label_5.setMaximumSize(QSize(120, 30))
        self.horizontalLayout_4.addWidget(self.label_5)
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.label_9.setMaximumSize(QSize(180, 30))
        self.horizontalLayout_4.addWidget(self.label_9)
        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.label_6.setMaximumSize(QSize(100, 30))
        self.horizontalLayout_3.addWidget(self.label_6)
        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.label_7.setMaximumSize(QSize(100, 30))
        self.horizontalLayout_3.addWidget(self.label_7)
        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.label_8.setMaximumSize(QSize(100, 30))
        self.horizontalLayout_3.addWidget(self.label_8)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.textBrowser = QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setMaximumSize(QSize(300, 360))
        self.verticalLayout.addWidget(self.textBrowser)

        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 1060, 26))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.actionOpen_camera)
        self.menu.addAction(self.actionBaseline)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", "驾驶员状态监测系统", None))

        self.actionOpen_camera.setText(QCoreApplication.translate("MainWindow", "启动驾驶状态监测", None))
        self.actionBaseline.setText(QCoreApplication.translate("MainWindow", "采集EAR/MAR/PERCLOS基线", None))

        self.label.setText(QCoreApplication.translate(
            "MainWindow",
            "欢迎使用驾驶员状态监测系统\n\n"
            "请点击右上角菜单open：\n\n"
            "启动驾驶状态监测 或 采集EAR/MAR/PERCLOS基线",
            None
        ))

        self.label_2.setText(QCoreApplication.translate("MainWindow", "FPS:", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", "清醒", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", "眨眼次数：0", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", "哈欠次数：0", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", "提示：", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", "", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", "手机", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", "抽烟", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", "喝水", None))

        self.menu.setTitle(QCoreApplication.translate("MainWindow", "Open", None))

    def printf(self, mes):
        self.textBrowser.append(mes)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(QTextCursor.End)