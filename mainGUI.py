from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(649, 589)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea.setGeometry(QtCore.QRect(10, -10, 801, 561))
        self.mdiArea.setObjectName("mdiArea")
        self.NeuralNetwork = QtWidgets.QWidget()
        self.NeuralNetwork.setObjectName("NeuralNetwork")
        self.spinBox = QtWidgets.QSpinBox(self.NeuralNetwork)
        self.spinBox.setGeometry(QtCore.QRect(190, 40, 81, 22))
        self.spinBox.setProperty("value", 10)
        self.spinBox.setObjectName("spinBox")
        self.pushButton = QtWidgets.QPushButton(self.NeuralNetwork)
        self.pushButton.setGeometry(QtCore.QRect(120, 120, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.NeuralNetwork)
        self.label.setGeometry(QtCore.QRect(20, 40, 161, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.NeuralNetwork)
        self.label_2.setGeometry(QtCore.QRect(60, 160, 191, 20))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.NeuralNetwork)
        self.label_3.setGeometry(QtCore.QRect(20, 80, 161, 20))
        self.label_3.setObjectName("label_3")
        self.spinBox_2 = QtWidgets.QSpinBox(self.NeuralNetwork)
        self.spinBox_2.setGeometry(QtCore.QRect(190, 80, 81, 22))
        self.spinBox_2.setProperty("value", 10)
        self.spinBox_2.setObjectName("spinBox_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.NeuralNetwork)
        self.pushButton_3.setGeometry(QtCore.QRect(120, 190, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")

        self.HierarchicalClustering = QtWidgets.QWidget()
        self.HierarchicalClustering.setObjectName("HierarchicalClustering")
        self.pushButton_2 = QtWidgets.QPushButton(self.HierarchicalClustering)
        self.pushButton_2.setGeometry(QtCore.QRect(130, 120, 161, 23))
        self.pushButton_2.setObjectName("pushButton_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 649, 21))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_Open_Training_File = QtWidgets.QAction(MainWindow)
        self.action_Open_Training_File.setObjectName("action_Open_Training_File")
        self.action_Neural_Network = QtWidgets.QAction(MainWindow)
        self.action_Neural_Network.setObjectName("action_Neural_Network")
        self.actionHierarchical_Clustering = QtWidgets.QAction(MainWindow)
        self.actionHierarchical_Clustering.setObjectName("actionHierarchical_Clustering")
        self.menuMenu.addAction(self.action_Open_Training_File)
        self.menuMenu.addAction(self.action_Neural_Network)
        self.menuMenu.addAction(self.actionHierarchical_Clustering)
        self.menubar.addAction(self.menuMenu.menuAction())

        # Association Rule Mining
        self.AssociationRule = QtWidgets.QWidget()
        self.AssociationRule.setObjectName("AssociationRule")

        # AR - Text input
        self.ar_text_box = QtWidgets.QLineEdit(self.AssociationRule)
        self.ar_text_box.setGeometry(QtCore.QRect(10, 20, 180, 30))
        self.ar_text_box.setObjectName("ar_text_box")

        # AR - buttons
        self.ar_button = QtWidgets.QPushButton(self.AssociationRule)
        self.ar_button.setGeometry(QtCore.QRect(10, 60, 401, 30))
        self.ar_button.setObjectName("ar_button")
        self.ar_button.setText("Generate Association Suggestions")

        # AR - actions
        self.ar_action = QtWidgets.QAction(MainWindow)
        self.ar_action.setObjectName("ar_action")

        # AR - Menu action
        self.menuMenu.addAction(self.ar_action)

        # AR - List Widget
        self.ar_list_widget = QtWidgets.QListWidget(self.AssociationRule)
        self.ar_list_widget.setGeometry(QtCore.QRect(10, 100, 401, 192))
        self.ar_list_widget.setObjectName("ar_list_widget")

        # # AR - list View
        # self.ar_list_view = QtWidgets.QListView(self.AssociationRule)
        # self.ar_list_view.setGeometry(QtCore.QRect(10, 260, 401, 91))
        # self.ar_list_view.setObjectName("ar_list_view")


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Cuisine Data Mining"))
        self.NeuralNetwork.setWindowTitle(_translate("MainWindow", "Subwindow"))
        self.pushButton.setText(_translate("MainWindow", "Train"))
        self.label.setText(_translate("MainWindow", "Num of Hidden Layer Neurons:"))
        self.label_3.setText(_translate("MainWindow", "Num of epoch:"))
        self.pushButton_3.setText(_translate("MainWindow", "Create csv"))
        self.AssociationRule.setWindowTitle(_translate("MainWindow", "Subwindow"))
        self.pushButton_2.setText(_translate("MainWindow", "Run Clustering"))
        self.HierarchicalClustering.setWindowTitle(_translate("MainWindow", "Subwindow"))
        self.ar_button.setText(_translate("MainWindow", "Association Rule Suggestion"))
        self.menuMenu.setTitle(_translate("MainWindow", "&Menu"))
        self.action_Open_Training_File.setText(_translate("MainWindow", "&Open Training File"))
        self.action_Neural_Network.setText(_translate("MainWindow", "&Neural Network"))
        self.actionHierarchical_Clustering.setText(_translate("MainWindow", "Hierarchical &Clustering"))

        # AR
        self.ar_action.setText(_translate("MainWindow", "&Association Rule"))
