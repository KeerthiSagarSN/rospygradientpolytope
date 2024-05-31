# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interactive_ik.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(240, 320)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(10, 270, 221, 41))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.run_ik = QtWidgets.QPushButton(Dialog)
        self.run_ik.setGeometry(QtCore.QRect(30, 20, 181, 61))
        self.run_ik.setObjectName("run_ik")
        self.polytope_on = QtWidgets.QPushButton(Dialog)
        self.polytope_on.setGeometry(QtCore.QRect(30, 90, 181, 51))
        self.polytope_on.setObjectName("polytope_on")
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setGeometry(QtCore.QRect(10, 210, 221, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.polytope_off = QtWidgets.QPushButton(Dialog)
        self.polytope_off.setGeometry(QtCore.QRect(30, 150, 181, 51))
        self.polytope_off.setObjectName("polytope_off")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.run_ik.setText(_translate("Dialog", "Run IK"))
        self.polytope_on.setText(_translate("Dialog", "Display Polytope: ON"))
        self.polytope_off.setText(_translate("Dialog", "Display Polytope: OFF"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
