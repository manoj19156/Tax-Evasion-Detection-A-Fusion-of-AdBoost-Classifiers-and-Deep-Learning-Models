import sys
import os
from tax import *
from PyQt5 import QtWidgets, QtGui, QtCore

class MyForm(QtWidgets.QMainWindow):
  def __init__(self,parent=None):
     QtWidgets.QWidget.__init__(self,parent)
     self.ui = Ui_MainWindow()
     self.ui.setupUi(self)
     self.ui.pushButton.clicked.connect(self.abcacc)
     self.ui.pushButton_2.clicked.connect(self.nnacc)
     self.ui.pushButton_3.clicked.connect(self.nnpred)
     self.ui.pushButton_4.clicked.connect(self.abcpred)

  def abcacc(self):
    os.system("python -W ignore abc1.py")

  def nnacc(self):
    os.system("python -W ignore nn1.py")

  def nnpred(self):
    os.system("python -W ignore nn2.py")

  def abcpred(self):  
    os.system("python -W ignore abc2.py")

       
if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
