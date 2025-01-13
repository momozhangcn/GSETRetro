# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:50:51 2024

@author: DELL
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from ..uic.parameters_NPRetro import Ui_Dialog

class ParametersUI(QtWidgets.QDialog, Ui_Dialog):
    
    def __init__(self, parent=None):
        super(ParametersUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Setting")
        self.comboBox_Model_Type.addItems(['GSETransformer', 'GSETransformer+Retriver'])
        self.comboBox_Device.addItems(['CPU', 'GPU'])

        
        

if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = ParametersUI()
    ui.show()
    sys.exit(app.exec_())