# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:50:51 2024

@author: DELL
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
from ..uic.chemical_image import Ui_Form

class ChemicalImageUI(QtWidgets.QWidget, Ui_Form):
    
    def __init__(self, parent=None):
        super(ChemicalImageUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Image")
        


if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = ChemicalImageUI()
    ui.show()
    sys.exit(app.exec_())