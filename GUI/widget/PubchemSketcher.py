# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:50:51 2024

@author: DELL
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
from GUI.uic.pubchem_sketcher import Ui_Form

class PubchemSketcherUI(QtWidgets.QWidget, Ui_Form):
    
    def __init__(self, parent=None):
        super(PubchemSketcherUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Pubchem Sketcher")
        
        self.pushButton_confirm.clicked.connect(self.get_smiles)
        

    def get_smiles(self):
        self.close()



if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = PubchemSketcherUI()
    ui.show()
    sys.exit(app.exec_())