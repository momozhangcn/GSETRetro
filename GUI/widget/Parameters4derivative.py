# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:50:51 2024

@author: DELL
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from GUI.uic.parameters_derivative import Ui_Dialog

class ParametersUI(QtWidgets.QDialog, Ui_Dialog):
    
    def __init__(self, parent=None):
        super(ParametersUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Setting")
        
        self.comboBox_der_type.addItems(['Chemical', 'Biological'])
        self.comboBox_dta_model.addItems(['CNN-CNN', 'MPNN-CNN', 'Morgan-CNN', 'baseTransformer-CNN'])
        self.add_comboBox_der_model()
        self.comboBox_der_type.currentTextChanged.connect(self.add_comboBox_der_model)
        
    def add_comboBox_der_model(self):
        if self.comboBox_der_type.currentText() == 'Chemical':
            self.comboBox_der_model.clear()
            self.comboBox_der_model.addItems(['Chemical-Template-based',
                                              'Chemical-baseTransformer',
                                              'Chemical-MolBert'])
        
        if self.comboBox_der_type.currentText() == 'Biological':
            self.comboBox_der_model.clear()
            self.comboBox_der_model.addItems(['BioTransformer-EC-based', 
                                              'BioTransformer-CYP450', 
                                              'BioTransformer-Phase II', 
                                              'BioTransformer-Human gut microbial', 
                                              'BioTransformer-All Human', 
                                              'BioTransformer-Environmental microbial',
                                              'Biological-baseTransformer',
                                              'Biological-MolBert'])
        
        

if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = ParametersUI()
    ui.show()
    sys.exit(app.exec_())