# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:54:21 2024

@author: DELL
"""

import os
import sys
if sys.platform.startswith('win'):
    ## need to add environ aug here ## required by windows verision Graphviz
    ## maybe not needed
    os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin'

import shutil
import string
import random

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from graphviz import Digraph
from PIL import Image, ImageDraw,ImageFont

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QVariant, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QHeaderView

from GUI.uic.NPRetroMainWindow import Ui_MainWindow
from GUI.core import utils
from GUI.widget.PubchemSketcher import PubchemSketcherUI
from GUI.widget.ChemicalImage import ChemicalImageUI
from GUI.widget.Parameters_Setting import ParametersUI
from datetime import datetime


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, showAllColumn=False):
        QtCore.QAbstractTableModel.__init__(self)
        self.showAllColumn = showAllColumn
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if type(self._data.columns[col]) == tuple:
                return self._data.columns[col][-1]
            else:
                return self._data.columns[col]
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return (self._data.axes[0][col])
        return None

class Thread_MultiStepPlanning(QThread):
    _r = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, precursor_smiles_list, model_type, beam_size, exp_topk, iterations, route_topk, device, model_path):
        super().__init__()
        self.smiles_list = precursor_smiles_list
        self.model_type = model_type
        self.beam_size = beam_size
        self.exp_topk = exp_topk
        self.iterations = iterations
        self.route_topk = route_topk
        self.device = device
        self.send_model_path = model_path
    def run(self):
        pathway_list = utils.predict_compound_derivative_GSETransformer(self.smiles_list, self.model_type,
                    self.beam_size, self.exp_topk,self.iterations, self.route_topk,  self.device, self.send_model_path)

        self._r.emit(pathway_list)


class Thread_PredictADMET(QThread):
    _r = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, smiles_list):
        super().__init__()
        self.smiles_list = smiles_list

    def run(self):
        admet_list = utils.predict_compound_ADMET_property(smiles_list=self.smiles_list)
        self._r.emit(admet_list)

class NPRetro_App(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(NPRetro_App, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("NP Retrosynthesis Screening")
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')

        try:
            shutil.rmtree('data/temp_data')
            os.makedirs('data/temp_data')
        except:
            os.makedirs('data/temp_data')

        self.precursor = None

        self.PubchemSketcherUI = PubchemSketcherUI()
        self.ChemicalImageUI = ChemicalImageUI()
        self.ParametersUI = ParametersUI()
        # layout input area
        self.pushButton_chem_add.clicked.connect(self.add_precursor_to_list)
        self.pushButton_chem_upload.clicked.connect(self.upload_precursor_to_list)
        self.pushButton_chem_draw.clicked.connect(self.PubchemSketcherUI.show)
        self.pushButton_chem_clear.clicked.connect(self.listWidget_chem_list.clear)
        self.pushButton_chem_show.clicked.connect(self.show_chemical_image)
        # predict button
        self.pushButton_chem_predict.clicked.connect(self.do_multi_step_planning)
        # setting button
        self.pushButton_setting.clicked.connect(self.ParametersUI.open)
        self.ParametersUI.button_ModelPath.clicked.connect(self.load_model_path)
        self.model_path = None
        # save buttom
        self.pushButton_save.clicked.connect(self.save_as_file)  # new add here
        # data generated in the process
        self.pathway_list = None
        self.Thread_MultiStepPlanning = None
        self.ADMET_list = None
        self.Thread_ADMET = None
        # 'Retrosynthetic Route Prediction' area functions
        self.tableWidget_RouteList.setSortingEnabled(True)
        self.tableWidget_RouteList.cellClicked.connect(self.fill_table_single_route_list)
        # 'Precursor Compound' area functions functions
        self.tableWidget_Compounds.cellClicked.connect(self.fill_AMDET_table_click_compound)

    def load_model_path(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", "", "pt. Files (*.pt)",options=options)
        self.model_path = fileName
        print(self.model_path)

    def save_as_file(self):  # self.pathway_list 可能为none
        if isinstance(self.pathway_list, pd.DataFrame):
            if not self.pathway_list.empty:  # self.pathway_list不为None且不为空

                model_type = self.ParametersUI.comboBox_Model_Type.currentText()
                #beam_size = self.ParametersUI.spinBox_Beam_Size.value()
                exp_topk = self.ParametersUI.spinBox_Expansion_Number.value()
                iterations = self.ParametersUI.spinBox_Iterations_Number.value()
                route_topk = self.ParametersUI.spinBox_Route_Number.value()

                time = datetime.now().strftime("%m%d_%H%M%S")
                file_name = f'route_list_{model_type}_{exp_topk}xx{iterations}_{time}'
                destpath, filetype = QtWidgets.QFileDialog.getSaveFileName(self, "Select the save path", f"{file_name}",
                                                                           "csv Files (*.csv)")
                if destpath:
                    folder_path = destpath.rsplit('/', 1)[0]
                    print(folder_path, filetype)
                    self.pathway_list.to_csv(destpath)
                self.InforMsg('Finished')
            else:
                self.WarnMsg('No result to save')
        else:
            self.WarnMsg('No result to save')

    def WarnMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(Text)
        msg.setWindowTitle("Warning")
        msg.exec_()

    def ErrorMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(Text)
        msg.setWindowTitle("Error")
        msg.exec_()

    def InforMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(Text)
        msg.setWindowTitle("Information")
        msg.exec_()

    def _set_table_widget(self, widget, data):
        widget.setRowCount(0)

        widget.setRowCount(data.shape[0])
        widget.setColumnCount(data.shape[1])
        widget.setHorizontalHeaderLabels(data.columns)
        widget.setVerticalHeaderLabels(data.index.astype(str))
        widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)#列宽自动分配
        #widget.horizontalHeader().setDefaultSectionSize(110)#表头高度
        #widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)#手动调整

        widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)#根据内容分配列宽
        #widget.horizontalHeader().setVisible(False)#隐藏表头
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if type(data.iloc[i, j]) == np.float64:
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(Qt.EditRole, QVariant(float(data.iloc[i, j])))
                else:
                    item = QtWidgets.QTableWidgetItem(str(data.iloc[i, j]))
                widget.setItem(i, j, item)
    def _set_AMDET_list(self, msg):
        self.ADMET_list = msg

    def _set_pathway_list(self, msg):
        self.pathway_list = msg

    def _clear_all(self):
        self.pathway_list = None
        self.ADMET_list = None
        # clear all data
        self.tableWidget_RouteList.clear()
        self.tableWidget_Compounds.clear()
        self.tableWidget_EC_Number.clear()
        self.label_7.clear()

    def _set_disable(self):
        # setDisabled all pushButton
        self.pushButton_chem_add.setDisabled(True)
        self.pushButton_chem_upload.setDisabled(True)
        self.pushButton_chem_draw.setDisabled(True)
        self.pushButton_chem_clear.setDisabled(True)
        self.pushButton_chem_show.setDisabled(True)
        self.pushButton_chem_predict.setDisabled(True)

        self.pushButton_setting.setDisabled(True)
        self.pushButton_save.setDisabled(True)

    def _set_finished(self):# return to be abled
        self.pushButton_chem_add.setDisabled(False)
        self.pushButton_chem_upload.setDisabled(False)
        self.pushButton_chem_draw.setDisabled(False)
        self.pushButton_chem_clear.setDisabled(False)
        self.pushButton_chem_show.setDisabled(False)
        self.pushButton_chem_predict.setDisabled(False)

        self.pushButton_setting.setDisabled(False)
        self.pushButton_save.setDisabled(False)

    def add_precursor_to_list(self):
        precursor_smi = (self.plainTextEdit_chem_inp.toPlainText()).replace(' ', '')# if blank exists
        if precursor_smi == '':
            self.ErrorMsg('Invalid input molecule')
            return
        precursor_mol = Chem.MolFromSmiles(precursor_smi)
        if precursor_mol is None:
            self.ErrorMsg('Invalid input molecule')
            return
        self.listWidget_chem_list.addItem(Chem.MolToSmiles(precursor_mol))
        self.plainTextEdit_chem_inp.clear()
        self.InforMsg('Finished')

    def upload_precursor_to_list(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load", "", "Smiles Files (*.smi);;SDF Files (*.sdf)",
                                                            options=options)
        if fileName:
            if fileName.split('.')[-1] == 'smi':
                with open(fileName) as f:
                    smiles_list = f.readlines()
                mol_list = [Chem.MolFromSmiles(s) for s in smiles_list]
                mol_list = [m for m in mol_list if m is not None]
                smiles_list = [Chem.MolToSmiles(m) for m in mol_list]
            elif fileName.split('.')[-1] == 'sdf':
                mol_list = Chem.SDMolSupplier(fileName)
                smiles_list = [Chem.MolToSmiles(m) for m in mol_list]
            else:
                self.ErrorMsg("Invalid format")
                return None
            for smi in smiles_list:
                self.listWidget_chem_list.addItem(smi)
        self.InforMsg('Finished')

    def show_chemical_image(self):
        precursor_smi = self.listWidget_chem_list.currentItem()
        if not precursor_smi:
            self.WarnMsg('Please select a compound')
            return
        precursor_smi = precursor_smi.text()
        self.ChemicalImageUI.show()
        precursor_mol = Chem.MolFromSmiles(precursor_smi)
        file_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        Draw.MolToFile(precursor_mol, f'data/temp_data/{file_name}.png')
        self.ChemicalImageUI.label_chem_image.setPixmap(QPixmap(f'data/temp_data/{file_name}.png'))
    ####################################################################
    ## 01. Multi_Step_Planning function
    def do_multi_step_planning(self):

        precursor_smiles_list = [self.listWidget_chem_list.item(x).text() for x in
                                 range(self.listWidget_chem_list.count())]
        if len(precursor_smiles_list) == 0:
            self.ErrorMsg('No valid input,SMILES List is empty')
            return

        model_type = self.ParametersUI.comboBox_Model_Type.currentText()
        device = self.ParametersUI.comboBox_Device.currentText()
        exp_topk = self.ParametersUI.spinBox_Expansion_Number.value()
        # beam_size = self.ParametersUI.spinBox_Beam_Size.value()
        beam_size = exp_topk
        iterations = self.ParametersUI.spinBox_Iterations_Number.value()
        route_topk = self.ParametersUI.spinBox_Route_Number.value()
        model_path = self.model_path

        self._clear_all()
        self._set_disable()

        self.progressBar.setValue(30)
        self.progressBar.setFormat('Multi-step Planing')
        print('>>> Multi-step Planing')

        if model_path == None or not os.path.exists(model_path):
            self.ErrorMsg('Please specify a valid model path')
            self.progressBar.setValue(100)
            self.progressBar.setFormat('Ready')
            self._set_finished()

        else:
            self.Thread_MultiStepPlanning = Thread_MultiStepPlanning(
                precursor_smiles_list, model_type, beam_size, exp_topk, iterations, route_topk, device, model_path)
            self.Thread_MultiStepPlanning._r.connect(self._set_pathway_list)
            self.Thread_MultiStepPlanning.start()
            self.Thread_MultiStepPlanning.finished.connect(self.fill_table_route_prediction)

    def get_pure_route_from_pred_full_route(self, full_route):
        '''
        :param s: 'smiles1>score1>smiles2|smiles2>score2>smiles3|smiles3>score3>smiles4...'
        :return: [smiles1, smiles2, smiles3, ...]
        '''
        smi_lst = []
        reaction_lst = full_route.split('|')
        pure_route = ''
        for i in range(len(reaction_lst)):
            line = reaction_lst[i].split('>')
            if i == 0:
                smi_lst.append(line[0])
                pure_route += (line[0]+">")
            smi_lst.append(line[-1])
            pure_route += (line[-1] + ">")
        pure_route = pure_route.strip(">")

        return smi_lst, pure_route

    def fill_table_route_prediction(self):
        self._set_disable()
        if self.pathway_list.empty:
            self.ErrorMsg('No valid prediction for current molecule')
            self.progressBar.setValue(100)
            self.progressBar.setFormat('Ready')
            self._set_finished()
        else:
            raw_route_lst = list(self.pathway_list['pathway_prediction']) # GUI/multi_step_plan_4GUI.py
            # 1/2 fill tableWidget_RouteList
            df_routeList = pd.DataFrame({' ': raw_route_lst})
            self._set_table_widget(self.tableWidget_RouteList, df_routeList)
            self.tableWidget_RouteList.setCurrentCell(0, 0)
            # 2/2 do predict_ADMET for all_smiles
            full_smi_lst = []
            for r in raw_route_lst:
                smi_lst, _route = self.get_pure_route_from_pred_full_route(r)
                for smi in smi_lst:
                    if smi not in full_smi_lst and smi.find('kegg') == -1:
                        full_smi_lst.append(smi)
            self.predict_ADMET(full_smi_lst)

    ####################################################################
    ## 02. ADMET function
    def predict_ADMET(self, full_smi_lst):
        self.progressBar.setValue(70)
        self.progressBar.setFormat('Predicting AMDET')
        self.Thread_PredictAMDET = Thread_PredictADMET(full_smi_lst)
        self.Thread_PredictAMDET._r.connect(self._set_AMDET_list)
        self.Thread_PredictAMDET.start()
        self.Thread_PredictAMDET.finished.connect(self.finish_pop_up)
        # finished
    def finish_pop_up(self):
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self._set_finished()
        self.InforMsg('Finished')

    def fill_AMDET_table_click_compound(self):
        index = self.tableWidget_Compounds.currentRow()
        current_smiles = self.tableWidget_Compounds.item(index, 0).text()
        self.fill_AMDET_table(current_smiles)
    def fill_AMDET_table(self, current_smiles=None):
        if current_smiles.find('kegg') == -1:
            Physicochemical = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles,
                                                                   property_class='Physicochemical')
            Absorption = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class='Absorption')
            Distribution = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles,
                                                                property_class='Distribution')
            Metabolism = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class='Metabolism')
            Excretion = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class='Excretion')
            Toxicity = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class='Toxicity')

            self.tableView_prop_1.setModel(TableModel(Physicochemical))
            self.tableView_prop_2.setModel(TableModel(Absorption))
            self.tableView_prop_3.setModel(TableModel(Distribution))
            self.tableView_prop_4.setModel(TableModel(Metabolism))
            self.tableView_prop_5.setModel(TableModel(Excretion))
            self.tableView_prop_6.setModel(TableModel(Toxicity))
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self._set_finished()
    ####################################################################
    ## 03. Show selected single route:Compounds & EC Number Prediction & visualization
    def fill_table_single_route_list(self):
        # show compounds and synthesis-path for selected pathway
        # 0. get current/selected pathway
        index = self.tableWidget_RouteList.currentRow()
        get_full_route = self.tableWidget_RouteList.item(index, 0).text()
        self._set_disable()
        # 1. analysis
        all_paths = utils.find_all_paths(get_full_route)
        all_unique_mole_node = list(set([element for sublist in all_paths for element in sublist]))
        # 2. EC Number Predicting
        rxn_EC_dict = None
        if_ec_predict = self.ParametersUI.comboBox_ECPredict.currentText()
        if if_ec_predict == 'Yes':
            print('Reaction EC Number Predicting')
            self.progressBar.setValue(30)
            self.progressBar.setFormat('EC Number Predicting')
            df_all_ec, rxn_EC_dict = utils.predict_rxns_EC_number(get_full_route)
            self._set_table_widget(self.tableWidget_EC_Number, df_all_ec)
        else:
            self.tableWidget_EC_Number.clear()
        # 3. get_pure_route_from_pred_full_route
        smi_lst, route = self.get_pure_route_from_pred_full_route(get_full_route)
        self.fill_AMDET_table(smi_lst[0])
        df_smi_lst = pd.DataFrame({' ': smi_lst})
        self._set_table_widget(self.tableWidget_Compounds, df_smi_lst)
        # 4. draw pathway
        node_id_dict = {}
        for node_id in range(len(all_unique_mole_node)):
            node_id_dict[all_unique_mole_node[node_id]] = node_id
        current_path = os.path.dirname(__file__)
        G = Digraph('G', filename='data/temp_data/return_synthesis_path')
        G.attr('node', shape='box')
        G.format = 'png'
        G_save_path = 'data/temp_data/return_synthesis_path.png'

        already_node_lst = []
        for single_path in all_paths:
            for i in range(len(single_path)):
                if single_path[i].find('kegg') == -1:
                    try:
                        mol = Chem.MolFromSmiles(single_path[i])
                    except:
                        print('invalid molecular SMILES string')
                    else:
                        node_id = node_id_dict[single_path[i]]
                        save_path = f'{current_path}/data/temp_data/smi{node_id}-mol-graph.png'
                        Draw.MolToFile(mol, save_path, size=(400, 400))
                        if single_path[i] not in already_node_lst:
                            already_node_lst.append(single_path[i])
                            G.node(name=single_path[i], image=save_path, label='', labelloc='top')
                            if i >= 1:
                                if rxn_EC_dict != None:
                                    top1_ec = rxn_EC_dict[single_path[i - 1]]
                                    top1_ec = top1_ec.split('/')[0]
                                    G.edge(single_path[i - 1], single_path[i], label=f'{top1_ec}')
                                else:
                                    G.edge(single_path[i - 1], single_path[i])

                else:
                    node_id = node_id_dict[single_path[i - 1]]
                    save_path = f'{current_path}/data/temp_data/smi{node_id}-mol-graph.png'
                    open_image2change = Image.open(save_path)
                    draw = ImageDraw.Draw(open_image2change)
                    text = '↓' + '\n' + f"→{single_path[i].replace('keggpath=', ' ')}"
                    text_font = ImageFont.truetype("arial.ttf", size=20)
                    text_color = (0, 0, 0)
                    outline_color = (255, 0, 0)
                    position = (50, 350)
                    draw.text(position, text, font=text_font, fill=text_color, outline=outline_color)
                    open_image2change.save(save_path)

        G.render()
        load_image = QPixmap(G_save_path)
        width = load_image.width()
        height = load_image.height()  ##获取图片高度
        print('show_pathway: original pic size', width, height)
        print('show_pathway: label size', self.label_7.width(), self.label_7.height())
        ratio = (self.label_7.width() / width)
        new_width = int(width * ratio * 0.9)  ##定义新图片的宽和高
        new_height = int(height * ratio * 0.9)
        print('show_pathway: resized pic size', new_width, new_height)
        pic2show = QtGui.QPixmap(G_save_path).scaled(new_width, new_height, Qt.KeepAspectRatioByExpanding,
                                                     Qt.SmoothTransformation)
        self.label_7.setPixmap(pic2show)

        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self._set_finished()

# pyuic5 GUI/ui/NPRetroMainWindow.ui -o GUI/uic/NPRetroMainWindow.py

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ui = NPRetro_App()
    ui.show()
    sys.exit(app.exec_())