# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:54:21 2024

@author: DELL
"""

import os
import sys

if sys.platform.startswith('win'):  ## need to add environ aug here ## required by windows verision Graphviz
    os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin'

import shutil
import string
import random

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from graphviz import Digraph

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


class Thread_PredictDerivative(QThread):
    _r = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, precursor_smiles_list, model_type, beam_size, exp_topk, iterations, route_topk, device):
        super().__init__()
        self.smiles_list = precursor_smiles_list
        self.model_type = model_type
        self.beam_size = beam_size
        self.exp_topk = exp_topk
        self.iterations = iterations
        self.route_topk = route_topk
        self.device = device
    def run(self):

        if self.model_type == 'GSETransformer':
            derivative_list = utils.predict_compound_derivative_GSETransformer(self.smiles_list,
                        self.model_type, self.beam_size, self.exp_topk,self.iterations, self.route_topk, self.device)

        ###
        else:
            derivative_list = None
        self._r.emit(derivative_list)



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
        super(NPRetro_App, self).__init__(parent)   #NPDS_App
        self.setupUi(self)
        self.setWindowTitle("NP Retrosynthesis Screening")
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')

        try:
            shutil.rmtree('temp')
            os.mkdir('temp')
        except:
            pass

        self.precursor = None

        self.PubchemSketcherUI = PubchemSketcherUI()
        self.ChemicalImageUI = ChemicalImageUI()
        self.ParametersUI = ParametersUI()
        #layout1
        self.pushButton_chem_add.clicked.connect(self.add_precursor_to_list)
        self.pushButton_chem_upload.clicked.connect(self.upload_precursor_to_list)
        self.pushButton_chem_draw.clicked.connect(self.PubchemSketcherUI.show)
        self.pushButton_chem_clear.clicked.connect(self.listWidget_chem_list.clear)
        self.pushButton_chem_show.clicked.connect(self.show_chemical_image)
        #add
        self.pushButton_chem_predict.clicked.connect(self.predict_compound_derivative)
        #top button
        self.pushButton_setting.clicked.connect(self.ParametersUI.open)
        self.pushButton_save.clicked.connect(self.save_as_file)  # new add here
        #
        self.routeList = None
        self.precursorList = None
        self.ADMET_list = None
        #
        self.derivative_list = None
        self.Thread_PredictDerivative = None
        self.Thread_ADMET = None
        #add
        self.tableWidget_RouteList.setSortingEnabled(True)
        self.tableWidget_RouteList.cellClicked.connect(self.fill_single_route_list)
        #self.tableWidget_RouteList.cellClicked.connect(self.show_synthesis_path)

        self.tableWidget_PrecursorList.cellClicked.connect(self.fill_AMDET_table)

    def save_as_file(self):  # self.derivative_list 可能为none 可能为
        if isinstance(self.derivative_list, pd.DataFrame):
            if not self.derivative_list.empty:  # self.derivative_list不为None且不为空

                model_type = self.ParametersUI.comboBox_Model_Type.currentText()
                beam_size = self.ParametersUI.spinBox_Beam_Size.value()
                exp_topk = self.ParametersUI.spinBox_Expansion_Number.value()
                iterations = self.ParametersUI.spinBox_Iterations_Number.value()
                route_topk = self.ParametersUI.spinBox_Route_Number.value()

                time = datetime.now().strftime("%m%d_%H%M%S")
                file_name = f'derivative_list_{model_type}_{exp_topk}xx{iterations}_{time}'
                destpath, filetype = QtWidgets.QFileDialog.getSaveFileName(self, "Select the save path", f"{file_name}",
                                                                           "csv Files (*.csv)")

                if destpath:
                    folder_path = destpath.rsplit('/', 1)[0]
                    print(folder_path, filetype)
                    self.derivative_list.to_csv(destpath)
                    # self.DTI_list.to_csv(f'{folder_path}/DTI_list_{method}_{n_branch}xx{n_loop}_{time}.csv')
                    # self.ADMET_list.to_csv(f'{folder_path}/ADMET_list_{method}_{n_branch}xx{n_loop}_{time}.csv')

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


    def _set_derivative_list(self, msg):
        self.derivative_list = msg

    def _set_AMDET_list(self, msg):
        self.ADMET_list = msg

    def _clear_all(self):
        self.derivative_list = None
        self.ADMET_list = None
        # clear all data
        self.tableWidget_PrecursorList.clear()
        self.tableWidget_RouteList.clear()

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
        precursor_smi = self.plainTextEdit_chem_inp.toPlainText()
        if precursor_smi == '':
            self.ErrorMsg('Invalid input molecule')
            return
        precursor_mol = Chem.MolFromSmiles(precursor_smi)
        if precursor_mol is None:
            self.ErrorMsg('Invalid input molecule')
            return
        self.listWidget_chem_list.addItem(Chem.MolToSmiles(precursor_mol))
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
        Draw.MolToFile(precursor_mol, 'temp/{}.png'.format(file_name))
        self.ChemicalImageUI.label_chem_image.setPixmap(QPixmap('temp/{}.png'.format(file_name)))
####################################################################
    def predict_compound_derivative(self):

        precursor_smiles_list = [self.listWidget_chem_list.item(x).text() for x in
                                 range(self.listWidget_chem_list.count())]
        if len(precursor_smiles_list) == 0:
            self.ErrorMsg('Precursor List or Target List is empty')
            return

        model_type = self.ParametersUI.comboBox_Model_Type.currentText()
        device = self.ParametersUI.comboBox_Device.currentText()
        beam_size = self.ParametersUI.spinBox_Beam_Size.value()
        exp_topk = self.ParametersUI.spinBox_Expansion_Number.value()
        iterations = self.ParametersUI.spinBox_Iterations_Number.value()
        route_topk = self.ParametersUI.spinBox_Route_Number.value()


        if exp_topk > beam_size: # kee exp_topk <= beam_size
            exp_topk = beam_size

        self._clear_all()
        self.progressBar.setValue(30)
        self.progressBar.setFormat('Predicting and Planing')
        self.Thread_PredictDerivative = Thread_PredictDerivative(
            precursor_smiles_list, model_type, beam_size, exp_topk, iterations, route_topk, device)

        self.Thread_PredictDerivative._r.connect(self._set_derivative_list)
        self.Thread_PredictDerivative.start()
        self.Thread_PredictDerivative.finished.connect(self.fill_pred_route_list)
        #add

    def get_pure_route_from_pred_full_route(self,full_route):
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

    def fill_pred_route_list(self):
        if self.derivative_list.empty:
            self.ErrorMsg('No valid prediction for current input precursor compound')

            self.progressBar.setValue(100)
            self.progressBar.setFormat('Ready')
            self._set_finished()
        else:
            raw_route_lst = list(self.derivative_list['pathway_prediction'])
            full_smi_lst = []
            route_lst = []
            for r in raw_route_lst:
                smi_lst, route = self.get_pure_route_from_pred_full_route(r)
                for smi in smi_lst:
                    if smi not in full_smi_lst:
                        full_smi_lst.append(smi)
                route_lst.append(route)

            df_routeList = pd.DataFrame({' ':route_lst})


            self._set_table_widget(self.tableWidget_RouteList, df_routeList)
            self.tableWidget_RouteList.setCurrentCell(0, 0)

            self.predict_ADMET(full_smi_lst)

            self.progressBar.setValue(100)
            self.progressBar.setFormat('Ready')
            self._set_finished()

    def predict_ADMET(self, full_smi_lst):

        self.progressBar.setValue(70)
        self.progressBar.setFormat('Predicting AMDET')
        self.Thread_PredictAMDET = Thread_PredictADMET(full_smi_lst)
        self.Thread_PredictAMDET._r.connect(self._set_AMDET_list)
        self.Thread_PredictAMDET.start()
        #self.Thread_PredictAMDET.finished.connect()


    def fill_single_route_list(self):
        index = self.tableWidget_RouteList.currentRow()
        get_route = self.tableWidget_RouteList.item(index, 0).text()
        smi_lst = get_route.split('>')
        self.show_synthesis_path_from_route(smi_lst)
        df_smi_lst = pd.DataFrame({' ':smi_lst})

        self._set_table_widget(self.tableWidget_PrecursorList, df_smi_lst)
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self._set_finished()


    def fill_AMDET_table(self):
        index = self.tableWidget_PrecursorList.currentRow()
        current_smiles = self.tableWidget_PrecursorList.item(index, 0).text()
        print(current_smiles)

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

    #### new add here
    def show_synthesis_path_from_route(self,smi_lst):
        # 生成分子图，并生成路径图
        G = Digraph('G', filename='data/temp_data/return_synthesis_path')
        G.attr('node', shape='box')
        G.format = 'png'
        G_save_path = 'data/temp_data/return_synthesis_path.png'

        current_path = os.path.dirname(__file__)
        for i in range(len(smi_lst)):
            print(f'synthesis_path:{i + 1}/{len(smi_lst)}', '\n', smi_lst[i])
            try:
                mol = Chem.MolFromSmiles(smi_lst[i])
            except:
                print('invalid molecular SMILES string')
            else:
                save_path = f'{current_path}/data/temp_data/smi{i}-mol-graph.png'
                Draw.MolToFile(mol, save_path, size=(400, 400))
                ## G.node needs the absolute paths, otherwise something wrong
                ## 需要读取绝对路径，不然可能报错。
                G.node(name=smi_lst[i], image=save_path, label='', labelloc='top')  # label=smi_lst[i] f'mol{i + 1}'
                if i >= 1:
                    G.edge(smi_lst[i - 1], smi_lst[i])  # 可加label # , label=f"no-level-expansion"
        G.render()
        load_image = QPixmap(G_save_path)
        width = load_image.width()
        height = load_image.height()  ##获取图片高度
        print('original pic size', width, height)
        print('label size', self.label_7.width(), self.label_7.height())
        ratio = (self.label_7.width() / width)
        new_width = int(width * ratio * 0.9)  ##定义新图片的宽和高
        new_height = int(height * ratio * 0.9)
        print('resized pic size', new_width, new_height)
        pic2show = QtGui.QPixmap(G_save_path).scaled(new_width, new_height, Qt.KeepAspectRatioByExpanding,
                                                     Qt.SmoothTransformation)
        self.label_7.setPixmap(pic2show)
    # abandon # original way to show show_synthesis_path
    # def show_synthesis_path(self):
    #     df = self.derivative_list
    #     index = self.tableWidget_dta_out.currentRow()
    #     current_smiles = self.tableWidget_dta_out.item(index, 0).text()
    #     print(f'get_synthesis_path for current_SMILES:{current_smiles}')
    #     # 获取当前分子的合成路径
    #     search_line = df[df['derivant'].isin([current_smiles])].to_dict('records')
    #     smi_lst = []
    #     smi_lst.append(current_smiles)
    #     while len(search_line) != 0:  # 有对应前体。读取前体
    #         search_result = search_line[0]['precursor']
    #         smi_lst.insert(0, search_result)
    #         get_smiles = search_result
    #         search_line = df[df['derivant'].isin([get_smiles])].to_dict('records')
    #     # 生成分子图，并生成路径图
    #     G = Digraph('G', filename='temp/return_synthesis_path')
    #     G.attr('node', shape='box')
    #     G.format = 'png'
    #     G_save_path = 'temp/return_synthesis_path.png'
    #
    #     current_path = sys.path[0]
    #     for i in range(len(smi_lst)):
    #         print(f'synthesis_path:{i + 1}/{len(smi_lst)}', '\n', smi_lst[i])
    #         try:
    #             mol = Chem.MolFromSmiles(smi_lst[i])
    #         except:
    #             print('invalid molecular SMILES string')
    #         else:
    #             save_path = f'{current_path}/temp/smi{i}-mol-graph.png'
    #             Draw.MolToFile(mol, save_path, size=(400, 400))
    #             ## G.node 需要读取绝对路径，不然可能报错。
    #             G.node(name=smi_lst[i], image=save_path, label='', labelloc='top')  # label=smi_lst[i] f'mol{i + 1}'
    #             if i >= 1:
    #                 G.edge(smi_lst[i - 1], smi_lst[i])  # 可加label # , label=f"no-level-expansion"
    #     G.render()
    #     load_image = QPixmap(G_save_path)
    #     width = load_image.width()
    #     height = load_image.height()  ##获取图片高度
    #     print('original pic size', width, height)
    #     print('label size', self.label_7.width(), self.label_7.height())
    #     ratio = (self.label_7.width() / width)
    #     new_width = int(width * ratio * 0.9)  ##定义新图片的宽和高
    #     new_height = int(height * ratio * 0.9)
    #     print('resized pic size', new_width, new_height)
    #     pic2show = QtGui.QPixmap(G_save_path).scaled(new_width, new_height, Qt.KeepAspectRatioByExpanding,
    #                                                  Qt.SmoothTransformation)
    #     self.label_7.setPixmap(pic2show)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ui = NPRetro_App()
    ui.show()
    sys.exit(app.exec_())