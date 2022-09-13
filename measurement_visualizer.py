import json
import sys
from pathlib import PurePath
from time import sleep

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRunnable, QThreadPool, QObject, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QTreeWidgetItem, QLabel, QProgressBar
from numpy.fft import fftfreq
from ui.freq_response_ui import *
from freq_response_from_measurement import draw
from matplotlib import pyplot as plt


class MyThread(QThread):
    change_value = pyqtSignal(int)

    def run(self):
        cnt = 0
        while True:
            cnt += 1
            if cnt > 100:
                cnt = 0
            sleep(0.01)
            self.change_value.emit(cnt)


class WorkerSignals(QObject):
    finished = pyqtSignal(object, object)
    status = pyqtSignal(object)
    progress = pyqtSignal(object)
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class DrawWorker(QRunnable):
    def __init__(self, fn, *args):
        super(DrawWorker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        data, figure_props = self.fn(*self.args, self.signals)
        self.signals.finished.emit(data, figure_props)


class MyMainWindow(QMainWindow, Ui_MainWindow):
    BASE, TARGET = range(2)
    VERSION = '1.3'
    DATE = '20210319'

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.toolButtonBase.clicked.connect(lambda: self.on_toolButtonClicked_OpenFile(self.BASE))
        self.toolButtonTarget.clicked.connect(lambda: self.on_toolButtonClicked_OpenFile(self.TARGET))
        self.pushButtonAddMeasurement.clicked.connect(self.on_pushButtonClicked_AddMeasurement)
        self.pushButtonAddTarget.clicked.connect(self.on_pushButtonClicked_AddTarget)
        self.pushButtonDraw.clicked.connect(self.on_pushButtonClicked_Draw)
        self.listWidgetTargetList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listWidgetTargetList.customContextMenuRequested.connect(self.on_listWidgetCustomContextMenuRequested_TargetList)
        self.treeWidgetMeasurments.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWidgetMeasurments.customContextMenuRequested.connect(self.on_treeWidgetCustomContextMenuRequested_Measurements)
        self.treeWidgetRoot = QTreeWidgetItem(['measurements'])
        self.treeWidgetMeasurments.setColumnCount(1)
        self.treeWidgetMeasurments.addTopLevelItem(self.treeWidgetRoot)
        self.comboBoxType.currentIndexChanged.connect(self.on_comboBoxCurrentIndexChanged_Type)
        self.pushButtonClearTargets.clicked.connect(self.on_pushButtonClicked_ClearTargets)
        self.pushButtonClearMeasurements.clicked.connect(self.on_PushButtonClicked_ClearMeasurements)
        self.spinBoxTarget.valueChanged.connect(self.on_spinBoxValueChanged_Target)

        self.on_comboBoxCurrentIndexChanged_Type(self.comboBoxType.currentIndex())

        self.versionLabel = QLabel(f"Version: {self.VERSION:s}")
        self.relDateLabel = QLabel(f"Release Date: {self.DATE:s}")
        self.infoLabel = QLabel(f"")
        self.infoLabel.setVisible(False)
        self.progressBarDraw2 = QProgressBar()
        self.progressBarDraw2.setValue(0)
        self.progressBarDraw2.setMaximum(100)
        self.progressBarDraw2.setMinimum(0)
        self.progressBarDraw2.setVisible(False)
        self.progressBarDraw2.setMaximumWidth(100)
        self.progressBarDraw2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.statusbar.addWidget(self.progressBarDraw2)
        self.statusbar.addWidget(self.infoLabel)
        self.statusbar.addPermanentWidget(self.versionLabel)
        self.statusbar.addPermanentWidget(self.relDateLabel)

        #self.progressBarTh = MyThread()
        self.threadPool = QThreadPool()

    def on_toolButtonClicked_OpenFile(self, n):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose recording data", "",
                                                  "wav Files (*.wav);;All Files (*)", options=options)
        if n == self.BASE:
            self.lineEditBase.setText(fileName)
        elif n == self.TARGET:
            self.lineEditTarget.setText(fileName)
            if self.comboBoxType.currentIndex() != 2:
                self.lineEditTargetLabel.setText(f'{PurePath(fileName).stem:s} CH{self.spinBoxTarget.value():d}')
            else:
                self.lineEditTargetLabel.setText(f'{PurePath(fileName).stem:s}')

    def on_pushButtonClicked_AddTarget(self):
        file = self.lineEditTarget.text()
        if len(file) == 0:
            QMessageBox.warning(self, 'Warning', 'Please specify a target wave file.', QMessageBox.Ok)
            return
        label = self.lineEditTargetLabel.text()
        ch = self.spinBoxTarget.value()
        boost = self.doubleSpinBoxBoost.value()
        self.listWidgetTargetList.addItem(f'{file:s},{label:s},{ch:d},{boost:f}')

    def on_pushButtonClicked_AddMeasurement(self):
        base_file = self.lineEditBase.text()
        if self.comboBoxType.currentIndex() == 0:
            if len(base_file) == 0:
                QMessageBox.warning(self, 'Warning', 'Please specify a base wave file.', QMessageBox.Ok)
                return
            elif self.listWidgetTargetList.count() == 0:
                QMessageBox.warning(self, 'Warning', 'Please specify wave targets.', QMessageBox.Ok)
                return

        child_type = QTreeWidgetItem([self.comboBoxType.currentText()])

        if self.comboBoxType.currentIndex() == 0:
            child_base = QTreeWidgetItem(['base'])
            child_base_content = QTreeWidgetItem([f'{base_file:s},{self.spinBoxBase.value():d}'])
            child_base.addChild(child_base_content)
            child_type.addChild(child_base)

        child_target = QTreeWidgetItem(['target'])
        for it in range(self.listWidgetTargetList.count()):
            child_target_content = QTreeWidgetItem([self.listWidgetTargetList.item(it).text()])
            child_target.addChild(child_target_content)
        child_type.addChild(child_target)

        self.treeWidgetRoot.addChild(child_type)
        self.treeWidgetRoot.setExpanded(True)

    def on_pushButtonClicked_Draw(self):
        measurements = []
        root = self.treeWidgetRoot
        for i in range(root.childCount()):
            child = root.child(i)
            child_text = child.text(0)

            if child_text == 'Normalization':
                normalization = {'type': 'ANC_NORM'}
                for j in range(child.childCount()):
                    grand_child = child.child(j)
                    grand_child_text = grand_child.text(0)
                    if grand_child_text == 'base':
                        grand_grand_child_text = grand_child.child(0).text(0)
                        token = grand_grand_child_text.strip().split(',')
                        base = {'file': token[0], 'ch': int(token[1])}
                        normalization['base'] = base
                    elif grand_child_text == 'target':
                        anc = []
                        for k in range(grand_child.childCount()):
                            grand_grand_child_text = grand_child.child(k).text(0)
                            token = grand_grand_child_text.strip().split(',')
                            target = {'file': token[0], 'legend': token[1], 'ch': int(token[2]), 'boost': float(token[3])}
                            anc.append(target)

                        normalization['anc'] = anc

                measurements.append(normalization)

            elif child_text == 'Frequency Response':
                freq_resp = {'type': 'FREQ_RESP'}
                grand_child = child.child(0)
                grand_child_text = grand_child.text(0)
                if grand_child_text == 'target':
                    recording = []
                    for k in range(grand_child.childCount()):
                        grand_grand_child_text = grand_child.child(k).text(0)
                        token = grand_grand_child_text.strip().split(',')
                        target = {'file': token[0], 'legend': token[1], 'ch': int(token[2]), 'boost': float(token[3])}
                        recording.append(target)

                    freq_resp['recording'] = recording

                measurements.append(freq_resp)

            elif child_text == 'NLMS':
                nlms = {'type': 'NLMS'}
                grand_child = child.child(0)
                grand_child_text = grand_child.text(0)
                if grand_child_text == 'target':
                    recording = []
                    for k in range(grand_child.childCount()):
                        grand_grand_child_text = grand_child.child(k).text(0)
                        token = grand_grand_child_text.strip().split(',')
                        target = {'file': token[0], 'legend': token[1]}
                        recording.append(target)

                    nlms['recording'] = recording

                measurements.append(nlms)

            elif child_text == 'NLMS Residual':
                nlms_res = {'type': 'NLMS_RES'}
                grand_child = child.child(0)
                grand_child_text = grand_child.text(0)
                if grand_child_text == 'target':
                    recording = []
                    for k in range(grand_child.childCount()):
                        grand_grand_child_text = grand_child.child(k).text(0)
                        token = grand_grand_child_text.strip().split(',')
                        target = {'file': token[0], 'legend': token[1]}
                        recording.append(target)

                    nlms_res['recording'] = recording

                measurements.append(nlms_res)

        if len(measurements) == 0:
            QMessageBox.warning(self, 'Warning', 'No measurement data to be showed.', QMessageBox.Ok)
            return

        figure_properties = {
            'title': self.lineEditFigTitle.text(),
            'x_label': self.lineEditXCaption.text(),
            'y_label': self.lineEditYCaption.text(),
            'x_axis': [self.spinBoxXLimitMin.value(), self.spinBoxXLimitMax.value()],
            'y_axis': [self.spinBoxYLimitMin.value(), self.spinBoxYLimitMax.value()],
            'y_ticks': [self.spinBoxYTicksMin.value(), self.spinBoxYTicksMax.value(), self.spinBoxYTicksStep.value()],
            'bit_depth': self.comboBoxBitDepth.currentText(),
            'legend_size': self.spinBoxLabelFontSize.value()
        }

        config = {'figure_properties': figure_properties, 'measurements': measurements}
        print(config)

        with open('temp.json', 'w') as of:
            json.dump(config, of, indent=4)

        worker = DrawWorker(draw, config)
        worker.setAutoDelete(True)
        worker.signals.finished.connect(self.drawPlot)
        worker.signals.status.connect(self.setProgressBarText)
        worker.signals.progress.connect(self.setProgressVal)
        self.threadPool.start(worker)
        self.startProgressBar()

    def on_pushButtonClicked_ClearTargets(self):
        self.listWidgetTargetList.clear()

    def on_PushButtonClicked_ClearMeasurements(self):
        self.treeWidgetMeasurments.clear()
        self.treeWidgetRoot = QTreeWidgetItem(['measurements'])
        self.treeWidgetMeasurments.setColumnCount(1)
        self.treeWidgetMeasurments.addTopLevelItem(self.treeWidgetRoot)

    def on_comboBoxCurrentIndexChanged_Type(self, index):
        if index == 0:
            self.frameBase.setEnabled(True)
            self.spinBoxTarget.setEnabled(True)
            self.doubleSpinBoxBoost.setEnabled(False)
        elif index == 1:
            self.frameBase.setEnabled(False)
            self.spinBoxTarget.setEnabled(True)
            self.doubleSpinBoxBoost.setEnabled(True)
        elif index == 2 or index == 3:
            self.frameBase.setEnabled(False)
            self.spinBoxTarget.setEnabled(False)
            self.doubleSpinBoxBoost.setEnabled(False)

        self.listWidgetTargetList.clear()

    def on_listWidgetCustomContextMenuRequested_TargetList(self, pos):
        it = self.listWidgetTargetList.itemAt(pos)
        if it is None: return
        row = self.listWidgetTargetList.row(it)
        menu = QtWidgets.QMenu()
        delete_action = menu.addAction('Delete')
        action = menu.exec_(self.listWidgetTargetList.viewport().mapToGlobal(pos))
        if action == delete_action:
            item = self.listWidgetTargetList.takeItem(row)
            del item

    def on_treeWidgetCustomContextMenuRequested_Measurements(self, pos):
        it = self.treeWidgetMeasurments.itemAt(pos)
        if it is None: return
        elif it.text(0) == "measurements": return

        menu = QtWidgets.QMenu()
        delete_action = menu.addAction('Delete')
        action = menu.exec_(self.treeWidgetMeasurments.viewport().mapToGlobal(pos))
        if action == delete_action:
            parent = it.parent()
            if parent.text(0) == 'base' or parent.text(0) == 'target':
                grand_parent = parent.parent()
                if grand_parent.text(0) == 'Normalization' or grand_parent.text(0) == 'Frequency Response' \
                        or grand_parent.text(0) == 'NLMS' or grand_parent.text(0) == 'NLMS Residual':
                    self.treeWidgetRoot.removeChild(grand_parent)
            elif parent.text(0) == 'Normalization' or parent.text(0) == 'Frequency Response' \
                    or parent.text(0) == 'NLMS' or parent.text(0) == 'NLMS Residual':
                self.treeWidgetRoot.removeChild(parent)
            elif it.text(0) == 'Normalization' or it.text(0) == 'Frequency Response' \
                    or it.text(0) == 'NLMS' or it.text(0) == 'NLMS Residual':
                self.treeWidgetRoot.removeChild(it)

    def on_spinBoxValueChanged_Target(self, int):
        self.lineEditTargetLabel.setText(f'{PurePath(self.lineEditTarget.text()).stem:s} CH{self.spinBoxTarget.value():d}')

    def startProgressBar(self):
        #self.progressBarTh.change_value.connect(self.setProgressVal)
        #self.progressBarTh.start()
        self.infoLabel.setVisible(True)
        self.progressBarDraw2.setVisible(True)
        self.progressBarDraw2.setFormat('%p%')
        self.pushButtonDraw.setEnabled(False)

    def stopProgressBar(self):
        self.setProgressVal(100)
        self.infoLabel.setVisible(False)
        self.progressBarDraw2.setVisible(False)
        #self.progressBarDraw2.setFormat('Done...')
        self.pushButtonDraw.setEnabled(True)
        #self.progressBarTh.terminate()

    def setProgressVal(self, val):
        self.progressBarDraw2.setValue(val)

    def setProgressBarText(self, text):
        self.infoLabel.setText(text)

    def drawPlot(self, data, figure_props):
        self.stopProgressBar()

        subplot_max = 0
        if len(data[0]):
            subplot_max += 1
        if len(data[2]):
            subplot_max += 2
        if len(data[5]):
            subplot_max += int(len(data[5]) / 2)

        subplot_used = 1
        print(f"subplot_max:{subplot_max:d}, subplot_used:{subplot_used:d}")

        axis = figure_props['x_axis']
        axis.extend(figure_props['y_axis'])
        # Draw
        lines = []
        if len(data[0]):
            ax = plt.subplot(subplot_max, 1, subplot_used)
            for c in data[0]:
                total_samples = c['data'].shape[0]
                limit = int((total_samples / 2) - 1)
                freqs = fftfreq(c['data'].shape[0], 1 / c['fs'])
                line, = ax.semilogx(freqs[:limit], c['data'][:limit], linewidth=1)
                lines.append(line)
            ax.legend(lines, data[1], fontsize=figure_props['legend_size'])
            ax.set_title(figure_props['title'], size=10)
            ax.set_xlabel(figure_props['x_label'], fontsize=8)
            ax.set_ylabel(figure_props['y_label'], fontsize=8)
            ax.axis(axis)
            ax.set_yticks(range(*figure_props['y_ticks']))
            # ax.set_xticks(fontsize=8)
            ax.grid(True, which='both')
            subplot_used += 1

        # Draw NLMS
        if len(data[2]):
            ax = plt.subplot(subplot_max, 1, subplot_used)
            ss = range(0, 50)
            lines = []
            for impulse in data[2]:
                line, = ax.plot(ss, impulse[0:50], linewidth=1)
                lines.append(line)
            ax.legend(lines, data[4], fontsize=figure_props['legend_size'])
            ax.set_title('Impulse Response (NLMS)', size=10)
            ax.set_yticks(np.arange(-3, 3, step=0.5))
            ax.axis([0, 50, -3, 3])
            ax.grid(True, which='both')
            subplot_used += 1

        if len(data[3]):
            ax = plt.subplot(subplot_max, 1, subplot_used)
            s = range(0, 24000)
            lines = ax.semilogx(s, np.transpose(data[3]), linewidth=1)
            ax.legend(lines, data[4], fontsize=figure_props['legend_size'])
            ax.set_title(f"{figure_props['title']:s} (NLMS)", size=10)
            ax.set_xlabel(figure_props['x_label'], fontsize=8)
            ax.set_ylabel(figure_props['y_label'], fontsize=8)
            ax.set_yticks(range(*figure_props['y_ticks']))
            ax.axis(axis)
            ax.grid(True, which='both')
            subplot_used += 1

        if len(data[5]):
            for i in range(0, len(data[5]), 2):
                ax = plt.subplot(subplot_max, 1, subplot_used)
                s = range(0, 24000)
                lines = ax.semilogx(s, np.transpose(data[5][i]), np.transpose(data[5][i + 1]), linewidth=1)
                ax.legend(lines, [data[6][i], data[6][i + 1]], fontsize=figure_props['legend_size'])
                ax.set_title(f"{figure_props['title']:s} (5e Residual Measurement)", size=10)
                ax.set_xlabel(figure_props['x_label'], fontsize=8)
                ax.set_ylabel(figure_props['y_label'], fontsize=8)
                ax.set_yticks(range(*figure_props['y_ticks']))
                ax.axis(axis)
                ax.grid(True, which='both')
                subplot_used += 1

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())