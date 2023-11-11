from scipy.fft import fft
import scipy.signal as sig
from scipy import interpolate
from scipy import signal
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os
import time
from scipy.interpolate import interp1d
import pyqtgraph as pg
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QSlider, QVBoxLayout, QGraphicsScene ,QLabel , QHBoxLayout ,QComboBox ,QGroupBox 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from cmath import*
from numpy import *
import sys
import matplotlib
matplotlib.use('Qt5Agg')

class CreateSlider:
    def __init__(self):
        # Create a slider
        self.slider = QSlider()
        #sets the orientation of the slider to be vertical.
        self.slider.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)

    def get_slider(self):
        return self.slider

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(r'task3.ui', self)
        self.selected_mode = None
        self.frame_layout = QHBoxLayout(self.sliders_frame)
        # Connect the signal to set_combobox
        self.set_combobox()
        # Connect the activated signal to a custom slot
        self.modes_combobox.activated.connect(lambda: self.combobox_activated())
        
#YOUR CODE HERE 
# بسم الله الرحمن الرحييم
    def set_combobox(self):
        self.modes_combobox.addItem('Uniform Range')
        self.modes_combobox.addItem('Musical Instruments')
        self.modes_combobox.addItem('Animal Sounds')
        self.modes_combobox.addItem('ECG Abnormalities')

    def combobox_activated(self):
        # Get the selected item's text and display it in the label
        selected_text = self.modes_combobox.currentIndex()
        # store the mode in a global variable 
        self.selected_mode = selected_text 
        print(selected_text)
        self.add_slider(selected_text)
        
    def clear_layout(self ,layout):
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater() 

    def add_slider(self , selected_text):          
        if selected_text == 0:
            self.clear_layout(self.frame_layout)
            for _ in range(10):
                slider_creator = CreateSlider()
                slider = slider_creator.get_slider()
                self.frame_layout.addWidget(slider)  
        else:
            self.clear_layout(self.frame_layout) 
            for _ in range(4):
                slider_creator = CreateSlider()
                slider = slider_creator.get_slider()
                self.frame_layout.addWidget(slider)
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
