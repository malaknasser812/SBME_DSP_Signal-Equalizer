
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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene ,QLabel , QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from cmath import*
from numpy import *
import sys
import matplotlib

matplotlib.use('Qt5Agg')

class Modes:
    def __init__(self, name, labels, ranges, no_sliders):
        self.name = name
        self.labels = labels
        self.ranges = ranges #(for each slider whats the range of freq. )
        self.sliders = no_sliders
        self.slider_values = [[0, 10, 1]]*len(list(labels))

class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.modes = [Modes("Frequency", ['Label 1', 'Label 2'], [(0, 100), (100, 200)], []),
                    Modes("Animals", ['Label A', 'Label E'], [(0, 50), (50, 100)], []),
                    Modes("Music Instrument", ['Drum', 'Flute'], [(0, 100), (100, 200)], []),
                    Modes("Medical", ['Arrhythmia 1', 'Arrhythmia 2'], [(0, 50), (50, 100)], [])]

        # Load the UI Page
        uic.loadUi(r'task3.ui', self)


#YOUR CODE HERE 
# بسم الله الرحمن الرحييم


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
