from scipy.fft import fft
import scipy.signal as sig
from scipy import interpolate
from scipy import signal
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d  
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSlider, QVBoxLayout, QGraphicsScene ,QLabel , QHBoxLayout ,QComboBox ,QGroupBox, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot
import matplotlib as plt
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, uic
#from pydub import AudioSegment
from PyQt5.Qt import Qt
#import vlc
import os
import sys
from scipy.io import wavfile
plt.use('Qt5Agg')
from music21 import *
from music21.stream import Stream




class CreateSlider:
    def __init__(self , index ):
        # Create a slider
        
        self.index= index
        faro7a = MainWindow()
        self.range = faro7a.get_maloka(index)
        self.slider = QSlider()
        #sets the orientation of the slider to be vertical.
        self.slider.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)

    def get_slider(self):
        return self.slider

class Modes:
    def __init__(self, name, labels, ranges, no_sliders):
        self.name = name
        self.labels = labels
        self.ranges = ranges #(for each slider whats the range of freq. )
        self.sliders = no_sliders
        self.slider_values = [[0, 10, 1]]*len(list(labels))


class SmoothingWindow:
    def __init__(self, window_type, parameters=None):
        self.window_type = window_type
        self.parameters = parameters

    def apply(self, signal):
        if self.window_type == "Rectangle":
            return self.apply_rectangle(signal)
        elif self.window_type == "Hamming":
            return self.apply_hamming(signal)
        elif self.window_type == "Hanning":
            return self.apply_hanning(signal)
        elif self.window_type == "Gaussian":
            if self.parameters is not None:
                return self.apply_gaussian(signal, self.parameters)
            else:
                raise ValueError("Gaussian window requires parameters.")

    def apply_rectangle(self, signal):
        # Rectangle window does not modify the signal
        return signal

    def apply_hamming(self, signal):
        # Apply the Hamming window to the signal
        window = np.hamming(len(signal))
        smoothed_signal = signal * window
        return smoothed_signal

    def apply_hanning(self, signal):
        # Apply the Hanning window to the signal
        window = np.hanning(len(signal))
        smoothed_signal = signal * window
        return smoothed_signal

    def apply_gaussian(self, signal, sigma):
        # Apply the Gaussian window to the signal with a specified standard deviation (sigma)
        window = np.exp(-(np.arange(len(signal)) ** 2) / (2 * sigma ** 2))
        smoothed_signal = signal * window
        return smoothed_signal



class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(r'task3.ui', self)
        self.dictnoary_values = {0: [0, 1000],
                                        1: [1000, 2000],
                                        2: [3000, 4000],
                                        3: [4000, 5000],
                                        4: [5000, 6000],
                                        5: [6000, 7000],
                                        6: [7000, 8000],
                                        7: [8000, 9000],
                                        8: [9000, 10000],
                                        9: [10000 ,11000]
                                        }
        self.selected_mode = None
        self.selected_window = None
        self.frame_layout = QHBoxLayout(self.sliders_frame)
        # Connect the activated signal to a custom slot
        self.modes_combobox.activated.connect(lambda: self.modes_combobox_activated())
        self.smoothing_window_combobox.activated.connect(lambda: self.smoothing_window_combobox_activated())
    
    def get_maloka(self , index):
        return self.dictnoary_values[index]
        
#YOUR CODE HERE 
# بسم الله الرحمن الرحييم
    def set_slider_range(self, selected_text):
        if selected_text == 0:
                    self.dictnoary_values = {0: [0, 1000],
                                        1: [1000, 2000],
                                        2: [3000, 4000],
                                        3: [4000, 5000],
                                        4: [5000, 6000],
                                        5: [6000, 7000],
                                        6: [7000, 8000],
                                        7: [8000, 9000],
                                        8: [9000, 10000],
                                        9: [10000 ,11000]
                                        }
                    values_slider = [[0, 10, 1]]*len(list(self.dictnoary_values.keys()))

        elif selected_text == 'Animals':
            self.dictnoary_values = {"cat": [1900, 5000],
                                "dog": [1500, 3000],
                                "owl": [500, 2000],
                                "lion": [490, 2800]
                                }
            values_slider = [[0, 10, 1]]*len(list(self.dictnoary_values.keys()))

        elif selected_text == 'Music Instrument':
            self.dictnoary_values = {"Drum ": [0, 500],
                                "Flute": [500, 1000],
                                "Key": [1000, 2000],
                                "Piano": [2000, 5000]
                                }
            values_slider = [[0, 10, 1]]*len(list(self.dictnoary_values.keys()))

        elif selected_text == 'ECG':
            self.dictnoary_values = {"Arithmia_1 ": [0, 500],
                                "Arithmia_2": [500, 1000],
                                "Arithmia_3": [1000, 2000]
                                }
            values_slider = [[0, 10, 1]]*len(list(self.dictnoary_values.keys()))

    # def open(self):
    #         self.fname = QFileDialog.getOpenFileName(
    #             None, "Select a file...", os.getenv('HOME'), filter="All files (*)")
    #         path = self.fname[0]
    #         if '.mp3' in path:
    #             song = AudioSegment.from_mp3(path)
    #             song.export(r"./final.wav", format="wav")
    #             self.f_rate, self.yData = wavfile.read(r"./final.wav")
    #         else:
    #             self.f_rate, self.yData = wavfile.read(path)
    #         if  len(self.yData.shape) > 1:
    #             self.yData = self.yData[:,0]

    #         self.yData = self.yData / 2.0**15
    #         self.yAxisData = self.yData
    #         self.SIZE = len(self.yAxisData)
    #         self.xAxisData = np.linspace(
    #             0, self.SIZE / self.f_rate, num=self.SIZE)
    #         self.fourier()
    #         self.p = vlc.MediaPlayer(path)
    #         self.plot()
    #         self.play()

    def modes_combobox_activated(self):
        # Get the selected item's text and display it in the label
        selected_text = self.modes_combobox.currentIndex()
        # store the mode in a global variable 
        self.selected_mode = selected_text 
        self.set_slider_range(selected_text)
        self.add_slider(selected_text)

    def smoothing_window_combobox_activated(self):
        selected_item = self.smoothing_window_combobox.currentText()
        self.selected_window = selected_item
        if selected_item == 'Gaussian' :
            self.lineEdit_2.setVisible(True)
        else:
            self.lineEdit_2.setVisible(False)


        
    def clear_layout(self ,layout):
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater() 

    def add_slider(self , selected_text):          
        if selected_text == 0: #uniform range
            self.clear_layout(self.frame_layout)
            for i in range(10):

                slider_creator = CreateSlider(i)
                #print(slider_creator.range)
                slider = slider_creator.get_slider()
                self.frame_layout.addWidget(slider)  
        else:
            self.clear_layout(self.frame_layout) 
            for i in range(4): # either musical, animal or ecg
                slider_creator = CreateSlider(i)
                slider = slider_creator.get_slider()
                self.frame_layout.addWidget(slider)
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
