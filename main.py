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
import librosa
from pydub import AudioSegment
from pydub.playback import play

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

class Signal:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.time = []
        self.sample_rate = None
        self.Data_fft = None
        self.Freq_splits = None
        self.Amp_splits = None

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
        self.frame_layout = QHBoxLayout(self.sliders_frame)
        # Connect the signal to set_combobox
        self.modes_combobox
        # Connect the activated signal to a custom slot
        self.modes_combobox.activated.connect(lambda: self.combobox_activated())
        self.smoothing_window_combobox.activated.connect(lambda: self.smoothing_window_combobox_activated())
        self.lineEdit_2.setVisible(False)  # Initially hide the line edit for Gaussian window
        self.current_signal=None
        # self.audio_data = np.array([])
        # self.plot_audio()
        self.load_btn.clicked.connect(lambda: self.load())
        self.apply_btn.clicked.connect(lambda: self.apply_smoothing())

    def get_maloka(self , index):
        return self.dictnoary_values[index]
        
#OUR CODE HERE 
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
    def load(self):
        path_info = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select a signal...",os.getenv('HOME'), filter="Raw Data (*.csv *.wav *.mp3)")
        path = path_info[0]
        data = []
        time = []
        sample_rate = 0
        signal_name = path.split('/')[-1].split('.')[0]
        type = path.split('.')[-1]
        if type in ["wav", "mp3"]:
            data, sample_rate = librosa.load(path)
            Duration = librosa.get_duration(y=data, sr=sample_rate)
            time = np.linspace(0, Duration, len(data))
            self.audio_data = data
        elif type == "csv":
            data_of_signal = pd.read_csv(path)  
            time = data_of_signal.values[:, 0]
            data = data_of_signal.values[:, 1]
        signal = Signal(signal_name)
        signal.data = data
        signal.time = time
        signal.sample_rate = sample_rate
        T = 1 / signal.sample_rate
        x_data, y_data = self.get_Fourier(signal, T, len(signal.data))
        signal.Data_fft = np.array([x_data, y_data])
        self.current_signal = signal
        self.Data_spliting(signal)
        self.Plot(signal)

    def get_Fourier(self, signal, T, N):
            f_No = np.linspace(0.0, 1.0/(2.0*T), N//2) 
            freq_values = np.fft.fft(signal.data, N)  
            freq_values = (2/N) * np.abs(freq_values[:N//2])
            return f_No, freq_values
    
    def Data_spliting(self, signal):
        slices = 10 if self.modes_combobox.currentIndex() == 0 else 4
        x_data = signal.Data_fft[0]
        y_data = signal.Data_fft[1]
        x_data_slices = np.array_split(x_data, slices)
        y_data_slices = np.array_split(y_data, slices)
        min_slice = min(len(slice) for slice in x_data_slices)
        x_data_slices = [slice[:min_slice] for slice in x_data_slices]
        y_data_slices = [slice[:min_slice] for slice in y_data_slices]
        # Convert to 2D arrays
        signal.Freq_splits = np.array(x_data_slices).T 
        # to change from rows to columns get transpose
        signal.Amp_splits = np.array(y_data_slices).T

    def Plot(self, signal):
            if signal:
                self.frequancy_graph.clear()
                self.frequancy_graph.setLabel('left', "Amplitude")
                self.frequancy_graph.setLabel('bottom', "Frequency")
                # Accumulate data for all columns
                x_values, y_values = [], []
                for i in range(signal.Freq_splits.shape[1]):
                    x_values.extend(signal.Freq_splits[:, i])
                    y_values.extend(signal.Amp_splits[:, i])
                # Plot all columns together
                plot_item = self.frequancy_graph.plot(
                    x_values, y_values, name=f"{signal.name}")
                #add legend to the graph 
                if self.frequancy_graph.plotItem.legend is not None:
                    self.frequancy_graph.plotItem.legend.clear()
                legend = self.frequancy_graph.addLegend()
                legend.addItem(plot_item, name=f"{signal.name}")
                # Apply smoothing window if selected
                smoothing_window = self.smoothing_window_combobox.currentText()
                if smoothing_window != "None":
                    self.smooth_and_plot(signal, smoothing_window)

    def smooth_and_plot(self, signal, smoothing_window):
    # Apply the selected smoothing window to the data
        window_parameters = None
        if smoothing_window == "Gaussian":
            sigma = float(self.lineEdit_2.text())
            window_parameters = {"sigma": sigma}
        smoothing_window_obj = SmoothingWindow(
            smoothing_window, window_parameters)
        smoothed_data = smoothing_window_obj.apply(signal.Amp_splits)

        if smoothed_data is not None:
            # Plot the smoothed data
            x_values_smooth, y_values_smooth = [], []
            for i in range(smoothed_data.shape[1]):
                x_values_smooth.extend(signal.Freq_splits[:, i])
                y_values_smooth.extend(smoothed_data[:, i])
            plot_item_smooth = self.frequancy_graph.plot(
                x_values_smooth, y_values_smooth, pen='r', name=f"{signal.name} (Smoothed)")
            # Add legend for the smoothed plot
            legend_smooth = self.frequancy_graph.addLegend()
            legend_smooth.addItem(
                plot_item_smooth, name=f"{signal.name} (Smoothed)")
        else:
            print("Error: Smoothing operation failed.")
        
    def apply_smoothing(self):
        if self.current_signal:
            smoothing_window = self.smoothing_window_combobox.currentText()
            if smoothing_window != "None":
                self.smooth_and_plot(self.current_signal, smoothing_window)
    
    def combobox_activated(self):
        # Get the selected item's text and display it in the label
        selected_text = self.modes_combobox.currentIndex()
        # store the mode in a global variable 
        self.selected_mode = selected_text 
        self.set_slider_range(selected_text)
        self.add_slider(selected_text)

    def smoothing_window_combobox_activated(self):
        selected_item = self.smoothing_window_combobox.currentText()
        self.selected_window = selected_item
        # Show or hide the line edit based on the selected smoothing window
        self.lineEdit_2.setVisible(selected_item == 'Gaussian')
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
