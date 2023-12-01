from scipy.fft import fft
import numpy as np
import pandas as pd
import copy
from PyQt5.QtWidgets import QSlider,QHBoxLayout 
import matplotlib as plt
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, uic    
from pyqtgraph import ImageItem
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
from PySide6.QtMultimedia import QMediaPlayer
import os
import sys
plt.use('Qt5Agg')
import librosa
import bisect
import pyqtgraph as pg
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from scipy import signal as sg



class Signal:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.time = []
        self.sample_rate = None
        self.freq_data = None
        self.Ranges = []
        self.phase = None

class SmoothingWindow:
    def __init__(self, window_type, amp,sigma=None):
        self.window_type = window_type
        self.sigma = sigma
        self.amp = amp
    def apply(self, signal):
        if self.window_type == "Rectangular":
            window = sg.windows.boxcar(len(signal))
            smoothed_signal = self.amp * window 
            smoothed_signal = self.amp * window
            #print("smoothed_signal")
            #print(smoothed_signal)
            return smoothed_signal
        elif self.window_type == "Hamming":
            window = sg.windows.hamming(len(signal))
            smoothed_signal = self.amp * window
            return smoothed_signal
        elif self.window_type == "Hanning":
            window = sg.windows.hann(len(signal))
            smoothed_signal = self.amp * window
            return smoothed_signal
        elif self.window_type == "Gaussian":
            if self.sigma is not None:
            # Apply the Gaussian window to the signal with a specified standard deviation (sigma)
                window = sg.windows.gaussian(len(signal),self.sigma)
                smoothed_signal = self.amp * window
                return smoothed_signal
            else:
                raise ValueError("Gaussian window requires parameters.")

class CreateSlider:
    def __init__(self , index ):
        # Create a slider
        self.index= index
        self.slider = QSlider()
        #sets the orientation of the slider to be vertical.
        self.slider.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(20)
        self.slider.setValue(10)
        # self.sliderlabel.setText()
    def get_slider(self):
        return self.slider
    
class EqualizerApp(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(EqualizerApp, self).__init__(*args, **kwargs)
        # Load the UI Page
        uic.loadUi(r'task3.ui', self)
        # self.dictnoary_values = {}
        self.selected_mode = 'Uniform Range'
        self.selected_window = None
        self.frame_layout = QHBoxLayout(self.sliders_frame)
        self.current_signal=None
        self.player = QMediaPlayer(None,QMediaPlayer.StreamPlayback)
        self.player.setVolume(50)
        self.timer = QTimer(self)
        self.timer = QtCore.QTimer(self)
        self.elapsed_timer = QtCore.QElapsedTimer()
        self.timer.timeout.connect(self.updatepos)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.updatepos)
        self.line = pg.InfiniteLine(pos=0, angle=90, pen=None, movable=False)
        self.changed = True
        self.line_position = 0
        self.player.positionChanged.connect(self.updatepos)
        self.current_speed = 1
        self.slider_gain = {}
        self.equalized_bool = False
        self.time_eq_signal = Signal('timeeq')
        

        self.line = pg.InfiniteLine(pos=0.1, angle=90, pen=None, movable=False)
        # spectooooooo
        self.available_palettes = ['twilight', 'Blues', 'Greys', 'ocean', 'nipy_spectral']
        self.current_color_palette = self.available_palettes[0]
        self.spectrogram_widget = {
            'before': self.spectrogram_before,
            'after': self.spectrogram_after
        }
        self.sliders_list =[]
        # Ui conection
        self.modes_combobox.activated.connect(lambda: self.combobox_activated())
        self.smoothing_window_combobox.activated.connect(lambda: self.smoothing_window_combobox_activated())
        self.lineEdit_2.setVisible(False)  # Initially hide the line edit for Gaussian window
        self.load_btn.clicked.connect(lambda: self.load())
        self.hear_orig_btn.clicked.connect(self.playMusic)
        self.hear_eq_btn.clicked.connect(lambda: self.playMusic())
        self.apply_btn.clicked.connect(lambda: self.plot_freq_smoothing_window())
        self.play_pause_btn.clicked.connect(lambda: self.play_pause()) 
        self.replay_btn.clicked.connect(lambda: self.playMusic())
        self.speed_up_btn.clicked.connect(lambda: self.speed_up()) 
        self.speed_down_btn.clicked.connect(lambda: self.speed_down())  
        self.checkBox.stateChanged.connect(lambda : self.hide())

#OUR CODE HERE 
    def dict_ranges(self):
        self.selected_mode = self.modes_combobox.currentText()
        dictnoary_values ={}
        if self.selected_mode == 'Animal Sounds':
            dictnoary_values = {"cat": [400, 420],
                                "dog": [600, 700],
                                "owl": [1300, 1600],
                                "lion": [3000, 4000]
                                }
        elif self.selected_mode == 'Musical Instruments':
            dictnoary_values = {"Drum ": [0, 150],
                                "Flute": [150, 600],
                                "Key": [600, 800],
                                "Piano": [800, 1200]
                                }
        elif self.selected_mode == 'ECG Abnormalities':
            dictnoary_values = {"Arithmia_1 ": [0, 500],
                                "Arithmia_2": [500, 1000],
                                "Arithmia_3": [1000, 2000]
                                }
        return dictnoary_values

    def load(self):
        path_info = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select a signal...",os.getenv('HOME'), filter="Raw Data (*.csv *.wav *.mp3)")
        path = path_info[0]
        time = []
        sample_rate = 0
        data = []
        signal_name = path.split('/')[-1].split('.')[0]   # Extract signal name from file path
        type = path.split('.')[-1]
        # Check the file type and load data accordingly
        if type in ["wav", "mp3"]:
            data, sample_rate = librosa.load(path)
            Duration = librosa.get_duration(y=data, sr=sample_rate)
            time = np.linspace(0, Duration, len(data))
            self.audio_data = path
        elif type == "csv":
            data_of_signal = pd.read_csv(path)  
            time = data_of_signal.values[:, 0]
            data = data_of_signal.values[:, 1]
            if len(time) > 1:
                sample_rate = 1 /time[1]-time[0]
            else:
                sample_rate=1
        # Create a Signal object and set its attributes
        self.current_signal = Signal(signal_name)
        self.current_signal.data = data
        self.current_signal.time = time
        self.current_signal.sample_rate = sample_rate 
        # Calculate and set the Fourier transform of the signal
        T = 1 / self.current_signal.sample_rate
        x_data, y_data = self.get_Fourier(T, self.current_signal.data)
        self.current_signal.freq_data = [x_data, y_data]
        # self.set_slider_range()
        self.Plot("original")
        self.plot_spectrogram(data, sample_rate, 'before')
        # Determine frequency ranges based on the selected mode
        self.Range_spliting()
        # Plot the frequency smoothing window
        self.plot_freq_smoothing_window()

    def get_Fourier(self, T, data):
        N=len(data)
        freq_amp= np.fft.fft(data)
        # Calculate the corresponding frequencies
        Freq= np.fft.fftfreq(N, T)[:N//2]
        # Extracting positive frequencies and scaling the amplitude
        Amp = (2/N)*(np.abs(freq_amp[:N//2]))
        self.current_signal.phase = np.angle(Amp[:N//2])
        return Freq, Amp

    def Range_spliting(self):
        dictnoary_values = self.dict_ranges()
        print (self.modes_combobox.currentText())
        if self.modes_combobox.currentText() == 'Uniform Range':
            #print("hhhhhhhhhhhhhhh")
            # Divide the frequency range into 10 equal parts for the 'Uniform Range' mode
            #print("hhhhhhhhhhhhhhh")
            batch_size = int(len(self.current_signal.freq_data[0])/10) 
            self.current_signal.Ranges = [(i*batch_size,(i+1)*batch_size) for i in range(10)] 
            print (self.current_signal.Ranges)
        else :
            freq= self.current_signal.freq_data[0] #index zero for mag of freq
            print(dictnoary_values.items())
            # Calculate frequency indices for specified ranges
            for _,(start,end) in dictnoary_values.items():
                start_ind = bisect.bisect_left(freq, start)
                end_ind = bisect.bisect_right(freq, end) - 1  # Adjusted for inclusive end index
                self.current_signal.Ranges.append((start_ind, end_ind))
                print(self.current_signal.Ranges)


    def Plot(self, graph):
            signal= self.current_signal
            if signal:
                #time domain 
                graphs = [self.original_graph, self.equalized_graph]
                graph = graphs[0] if graph == "original"  else graphs[1]
                graph.clear()
                graph.setLabel('left', "Amplitude")
                graph.setLabel('bottom', "Time")
                plot_item = graph.plot(
                    signal.time, signal.data, name=f"{signal.name}")
                # Add legend to the graph
                if graph.plotItem.legend is not None:
                    graph.plotItem.legend.clear()
                legend = graph.addLegend()
                legend.addItem(plot_item, name=f"{signal.name}")

    def plot_freq_smoothing_window (self):
        signal = self.eqsignal if self.equalized_bool  else self.current_signal
        if signal and signal.Ranges:  # Check if signal is not None and signal.Ranges is not empty
            start_last_ind, end_last_ind = signal.Ranges[-1]
            #print("helloooo")
            #frequency domain
            self.frequancy_graph.clear()
            # Plot the original frequency data in white
            self.frequancy_graph.plot(signal.freq_data[0],
                    signal.freq_data[1],pen={'color': 'w'})
            # Iterate through the frequency ranges and plot smoothed windows
            for i in range(len(signal.Ranges)):
                if i!= len(signal.Ranges)-1 :
                    start_ind,end_ind = signal.Ranges[i]
                    v_line_pos = signal.freq_data[0][start_ind]
                    # Get smoothing window parameters
                    windowtype = self.smoothing_window_combobox.currentText()
                    # Convert sigma_text to integer if not empty, otherwise set a default value
                    sigma_text = self.lineEdit_2.text()
                    if sigma_text:
                        sigma = int(sigma_text)
                    else:
                        sigma = 20  # Set a default value if the text is empty
                    amp = max(signal.freq_data[1][start_ind:end_ind])
                    # Apply the smoothing window
                    smooth_window = SmoothingWindow(windowtype,amp,sigma)
                    curr_smooth_window = smooth_window.apply(signal.freq_data[1][start_ind:end_ind])
                    # print(len(signal.freq_data[0][start_ind:end_ind]))
                    # print( len(curr_smooth_window))
                    self.frequancy_graph.plot(signal.freq_data[0][start_ind:end_ind],
                            curr_smooth_window,pen={'color': 'r', 'width': 2})
                else:
                    v_line_pos = signal.freq_data[0][end_last_ind-1]
                    v_line_pos_start = signal.freq_data[0][start_last_ind]
                    windowtype = self.smoothing_window_combobox.currentText()
                    sigma_text = self.lineEdit_2.text()
                    if sigma_text:
                        sigma = int(sigma_text)
                    else:
                        sigma = 20  # Set a default value if the text is empty
                    amp = max(signal.freq_data[1][start_ind:end_ind])
                    smooth_window = SmoothingWindow(windowtype,amp,sigma)     
                    curr_smooth_window = smooth_window.apply(signal.freq_data[1][start_ind:end_ind])
                    # print( len(signal.freq_data[0][start_ind:end_ind]))
                    # print(len(curr_smooth_window))
                    self.frequancy_graph.plot(signal.freq_data[0][start_last_ind:end_last_ind],
                            curr_smooth_window,pen={'color': 'r', 'width': 2})
                    # Add a vertical line for the starting position of the last range
                    v_line_start = pg.InfiniteLine(pos=v_line_pos_start, angle=90, movable=False, pen=pg.mkPen('r', width=2))
                    self.frequancy_graph.addItem(v_line_start)
                # Add a vertical line for the current position
                v_line = pg.InfiniteLine(pos=v_line_pos, angle=90, movable=False, pen=pg.mkPen('r', width=2))
                self.frequancy_graph.addItem(v_line)

    def plot_spectrogram(self, samples, sampling_rate, widget):
        # Clear the previous content of the spectrogram widget
        self.spectrogram_widget[widget].clear()
        # Add a subplot to the spectrogram widget
        spectrogram_axes = self.spectrogram_widget[widget].getPlotItem()
        # Convert input samples to float32
        data = samples.astype('float32')
        # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
        n_fft=1024
        # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
        hop_length=320
        # Specify the window type for FFT/STFT
        window_type ='hann'

        # Compute the short-time Fourier transform magnitude squared
        # it calculates the spectrograme for the givven data but the scale of the y axix is not good
        # it gives you a totally dark img so after that we convert it to the mel scale 
        frequency_magnitude = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type)) ** 2

        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(S=frequency_magnitude, y=data, sr=sampling_rate, n_fft=n_fft,
                    hop_length=hop_length, win_length=n_fft, window=window_type, n_mels =128)
        #because of mel scale is not redable we must calculate the db scale 
        # Convert power spectrogram to decibels
        #it is the log mel spectrogram
        decibel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Create ImageItem for displaying the spectrogram
        spectrogram_image = ImageItem(image=decibel_spectrogram)
        # Add ImageItem to the spectrogram widget
        self.spectrogram_widget[widget].addItem(spectrogram_image)
        # Add colorbar to the spectrogram plot (if needed)
        # self.spectrogram_widget[widget].getFigure().colorbar(spectrogram_image, ax=spectrogram_axes, format='%+2.0f dB')
        # Redraw the spectrogram widget
        self.spectrogram_widget[widget].draw()

    def playMusic(self):
        self.changed =  True
        # Create a QMediaContent object from the local audio file
        media = QMediaContent(QUrl.fromLocalFile(self.audio_data))
        #self.sampling_freq, _ = wavfile.read(self.audio_data)
        # Set the media content for the player and start playing
        self.player.setMedia(media)
        self.player.play()
        # Add a vertical line to the original graph
        self.original_graph.addItem (self.line)
        self.timer.start()

    def updatepos(self):
    # Get the current position in milliseconds
        position = self.player.position()/1000
        # Update the line position based on the current position
        self.line_position = position 
        max_x = self.original_graph.getViewBox().viewRange()[0][1]
        if self.line_position > max_x:
            self.line_position = max_x
        self.line_position = position
        max_x = self.original_graph.getViewBox().viewRange()[0][1]
        if self.line_position > max_x:
            self.line_position = max_x -0.052
        self.line.setPos(self.line_position)
        #print (self.line.getPos()[0], self.player.position())

    def speed_up(self):
        # Increase the playback speed
        self.current_speed = self.current_speed + 0.1  # You can adjust the increment as needed
        self.player.setPlaybackRate(self.current_speed)
        #print(self.current_speed)

    def speed_down(self):
        # Decrease the playback speed
        self.current_speed = self.current_speed - 0.1  # You can adjust the increment as needed
        new_speed = max(0.1, self.current_speed - 0.1)  # Ensure speed doesn't go below 0.1
        self.player.setPlaybackRate(new_speed)
        #print(new_speed)

    def play_pause(self):
        if self.changed:
            self.player.pause()
            self.timer.stop()
            self.changed = not self.changed
        else:
            self.player.play()
            self.timer.start()
            self.changed = not self.changed


    # def position_changed(self): 
    #     current_time = self.player.get_time()
    #     total_duration = self.player.get_length()

    #     if total_duration != 0:
    #         progress = current_time / total_duration
    #         max_index = self.original_graph.getViewBox().width()

    #         # Calculate the index based on the previous position
    #         previous_position = progress * max_index/100
    #         index = int(previous_position * self.f_rate / 1000)

    #         # Ensure the index is within the bounds of the plotted signal
    #         index = max(0, min(index, max_index))
    #         self.line.setPos(index)
            # Add vertical leines for start and end indices for each mode
            # for start_ind, end_ind in signal.Ranges:
                
            # # Add vertical line at the end for 'Uniform Range'
            # if self.modes_combobox.currentIndex() == 'Uniform Range' and signal.Ranges:
            #     _, end_ind = signal.Ranges[-1]
            #     v_line_end_uniform = pg.InfiniteLine(pos=signal.freq_data[0][end_ind], angle=90, movable=False, pen=pg.mkPen('b', width=2))
            #     self.frequancy_graph.addItem(v_line_end_uniform)


    def combobox_activated(self):
        # Get the selected item's text and display it in the label
        selected_index = self.modes_combobox.currentIndex()
        self.selected_mode = self.modes_combobox.currentText()
        # store the mode in a global variable 
        # self.set_slider_range()
        self.add_slider(selected_index)
        self.Range_spliting()

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

    def add_slider(self , selected_index):          
        if selected_index == 0: #uniform range
            self.clear_layout(self.frame_layout)
            for i in range(10):
                slider_creator = CreateSlider(i)
                #print(slider_creator.range)
                slider = slider_creator.get_slider()
                self.slider_gain[i] = 10
                slider.valueChanged.connect(lambda value, i=i: self.update_slider_value(i, value/10))
                self.frame_layout.addWidget(slider) 
                self.sliders_list.append(slider) 
        else:
            self.clear_layout(self.frame_layout) 
            for i in range(4): # either musical, animal or ecg
                slider_creator = CreateSlider(i)
                slider = slider_creator.get_slider()
                # self.slider_gain = {i:1}
                self.frame_layout.addWidget(slider)
                self.sliders_list.append(slider) 
        #print(self.sliders_list[0].value())
        #print (self.slider_gain)

    def update_slider_value(self, slider_index, value):
        # This method will be called whenever a slider is moved
        self.slider_gain[slider_index] = value
        #print (self.slider_gain)
        self.equalized(slider_index, value)
        #self.Plot('equalized')

    def equalized(self, slider_index,value):
        #print (value)
        self.equalized_bool = True
        self.eqsignal = copy.deepcopy(self.current_signal) 
        for i in range(self.current_signal.Ranges[slider_index][0],self.current_signal.Ranges[slider_index][1]):  
            #print('before',self.current_signal.freq_data[1][i])        
            self.eqsignal.freq_data[1][i] = self.current_signal.freq_data[1][i] * value
            #print('after',self.eqsignal.freq_data[1][i], self.current_signal.freq_data[1][i])
        self.plot_freq_smoothing_window()
        # self.eqsignal.phase = self.current_signal.phase
        self.time_eq_signal = self.recovered_signal(self.eqsignal.freq_data[1], self.eqsignal.phase)
        self.eqsignal.time = self.current_signal.time
        self.Plot("equalized")

    def recovered_signal(self,Amp, phase):
        # complex array from amp and phase comination
        complex_value = Amp * np.exp(1j*phase)
        # taking inverse fft to get recover signal
        recovered_signal = np.fft.ifft(complex_value)
        # taking only the real part of the signal
        return np.real(recovered_signal)

    def hide(self):
        if (self.checkBox.isChecked()):
            self.spectrogram_before.hide()
            self.label_3.setVisible(False)
            self.spectrogram_after.hide()
            self.label_4.setVisible(False)
        else:
            self.spectrogram_before.show()
            self.label_3.setVisible(True)
            self.spectrogram_after.show()
            self.label_4.setVisible(True)
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = EqualizerApp()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()