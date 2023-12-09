from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
import copy
from PyQt5.QtWidgets import QSlider,QHBoxLayout , QLabel
import matplotlib as plt
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, uic 
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
import os
import sys
plt.use('Qt5Agg')
import librosa
import bisect
import pyqtgraph as pg
from scipy import signal as sg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sounddevice as sd
import numpy as np

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
        self.amp = amp
        self.sigma = sigma
    def apply(self, signal):
        if self.window_type == "Rectangular":
            window = sg.windows.boxcar(len(signal))
            smoothed_signal = self.amp * window 
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
        self.original_graph.setBackground("#ffff")
        self.equalized_graph.setBackground("#ffff")
        self.frequancy_graph.setBackground("#ffff")
        self.selected_mode = None
        self.selected_window = None
        self.frame_layout = QHBoxLayout(self.sliders_frame)
        self.current_signal=None
        self.player = QMediaPlayer(None,QMediaPlayer.StreamPlayback)
        self.player.setVolume(50)
        self.timer = QTimer(self)
        self.timer = QtCore.QTimer(self)
        self.elapsed_timer = QtCore.QElapsedTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.updatepos)
        self.line = pg.InfiniteLine(pos=0, angle=90, pen=None, movable=False)
        self.changed_orig = False
        self.changed_eq = False
        self.player.positionChanged.connect(self.updatepos)
        self.current_speed = 1
        self.slider_gain = {}
        self.equalized_bool = False
        self.time_eq_signal = Signal('EqSignalInTime')
        self.eqsignal = None
        self.sampling_rate = None
        self.line = pg.InfiniteLine(pos=0.1, angle=90, pen=None, movable=False)
        # spectooooooo
        self.available_palettes = ['twilight', 'Blues', 'Greys', 'ocean', 'nipy_spectral']
        self.current_color_palette = self.available_palettes[0]
        self.spectrogram_widget = {
            'before': self.spectrogram_before,
            'after': self.spectrogram_after
        }
        # Ui conection
        self.modes_combobox.activated.connect(lambda: self.combobox_activated())
        self.smoothing_window_combobox.activated.connect(lambda: self.smoothing_window_combobox_activated())
        self.lineEdit_2.setVisible(False)  # Initially hide the line edit for Gaussian window
        self.load_btn.clicked.connect(lambda: self.load())
        self.hear_orig_btn.clicked.connect(lambda:self.playMusic('orig'))
        self.hear_eq_btn.clicked.connect(lambda:self.playMusic('equalized'))
        self.apply_btn.clicked.connect(lambda: self.plot_freq_smoothing_window())
        self.play_pause_btn.clicked.connect(lambda: self.play_pause()) 
        self.replay_btn.clicked.connect(lambda: self.replay())
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_in())
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_out())
        self.speed_up_btn.clicked.connect(lambda: self.speed_up()) 
        self.speed_down_btn.clicked.connect(lambda: self.speed_down())  
        self.checkBox.stateChanged.connect(lambda : self.hide())
        self.dictionary = {
            'Uniform Range':{},
            'Musical Instruments': {"Guitar": [40,400],
                                "Flute": [400, 800],
                                "Violin ": [950, 4000],
                                "Xylophone": [5000, 14000]
                                },
            "Animal Sounds":{"Dog": [0, 450],
                                "Wolf": [450, 1100],
                                "Crow": [1100, 3000],
                                "Bat": [3000, 9000]
                                },
            'ECG Abnormalities': {"Normal" : [0,35],
                                "Arithmia_1 ": [48, 52],
                                "Arithmia_2": [55, 94],
                                "Arithmia_3": [95, 155]
                                }
        }

#OUR CODE HERE 
    def load(self):
        path_info = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select a signal...",os.getenv('HOME'), filter="Raw Data (*.csv *.wav *.mp3)")
        path = path_info[0]
        # print(path)
        time = []
        self.equalized_bool = False
        sample_rate = 0
        data = []
        signal_name = path.split('/')[-1].split('.')[0]   # Extract signal name from file path
        type = path.split('.')[-1]
        # Check the file type and load data accordingly
        if type in ["wav", "mp3"]:
            data, sample_rate = librosa.load(path)
            Duration = librosa.get_duration(y=data, sr=sample_rate)
            self.duration = Duration
            time = np.linspace(0, Duration, len(data))
            self.audio_data = path
        elif type == "csv":
            data_of_signal = pd.read_csv(path)  
            time = np.array(data_of_signal.iloc[:,0].astype(float).tolist())
            data = np.array(data_of_signal.iloc[:,1].astype(float).tolist())
            if len(time) > 1:
                sample_rate = 1 /( time[1]-time[0])
                sample_rate = 1 /(time[1]-time[0])
            else:
                sample_rate=1
        # Create a Signal object and set its attributes
        self.current_signal = Signal(signal_name)
        self.current_signal.data = data
        self.current_signal.time = time
        self.current_signal.sample_rate = sample_rate 
        self.sampling_rate = sample_rate
        # Calculate and set the Fourier transform of the signal
        T = 1 / self.current_signal.sample_rate
        x_data, y_data = self.get_Fourier(T, self.current_signal.data)
        self.current_signal.freq_data = [x_data, y_data]
        for i in range(10):
            self.batch_size = len(self.current_signal.freq_data[0])//10 
            self.dictionary['Uniform Range'][i] = [i*self.batch_size,(i+1)*self.batch_size]  
        # selected_index = None
        # self.add_slider(selected_index)
        self.frequancy_graph.clear()
        if self.spectrogram_after.count() > 0:
            # If yes, remove the existing canvas
            self.spectrogram_after.itemAt(0).widget().setParent(None)
        self.Plot("original")
        self.plot_spectrogram(data, sample_rate , self.spectrogram_before)
        self.frequancy_graph.plot(self.current_signal.freq_data[0],
                    self.current_signal.freq_data[1],pen={'color': 'b'})
        self.eqsignal = copy.deepcopy(self.current_signal)

    def get_Fourier(self, T, data):
        N=len(data)
        freq_amp= np.fft.fft(data)
        self.current_signal.phase = np.angle(freq_amp[:N//2])
        # Calculate the corresponding frequencies
        Freq= np.fft.fftfreq(N, T)[:N//2]
        # Extracting positive frequencies and scaling the amplitude
        Amp = (2/N)*(np.abs(freq_amp[:N//2]))
        return Freq, Amp

    def Range_spliting(self):
        freq = self.current_signal.freq_data[0]  # Index zero for values of freq
        if self.selected_mode == 'Uniform Range':
            self.current_signal.Ranges = [(i*self.batch_size,(i+1)*self.batch_size) for i in range(10)] 
        else: 
            dict = self.dictionary[self.selected_mode]
            # Calculate frequency indices for specified ranges
            for  _, (start,end) in dict.items():
                start_ind = bisect.bisect_left(freq, start)
                end_ind = bisect.bisect_right(freq, end) - 1  # Adjusted for inclusive end index
                self.current_signal.Ranges.append((start_ind, end_ind))
        self.eqsignal.Ranges = copy.deepcopy(self.current_signal.Ranges)

    def Plot(self, graph):
            signal= self.time_eq_signal if self.equalized_bool else self.current_signal
            if signal:
                #time domain 
                self.equalized_graph.clear()
                graphs = [self.original_graph, self.equalized_graph]
                graph = graphs[0] if graph == "original" else graphs[1]                
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
        #print(signal.Ranges)
        if signal and signal.Ranges:  # Check if signal is not None and signal.Ranges is not empty
            _, end_last_ind = signal.Ranges[-1]
            #frequency domain
            self.frequancy_graph.clear()
            # Plot the original frequency data in white
            self.frequancy_graph.plot(signal.freq_data[0][:end_last_ind],
                    signal.freq_data[1][:end_last_ind],pen={'color': 'b'})
            # Iterate through the frequency ranges and plot smoothed windows
            for i in range(len(signal.Ranges)):
                if i!= len(signal.Ranges) :
                    #print(signal.Ranges[i])
                    start_ind,end_ind = signal.Ranges[i]
                    # print(signal.Ranges[i])
                    # Get smoothing window parameters
                    windowtype = self.smoothing_window_combobox.currentText()
                    # Convert sigma_text to integer if not empty, otherwise set a default value
                    sigma_text = self.lineEdit_2.text()
                    sigma = int(sigma_text) if sigma_text else 20  # Set a default value if the text is empty
                    amp = max(signal.freq_data[1][start_ind:end_ind])
                    # Apply the smoothing window
                    smooth_window = SmoothingWindow(windowtype,amp,sigma)
                    curr_smooth_window = smooth_window.apply(signal.freq_data[1][start_ind:end_ind])
                    # print(len(signal.freq_data[0][start_ind:end_ind]))
                    # print( len(curr_smooth_window))
                    self.frequancy_graph.plot(signal.freq_data[0][start_ind:end_ind],
                            curr_smooth_window,pen={'color': 'r', 'width': 2})
                    start_line = signal.freq_data[0][start_ind]
                    end_line = signal.freq_data[0][end_ind-1]
                # Add a vertical line for the current position
                v_line_start = pg.InfiniteLine(pos=start_line, angle=90, movable=False, pen=pg.mkPen('r', width=2))
                self.frequancy_graph.addItem(v_line_start)
                v_line_end = pg.InfiniteLine(pos=end_line, angle=90, movable=False, pen=pg.mkPen('r', width=2))
                self.frequancy_graph.addItem(v_line_end)

    def plot_spectrogram(self, samples, sampling_rate , widget):
        if widget.count() > 0:
            # If yes, remove the existing canvas
            widget.itemAt(0).widget().setParent(None)
        data = samples.astype('float32')
        # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
        n_fft=500
        # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
        hop_length=320
        window_type ='hann'
        # Compute the short-time Fourier transform magnitude squared
        frequency_magnitude = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type)) ** 2
        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(S=frequency_magnitude, y=data, sr=sampling_rate, n_fft=n_fft,
                    hop_length=hop_length, win_length=n_fft, window=window_type, n_mels =128)
        # Convert power spectrogram to decibels
        decibel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        time_axis = np.linspace(0, len(data) / sampling_rate)
        fig = Figure()
        fig = Figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ax.imshow(decibel_spectrogram, aspect='auto', cmap='viridis',extent=[time_axis[0], time_axis[-1], 0, sampling_rate / 2])
        ax.axes.plot()
        canvas = FigureCanvas(fig)
        widget.addWidget(canvas)

    def replay (self):
        if self.type == 'orig':
            self.playMusic('orig')
        else: self.playMusic('equalized')

    def playMusic(self, type):
        self.current_speed = 1
        self.line_position = 0
        self.player.setPlaybackRate(self.current_speed)
        media = QMediaContent(QUrl.fromLocalFile(self.audio_data))
        # Set the media content for the player and start playing
        self.player.setMedia(media)
        self.type = type
        if type == 'orig':
            sd.stop()
            self.timer.stop()
            self.changed_orig = True
            self.changed_eq = False
            # Create a QMediaContent object from the local audio file
            self.player.play()
            self.player.setVolume(100)
            # Add a vertical line to the original graph
            self.equalized_graph.removeItem(self.line)
            self.original_graph.addItem(self.line)
            self.timer.start()
        else:
            self.changed_eq = True
            self.changed_orig = False
            self.timer.start()
            self.player.play()
            self.player.setVolume(0)
            self.original_graph.removeItem(self.line)
            self.equalized_graph.addItem(self.line)
            sd.play(self.time_eq_signal.data, self.current_signal.sample_rate, blocking=False)
            self.player.play()
                        
    def updatepos(self):
            max_x = self.original_graph.getViewBox().viewRange()[0][1]
            graphs = [self.original_graph, self.equalized_graph]
            graph = graphs[0] if self.changed_orig  else graphs[1]
        # Get the current position in milliseconds
            position = self.player.position()/1000
            # Update the line position based on the current position
            self.line_position = position 
            max_x = graph.getViewBox().viewRange()[0][1]
            #print(position)
            if self.line_position > max_x:
                self.line_position = max_x
            self.line_position = position
            self.line.setPos(self.line_position)
        
    def speed_up(self):
        # Increase the playback speed
        self.current_speed = self.current_speed + 0.1  # You can adjust the increment as needed
        self.player.setPlaybackRate(self.current_speed)
        if self.changed_eq :
            sd.play(self.time_eq_signal.data, self.current_signal.sample_rate, speed = self.current_speed, volume = 1.0 )
        #print(self.current_speed)

    def speed_down(self):
        # Decrease the playback speed
        self.current_speed = self.current_speed - 0.1  # You can adjust the increment as needed
        new_speed = max(0.1, self.current_speed - 0.1)  # Ensure speed doesn't go below 0.1
        self.player.setPlaybackRate(new_speed)
        if self.changed_eq :
            sd.play(self.time_eq_signal.data, self.current_signal.sample_rate, speed = self.current_speed, volume = 1.0 )
        #print(new_speed)

    def replay (self):
        if self.type == 'orig':
            self.playMusic('orig')
        else: self.playMusic('equalized')

    def zoom_in(self):
        self.original_graph.getViewBox().scaleBy((0.5, 0.5))
        self.equalized_graph.getViewBox().scaleBy((0.5, 0.5))

    def zoom_out(self):
        self.original_graph.getViewBox().scaleBy((2, 2))
        self.equalized_graph.getViewBox().scaleBy((2, 2))

    def play_pause(self):
        if self.changed_orig:
            self.player.pause()
            self.timer.stop()
            self.changed_orig = not self.changed_orig
        else:
            self.player.play()
            self.timer.start()
            self.changed_orig = not self.changed_orig

    def combobox_activated(self):
        # Get the selected item's text and display it in the label
        selected_index = self.modes_combobox.currentIndex()
        self.selected_mode = self.modes_combobox.currentText()
        # store the mode in a global variable 
        self.add_slider(self.selected_mode)
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

    def add_slider(self, selected_index):          
        self.clear_layout(self.frame_layout) 
        dictinoary = self.dictionary[selected_index]
        for i,(key,_ )in enumerate(dictinoary.items()):
            # print(f"Index: {i}, Key: {key}")
            label = QLabel(str(key))  # Create a label with a unique identifier
            slider_creator = CreateSlider(i)
            slider = slider_creator.get_slider()
            self.slider_gain[i] = 10
            slider.valueChanged.connect(lambda value, i=i: self.update_slider_value(i, value/10))
            self.frame_layout.addWidget(slider)
            self.frame_layout.addWidget(label)
        
    def update_slider_value(self, slider_index, value):
        # This method will be called whenever a slider is moved
        self.slider_gain[slider_index] = value
        #print (self.slider_gain)
        self.equalized(slider_index, value)
        #self.Plot('equalized')

    def zoom_in(self):
        self.original_graph.getViewBox().scaleBy((0.5, 0.5))
        self.equalized_graph.getViewBox().scaleBy((0.5, 0.5))
        print('zoomed in')
       

    def zoom_out(self):
        self.original_graph.getViewBox().scaleBy((2, 2))
        self.equalized_graph.getViewBox().scaleBy((2, 2))
        print('zoomed out')

    

    def equalized(self, slider_index,value):
        #print (value)
        self.equalized_bool = True
        self.time_eq_signal.time = self.current_signal.time
        # Get smoothing window parameters
        windowtype = self.smoothing_window_combobox.currentText()
        # Convert sigma_text to integer if not empty, otherwise set a default value
        sigma_text = self.lineEdit_2.text()
        sigma = int(sigma_text) if sigma_text else 20  # Set a default value if the text is empty
        start,end = self.current_signal.Ranges[slider_index]
        # Apply the smoothing window
        smooth_window = SmoothingWindow(windowtype,1,sigma)
        curr_smooth_window = smooth_window.apply(self.current_signal.freq_data[1][start:end])
        curr_smooth_window *= value
        Amp = np.array(self.current_signal.freq_data[1][start:end])   
        new_amp = Amp * curr_smooth_window
        self.eqsignal.freq_data[1][start:end] = new_amp
        self.plot_freq_smoothing_window()
        self.time_eq_signal.time = self.current_signal.time
        self.time_eq_signal.data = self.recovered_signal(self.eqsignal.freq_data[1], self.current_signal.phase)
        #print(len(self.time_eq_signal.data))
        #print(len(self.time_eq_signal.time))
        excess = len(self.time_eq_signal.time)-len(self.time_eq_signal.data)
        self.time_eq_signal.time = self.time_eq_signal.time[:-excess]
        self.Plot("equalized")
        self.plot_spectrogram(self.time_eq_signal.data, self.current_signal.sample_rate , self.spectrogram_after)

    def recovered_signal(self,Amp, phase):
        # complex array from amp and phase comination
        Amp = Amp * len(self.current_signal.data)/2 #N/2 as we get amp from foureir by multiplying it with fraction 2/N 
        complex_value = Amp * np.exp(1j*phase)
        # taking inverse fft to get recover signal
        recovered_signal = np.fft.irfft(complex_value)
        # taking only the real part of the signal
        return (recovered_signal)
    
    def hide(self):
        if (self.checkBox.isChecked()):
            self.specto_frame_before.hide()
            self.label_3.setVisible(False)
            self.specto_frame_after.hide()
            self.label_4.setVisible(False)
        else:
            self.specto_frame_before.show()
            self.label_3.setVisible(True)
            self.specto_frame_after.show()
            self.label_4.setVisible(True)
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = EqualizerApp()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()