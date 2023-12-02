# SBME_DSP_Signal-Equalizer
## Introduction :-
The Signal Equalizer Desktop Application is a versatile tool designed to manipulate signal frequencies in various modes, catering to the needs of music, speech, biomedical applications, and more. The application allows users to open signal files, adjust frequency components using sliders, and reconstruct the modified signal in real-time. It also gives you the ability to play, pause, and Equalize any one-channel wav audio file.
## Features :-
### Modes:

The application operates in different modes:

- Uniform Range Mode: Divides the signal's frequency range into 10 equal segments, each controlled by a slider.
- Musical Instruments Mode: Enables manipulation of specific musical instrument frequencies in a mixed music signal.
- Animal Sounds Mode: Allows control over distinct animal sound frequencies within a mixture of animal sounds.
- ECG Abnormalities Mode: Provides control over arrhythmia components in ECG signals, offering four different arrhythmia types.

### Slider Control and Smoothing Windows:

- Users can adjust frequency magnitudes using sliders, employing four multiplication/smoothing windows (Rectangle, Hamming, Hanning, Gaussian).
- Customization of window parameters is available via a visual interface, offering flexibility in signal manipulation.

### User Interface:
- Easy mode-switching via an option menu or combobox with minimal UI changes among modes.
- Slider captions dynamically change based on the selected mode.
- UI includes:
  - Two linked cine signal viewers for input and output signals with full functionality panel (play/stop/pause/speed-control/zoom/pan/reset).
  - Synchronous viewing of signals ensures both viewers display the same time-part of the signal during scroll or zoom.
  - Toggle option for displaying/hiding spectrograms (input and output).
  - Spectrograms reflect real-time changes made through the equalizer sliders.

 ## Installation :-
1. Clone the repository
```sh
   git clone https://github.com/malaknasser812/SBME_DSP_Signal-Equalizer.git
 ```
2. Install project dependencies
```sh
   pip install bisect
   pip install PySide6
   pip install PyQt5
   pip install librosa
   pip install pandas
   pip install numpy
   pip install pyqtgraph
   pip install matplotlib
 ```
3. Run the application
```sh
   python Main.py
```
## Libraries :-
- PyQt5
- pyqtgraph
- librosa
- bisect
- matplotlib
- QMediaContent
- QMediaPlayer
- scipy.signal
- scipy.io.wavfile
- scipy.signal.spectrogram
- numpy
- pandas

## Usage :-

### Opening and Selecting Modes:

- Launch the Signal Equalizer application.
- Navigate to the mode selection menu or combobox.
- Choose the desired mode based on the signal you wish to manipulate:
  - Uniform Range Mode: If you want to evenly control different frequency ranges across the signal.
  - Musical Instruments Mode: When manipulating specific musical instrument frequencies in a mixed music signal.
  - Animal Sounds Mode: For adjusting distinct animal sound frequencies within a mixture of animal sounds.
  - ECG Abnormalities Mode: To control arrhythmia components in ECG signals.

### Adjusting Frequency Components:

- Once in the selected mode, sliders corresponding to the signal components will appear on the UI.
- Use these sliders to modify the magnitude of frequency components:
  - In Uniform Range Mode, each slider represents a specific frequency range.
  - In Musical Instruments and Animal Sounds Modes, sliders correspond to individual instruments or animal sounds.
  - In ECG Abnormalities Mode, sliders control different types of arrhythmia components.

### Windowing and Smoothing:

- Choose a suitable multiplication/smoothing window (Rectangle, Hamming, Hanning, Gaussian) via the UI.
- Customize window parameters visually if necessary to fine-tune the signal manipulation process.
Visualizing Signals.

### Visualizing Signals:

- Utilize the two linked cine signal viewers for input and output signals.
  - These viewers offer a full set of functionality such as play, stop, pause, speed control, zoom, pan, and reset.
  - Ensure synchronous viewing, enabling both viewers to display the same time-part of the signal during scroll or zoom.
- Toggle the visibility of spectrograms (input and output) as needed to visualize frequency changes in the signals.
  - Spectrograms reflect real-time modifications made through the equalizer sliders.
 
### Validation:

- For testing and validation purposes, use the provided synthetic signal file.
- Track frequency modifications and observe the behavior of the application to ensure correct functionality across different modes.

## Our Team

- Hager Samir
- Camellia Marwan
- Farah Ossama
- Malak Nasser
