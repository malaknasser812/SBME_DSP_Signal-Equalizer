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
## Screenshot:
### Applying Animals Mode:
### ![Screenshot (49)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/73aa846b-9e90-4811-bb3a-2043ffe59d7b)
### Equalize Animals Mode Using Rectangular:
### ![Screenshot (50)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/ae8cc03a-324f-47be-be1f-729af94f12a8)
### Equalize Animals Mode Using Hamming:
### ![Screenshot (51)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/23e1315e-53db-49b2-ba84-88d0ed2e9617)
### Equalize Animals Mode Using Hanning:
### ![Screenshot (52)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/68f9cede-4fcb-43c9-b378-b388c2e14527)
### Equalize Animals Mode Using Gaussian with 50 std:
### ![Screenshot (53)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/4328a5ab-3f1e-4092-8375-4822175d7ca7)
### Equalize Musical Instruments Mode Using Gaussian with 50 std:
### ![Screenshot (54)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/22de29cb-6707-4148-9743-b5edba5eb0bf)
### Equalize ECG Abnormalities Mode Using Rectangular:
### ![Screenshot (55)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/09670bc7-5445-425c-aaf6-9d8683316c61)
### Hide Spectogram:
### ![Screenshot (56)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/28275f95-3647-4964-ae25-86c7eb1c6430)
### Applying Uniform Mode Using Rectangular:
### ![Screenshot (57)](https://github.com/hagersamir/SBME_DSP_Signal-Equalizer/assets/105936147/b3cf98d1-2e08-4952-8663-ffba42594b84)

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
