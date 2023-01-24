import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from midi2audio import FluidSynth
import IPython.display as ipd
# from Preprocessing import *
from Preprocessing import make_midi
from Preprocessing import import_input_data_from_csv
from Preprocessing import export_output_data_to_csv

modelfile = "./mymodel"

def start_process():
  print("Creating a melody...")
  model = tf.keras.models.load_model(modelfile)
  file_path = "./tf_inputMon_Dec_19_10-30-04_GMT+09-00_2022.csv"
  file_name = file_path.replace("./", "").replace(".csv", "")
  model.summary()
  print(model.inputs)
  # n_steps = model.inputs[0].shape[1]

  # x = make_input_data(data, n_steps, 360, 36, 96)
  x = import_input_data_from_csv(file_path)
  x = np.expand_dims(np.expand_dims(x, axis=-1), axis=0)
  y = model.predict(x) 
  export_output_data_to_csv(y, file_name)
  plt.matshow(np.transpose(np.squeeze(y)))
  plt.show()
  print("Generating a MIDI file...")
  melody1, melody2, chord = make_midi(np.squeeze(x), np.squeeze(y), "output.mid", 36, 4)
  print("Converting the MIDI file to a MP3 file...")
  fs = FluidSynth(sound_font='font.sf2')
  fs.midi_to_audio("output.mid", "output.mp3")
  ipd.display(ipd.Audio("output.mp3"))



if __name__ == "__main__":
    start_process()


