
import numpy as np
import music21
import pretty_midi
import pandas as pd


default_chords = ["C7", "F7", "C7", "C7", "F7", "F7", "C7", "C7","G7", "F7", "C7", "G7"]


def make_midi(x, y, filename, pitch_from, div):
    My = y.shape[1]
    melody1 = y[:, 0:((My-1)//2)]
    melody2 = y[:, ((My-1)//2):(My-1)]
    Mx = x.shape[1]
    chords = x[:, (Mx-12):Mx]
    midi = pretty_midi.PrettyMIDI(resolution=480)
    midi.instruments.append(
    make_note_msgs(melody1, melody2, 1, pitch_from, 100, div))
    midi.instruments.append(make_note_msgs(chords, np.zeros(chords.shape), div, 48, 60, div))
    midi.write(filename)
    return melody1, melody2, chords

def make_note_msgs(pianoroll, pianoroll2, reso, pitch_from, velocity, div2):
    instr = pretty_midi.Instrument(program=1)
    for i in range(pianoroll.shape[0]):
        if (i % reso == 0):
            for j in range(pianoroll.shape[1]):
                if pianoroll[i, j] <= 0.5 and pianoroll2[i, j] > 0.5:
                    if i >= 1 and pianoroll2[i-1, j] < 0.5 and pianoroll[i-1, j] < 0.5:
                        pianoroll[i, j] += pianoroll2[i, j]
                        pianoroll2[i, j] = 0
                if pianoroll[i, j] > 0.5:
                    dur = 1
                    for k in range(i+1, pianoroll.shape[0]):
                        if pianoroll2[k, j] > 0.5:
                            dur += 1
#                       pianoroll2[k, j] = 0
                        else:
                            break
                    instr.notes.append(pretty_midi.Note(start=i / div2 / 2, 
                                            end=(i+dur) / div2 / 2, 
                                            pitch=pitch_from+j, 
                                            velocity=velocity))
    return instr


def make_chord_matrix(chords, count, repeat):

    matrix = np.zeros((len(chords) * count * repeat, 12))
    i = 0
    for r in range(repeat):
        for c in chords:
            notes = music21.harmony.ChordSymbol(c)._notes
            for k in range(count):
                for n in notes:
                    matrix[i, n.pitch.pitchClass] = 1
                i += 1
    return matrix

def make_onehot(data, nn_min, nn_max, extra):

    matrix = np.zeros((len(data), nn_max - nn_min + extra + 1))
    for i in range(len(data)):
        matrix[i, data[i] - nn_min if data[i] > 0 else -1] = 1
    return matrix

def make_input_data(data, n_steps, height, nn_min, nn_max):
    
    new_data = np.zeros(n_steps)
    delta = int(len(data) / n_steps)
    for i in range(min(int(len(data) / delta), n_steps)):
        new_data[i] = data[delta * i]
        nn_data = [(height - x) / height * (nn_max - nn_min) + nn_min for x in new_data]
        nn_data = [int(x) if not np.isnan(x) else -1 for x in nn_data]
        matrix1 = make_onehot(nn_data, nn_min, nn_max, nn_max - nn_min)
        matrix2 = make_chord_matrix(default_chords, 16, 2)
        matrix = np.concatenate([matrix1, matrix2], axis=1)
    return matrix


def import_input_data_from_csv(filepath):

    names = [ f"melody_{i}" if i<=120 else f"chord_{i}" for i in range(1, 134 ,1) ]
    new_data = np.zeros((368, 133))
    names[120] = "rest"
    df = pd.read_csv(filepath, names=names, index_col=False)
    data =np.array(df.values)
    input_data = np.concatenate([data, new_data], axis=0)
    return input_data

def export_output_data_to_csv(output_data, file_name):

    data = np.squeeze(output_data)
    df = pd.DataFrame(data)
    df.to_csv(f"./log/output_from_{file_name}.csv", header=False, index=False)

