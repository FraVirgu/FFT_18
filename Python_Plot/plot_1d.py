import numpy as np

import matplotlib.pyplot as plt

def plot_fft_result(frequencies, fft_result, title="FFT Magnitude", xlabel="Frequency", ylabel="Magnitude", output_file="../Plot_result/fft_plot.png"):
    """
    Plotta e salva i risultati della FFT.

    Args:
        frequencies (list or np.array): Array con le frequenze corrispondenti.
        fft_result (list or np.array): Risultati complessi della FFT.
        title (str): Titolo del grafico.
        xlabel (str): Etichetta dell'asse x.
        ylabel (str): Etichetta dell'asse y.
        output_file (str): Nome del file per salvare il grafico (es. 'output.png').
    """
    # Calcolo della magnitudine
    magnitudes = np.abs(fft_result)
    
    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, magnitudes, label="FFT Magnitude")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Salva il grafico su file
    plt.savefig(output_file)
    print(f"Grafico salvato come {output_file}")
# Carica i dati esportati da C++
def load_fft_data(filename):
    """
    Carica i dati FFT esportati dal file CSV.

    Args:
        filename (str): Nome del file CSV.

    Returns:
        tuple: Frequenze e valori FFT complessi.
    """
    data = np.genfromtxt(filename, delimiter=',', dtype=complex)
    num_points = len(data)
    frequencies = np.fft.fftfreq(num_points)  # Frequenze normalizzate
    return frequencies, data


import os
print("Current working directory:", os.getcwd())

# Percorso al file esportato dal C++
filename = "../CUDA_FFT/1D/fft_output.csv"

# Carica i dati
frequencies, fft_result = load_fft_data(filename)

# Plotta i dati
plot_fft_result(frequencies, fft_result, title="FFT Visualization", xlabel="Frequency (Hz)", ylabel="Magnitude")
