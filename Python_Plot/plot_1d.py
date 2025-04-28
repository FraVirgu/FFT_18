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


def load_singular_values(filename):
    """
    Carica i valori singolari da un file CSV (un valore per riga).

    Args:
        filename (str): Path del file CSV.

    Returns:
        np.array: Array di valori singolari.
    """
    data = np.loadtxt(filename)
    return data

def plot_singular_values(singular_values, title="Singular Values", xlabel="Index", ylabel="Value",output_file="../Plot_result/singular_values.png"):
    """
    Plotta i valori singolari e opzionalmente salva il grafico.

    Args:
        singular_values (np.array): Array di valori singolari.
        title (str): Titolo del grafico.
        xlabel (str): Etichetta asse x.
        ylabel (str): Etichetta asse y.
        output_file (str, opzionale): Path per salvare il grafico.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(singular_values)), singular_values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Grafico salvato come {output_file}")
    plt.show()

def load_error_vs_threshold(filename):
    """
    Carica i dati di errore vs soglia da un file CSV.

    Args:
        filename (str): Percorso al file CSV.

    Returns:
        tuple: thresholds (np.array), errors (np.array)
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    thresholds = data[:, 0]
    errors = data[:, 1]
    return thresholds, errors

def plot_error_vs_threshold(thresholds, errors, title="Reconstruction Error vs Threshold", xlabel="Threshold", ylabel="Reconstruction Error", output_file="../Plot_result/Error_vs_Threshold.png"):
    """
    Plotta l'errore di ricostruzione rispetto alla soglia.

    Args:
        thresholds (np.array): Valori di soglia.
        errors (np.array): Errori corrispondenti.
        title (str): Titolo del grafico.
        xlabel (str): Etichetta asse x.
        ylabel (str): Etichetta asse y.
        output_file (str, opzionale): Path per salvare il grafico.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, errors, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Grafico salvato come {output_file}")
    plt.show()