import numpy as np
import matplotlib.pyplot as plt
import os


def plot_fft_result_2d(fft_result, title="FFT Magnitude (2D)", xlabel="Frequency X", ylabel="Frequency Y", output_file="fft_plot_2d.png"):
    """
    Plotta e salva i risultati della FFT 2D.

    Args:
        fft_result (np.array): Risultati complessi della FFT 2D.
        title (str): Titolo del grafico.
        xlabel (str): Etichetta dell'asse x.
        ylabel (str): Etichetta dell'asse y.
        output_file (str): Nome del file per salvare il grafico (es. 'output.png').
    """
    # Calcolo della magnitudine
    magnitudes = np.abs(fft_result)

    # Creazione del grafico 2D
    plt.figure(figsize=(8, 8))
    im = plt.imshow(magnitudes, extent=[0, magnitudes.shape[1], 0, magnitudes.shape[0]], origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Magnitude")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Salva il grafico su file
    plt.savefig(output_file)
    print(f"Grafico salvato come {output_file}")

def load_fft_data_2d(filename):
    """
    Carica i dati FFT 2D esportati dal file CSV.

    Args:
        filename (str): Nome del file CSV.

    Returns:
        np.array: Risultati complessi della FFT 2D.
    """
    with open(filename, 'r') as file:
        data = file.readlines()

    rows = []
    for i, line in enumerate(data):
        row = []
        for j, val in enumerate(line.strip().split(',')):
            try:
                row.append(complex(val.strip()))  # Rimuove spazi extra e converte
            except ValueError:
                print(f"Valore malformato: '{val.strip()}' nella riga {i + 1}, colonna {j + 1}")
                row.append(0 + 0j)  # Sostituisce il valore malformato con 0+0j
        rows.append(row)

    return np.array(rows, dtype=complex)



# Debug: Verifica la directory corrente
print("Current working directory:", os.getcwd())

# Percorso al file esportato dal C++
filename = "../CUDA_FFT/2D/fft_output_2d.csv"

# Carica i dati
fft_result_2d = load_fft_data_2d(filename)

# Percorso per salvare il grafico
output_file_path = "../Plot_result/fft_plot_2d.png"

# Plotta i dati
plot_fft_result_2d(fft_result_2d, title="FFT Visualization (2D)", xlabel="Frequency X", ylabel="Frequency Y", output_file=output_file_path)
