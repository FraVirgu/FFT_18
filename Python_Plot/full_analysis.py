import plot_1d
import plot_2d
import os

# --- Imposta cartella di output ---
output_dir = "../Plot_result"
os.makedirs(output_dir, exist_ok=True)

print("Plotting Singular Values...")
singular_values = plot_1d.load_singular_values("../image_compression/output/csv_output/singular_values.csv")
plot_1d.plot_singular_values(
    singular_values,
    title="Singular Values Decay",
    xlabel="Index",
    ylabel="Singular Value",
    output_file=os.path.join(output_dir, "singular_values.png")
)

print("Plotting FFT Magnitude (2D)...")
fft_result_2d = plot_2d.load_fft_data_2d("../image_compression/output/csv_output/fft_output_2d.csv")
sampling_rate = 1000
plot_2d.plot_fft_result_2d(
    fft_result_2d,
    sampling_rate,
    title="FFT Magnitude (2D)",
    output_file=os.path.join(output_dir, "fft_magnitude_2d.png")
)

print("Plotting Filtered FFT Magnitude (2D)...")
filtered_magnitude = plot_2d.load_magnitude_data_2d("../image_compression/output/csv_output/fft_magnitude_filtered.csv")
plot_2d.plot_magnitude_2d(
    filtered_magnitude,
    title="Filtered FFT Magnitude (2D)",
    output_file=os.path.join(output_dir, "filtered_fft_magnitude_2d.png")
)
plot_2d.plot_magnitude_2d_log(
    filtered_magnitude,
    title="Filtered FFT Magnitude (2D) - Log Scale",
    output_file=os.path.join(output_dir, "filtered_fft_magnitude_2d_log.png")
)

print("Plotting Reconstruction Error vs Threshold...")
thresholds, errors = plot_1d.load_error_vs_threshold("../image_compression/output/csv_output/error_vs_threshold.csv")
plot_1d.plot_error_vs_threshold(
    thresholds,
    errors,
    title="Reconstruction Error vs Threshold",
    xlabel="Threshold",
    ylabel="Error",
    output_file=os.path.join(output_dir, "error_vs_threshold.png")
)

print("\nAll plots completed and saved in '../Plot_result' folder!")