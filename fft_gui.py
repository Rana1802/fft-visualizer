import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from math import log2, log
import tkinter as tk
from tkinter import messagebox

# ------------------------
# Utilities
# ------------------------
def is_power_of(n, base):
    if n < 1:
        return False
    while n % base == 0:
        n = n // base
    return n == 1

def bit_reverse_indices(n):
    bits = int(np.log2(n))
    return np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)])

# ------------------------
# FFT Algorithms
# ------------------------
def radix2_fft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    indices = bit_reverse_indices(N)
    x = x[indices]
    stages = int(log2(N))
    stage_values = [x.copy()]

    m = 2
    while m <= N:
        half_m = m // 2
        twiddle_factors = np.exp(-2j * np.pi * np.arange(half_m) / m)
        for k in range(0, N, m):
            for j in range(half_m):
                t = twiddle_factors[j] * x[k + j + half_m]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + half_m] = u - t
        stage_values.append(x.copy())
        m *= 2
    return x, stage_values

def radix3_fft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    stages = int(round(log(N, 3)))
    stage_values = [x.copy()]

    def fft_recursive(x):
        N = len(x)
        if N == 1:
            return x
        x0 = fft_recursive(x[0::3])
        x1 = fft_recursive(x[1::3])
        x2 = fft_recursive(x[2::3])

        X = [0] * N
        for k in range(N // 3):
            w1 = np.exp(-2j * np.pi * k / N)
            w2 = np.exp(-4j * np.pi * k / N)
            for i in range(3):
                idx = k + (N // 3) * i
                a = x0[k]
                b = x1[k] * (w1 ** i)
                c = x2[k] * (w2 ** i)
                X[idx] = a + b + c
        return X

    output = np.array(fft_recursive(x))
    stage_values.append(output.copy())
    return output, stage_values

# ------------------------
# GUI App
# ------------------------
class FFTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FFT Visualizer")
        self.root.configure(bg="#1e1e2f")

        tk.Label(root, text="Signal (comma-separated):", fg="white", bg="#1e1e2f").pack()
        self.signal_entry = tk.Entry(root, width=50)
        self.signal_entry.pack(pady=5)

        tk.Label(root, text="Signal Length (must be power of 2 or 3):", fg="white", bg="#1e1e2f").pack()
        self.length_entry = tk.Entry(root, width=10)
        self.length_entry.pack(pady=5)

        tk.Button(root, text="Run FFT", command=self.run_fft, bg="#3366cc", fg="white").pack(pady=10)

        self.canvas_frame = tk.Frame(root, bg="#1e1e2f")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(root, height=20, bg="#2e2e3f", fg="white", font=("Courier", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def run_fft(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        self.output_text.delete("1.0", tk.END)

        try:
            signal = [complex(x.strip()) for x in self.signal_entry.get().split(',')]
            N = int(self.length_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid signal values or length")
            return

        if len(signal) < N:
            signal.extend([0.0] * (N - len(signal)))
        elif len(signal) > N:
            signal = signal[:N]

        if is_power_of(N, 2):
            fft_out, stages = radix2_fft(signal)
            radix = 2
        elif is_power_of(N, 3):
            fft_out, stages = radix3_fft(signal)
            radix = 3
        else:
            messagebox.showerror("Input Error", "Signal length must be a power of 2 or 3")
            return

        self.display_output(fft_out)
        self.display_magnitude_and_phase_plot(fft_out)
        self.display_butterfly(stages, radix)

    def display_output(self, fft_out):
        self.output_text.insert(tk.END, "FFT Output (first few values):\n")
        for i, val in enumerate(fft_out):
            self.output_text.insert(tk.END, f"X[{i}] = {val.real:.3f}{val.imag:+.3f}j\n")

    def display_magnitude_and_phase_plot(self, fft_out):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5), facecolor="#1e1e2f")
        fig.subplots_adjust(hspace=0.4)
        ax1.set_facecolor("#1e1e2f")
        ax2.set_facecolor("#1e1e2f")

        ax1.stem(np.abs(fft_out), basefmt=" ", linefmt='lightblue', markerfmt='bo')
        ax1.set_title("Magnitude Spectrum", color='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)

        ax2.stem(np.angle(fft_out), basefmt=" ", linefmt='lightgreen', markerfmt='go')
        ax2.set_title("Phase Spectrum", color='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def display_butterfly(self, stages_values, radix):
        N = len(stages_values[0])
        stages = len(stages_values) - 1

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1e1e2f")
        ax.set_facecolor("#1e1e2f")
        ax.set_xlim(0, stages)
        ax.set_ylim(-1, N)
        ax.axis("off")

        for stage in range(stages + 1):
            for i in range(N):
                ax.plot(stage, i, 'o', color='skyblue')
                val = stages_values[stage][i]
                ax.text(stage, i + 0.25, f"{val.real:.1f}{val.imag:+.1f}j", fontsize=7, ha='center', color='white')

        if radix == 2:
            for stage in range(stages):
                step = 2 ** stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        src1 = i + j
                        src2 = i + j + step
                        ax.plot([stage, stage + 1], [src1, src1], color='gray')
                        ax.plot([stage, stage + 1], [src2, src1], color='gray')
                        ax.plot([stage, stage + 1], [src1, src2], color='gray')
                        ax.plot([stage, stage + 1], [src2, src2], color='gray')
        elif radix == 3:
            base = 3
            for stage in range(stages):
                group_size = base ** (stage + 1)
                for group in range(N // group_size):
                    offset = group * group_size
                    for i in range(group_size // base):
                        i0 = offset + i
                        i1 = offset + i + group_size // base
                        i2 = offset + i + 2 * (group_size // base)
                        for target in [i0, i1, i2]:
                            ax.plot([stage, stage + 1], [i0, target], color='gray')
                            ax.plot([stage, stage + 1], [i1, target], color='gray')
                            ax.plot([stage, stage + 1], [i2, target], color='gray')

        ax.set_title(f"Radix-{radix} FFT Butterfly Diagram", color='white')
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# ------------------------
# Run GUI
# ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FFTApp(root)
    root.mainloop()
