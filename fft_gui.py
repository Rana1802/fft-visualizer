import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from math import log2, log
import tkinter as tk
from tkinter import messagebox

# ------------------------
# Signal Generator
# ------------------------
def generate_signal(frequencies, sampling_rate=81, duration=1):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    return t, signal

# ------------------------
# Power Check Utility
# ------------------------
def is_power_of(n, base):
    if n < 1:
        return False
    while n % base == 0:
        n = n // base
    return n == 1

# ------------------------
# Radix-2 FFT
# ------------------------
def radix2_fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = radix2_fft(x[::2])
    odd = radix2_fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]

# ------------------------
# Radix-3 FFT
# ------------------------
def radix3_fft(x):
    N = len(x)
    if N == 1:
        return x
    elif N % 3 != 0:
        raise ValueError("Length must be a power of 3 for Radix-3 FFT")

    x0 = radix3_fft(x[0::3])
    x1 = radix3_fft(x[1::3])
    x2 = radix3_fft(x[2::3])

    X = [0] * N
    for k in range(N // 3):
        twiddle1 = np.exp(-2j * np.pi * k / N)
        twiddle2 = np.exp(-4j * np.pi * k / N)
        for i in range(3):
            idx = k + (N // 3) * i
            a = x0[k]
            b = x1[k] * twiddle1**i
            c = x2[k] * twiddle2**i
            X[idx] = a + b + c
    return X

# ------------------------
# GUI Application
# ------------------------
class FFTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FFT Visualizer")
        self.root.configure(bg="#1e1e2f")

        tk.Label(root, text="Frequencies (comma-separated):", fg="white", bg="#1e1e2f").pack()
        self.freq_entry = tk.Entry(root, width=40)
        self.freq_entry.pack(pady=5)

        tk.Label(root, text="Signal Length (power of 2 or 3):", fg="white", bg="#1e1e2f").pack()
        self.len_entry = tk.Entry(root, width=20)
        self.len_entry.pack(pady=5)

        tk.Button(root, text="Run FFT", command=self.run_fft, bg="#3366cc", fg="white").pack(pady=10)

        self.canvas_frame = tk.Frame(root, bg="#1e1e2f")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def run_fft(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        try:
            frequencies = [float(f.strip()) for f in self.freq_entry.get().split(",") if f.strip()]
            N = int(self.len_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid frequencies or signal length")
            return

        if not frequencies:
            messagebox.showerror("Input Error", "Please enter at least one frequency")
            return

        if not (is_power_of(N, 2) or is_power_of(N, 3)):
            messagebox.showerror("Input Error", "Signal length must be power of 2 or 3")
            return

        self.signal_len = N
        self.signal_data = generate_signal(frequencies, sampling_rate=N, duration=1)[1]

        if is_power_of(N, 2):
            fft_out = radix2_fft(self.signal_data)
            title = "Radix-2 FFT"
        else:
            fft_out = radix3_fft(self.signal_data)
            title = "Radix-3 FFT"

        self.display_plots(self.signal_data, fft_out, N, title)

    def display_plots(self, signal, fft_result, N, title):
        plot_frame = tk.Frame(self.canvas_frame, bg="#1e1e2f")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(plot_frame, bg="#1e1e2f")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        right_frame = tk.Frame(plot_frame, bg="#1e1e2f")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.plot_result(signal, fft_result, title, left_frame)
        self.show_butterfly(N, title, right_frame)

    def plot_result(self, signal, fft_result, title, frame):
        N = len(signal)
        freq = np.fft.fftfreq(N)
        magnitude = np.abs(fft_result)

        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#1e1e2f")
        ax.set_facecolor("#1e1e2f")
        ax.stem(freq[:N//2], magnitude[:N//2], basefmt=" ", linefmt='lightblue', markerfmt='bo')
        ax.set_title(title, color='white')
        ax.set_xlabel("Frequency (normalized)", color='white')
        ax.set_ylabel("Magnitude", color='white')
        ax.tick_params(colors='white')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_butterfly(self, N, title, frame):
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1e1e2f")
        ax.set_facecolor("#1e1e2f")
        ax.axis("off")

        if "Radix-2" in title:
            stages = int(log2(N))
            color = 'skyblue'
        else:
            stages = int(round(log(N, 3)))
            color = 'lightgreen'

        for stage in range(stages + 1):
            for i in range(N):
                ax.plot(stage, i, 'o', color=color)

        if "Radix-2" in title:
            for stage in range(stages):
                step = 2 ** stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        src1 = i + j
                        src2 = i + j + step
                        ax.plot([stage, stage + 1], [src1, src1], color='gray', linewidth=1)
                        ax.plot([stage, stage + 1], [src2, src1], color='gray', linewidth=1)
                        ax.plot([stage, stage + 1], [src1, src2], color='gray', linewidth=1)
                        ax.plot([stage, stage + 1], [src2, src2], color='gray', linewidth=1)
        else:
            for stage in range(stages):
                group_size = 3 ** (stage + 1)
                for group in range(N // group_size):
                    base = group * group_size
                    for i in range(group_size // 3):
                        i0 = base + i
                        i1 = base + i + group_size // 3
                        i2 = base + i + 2 * (group_size // 3)
                        for target in [i0, i1, i2]:
                            ax.plot([stage, stage + 1], [i0, target], color='gray', linewidth=1)
                            ax.plot([stage, stage + 1], [i1, target], color='gray', linewidth=1)
                            ax.plot([stage, stage + 1], [i2, target], color='gray', linewidth=1)

        ax.set_title(f"{title} Butterfly Diagram", color='white')
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ------------------------
# Run the App
# ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FFTApp(root)
    root.mainloop()
