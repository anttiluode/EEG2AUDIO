"""
RAJAPINTA EEG 2 AUDIO STUDIO
============================
A full GUI for turning EEG into audio with the Inverse Cochlea.

Features:
- Load any EDF file
- Choose channel + time range
- Real-time preview (waveform + spectrogram)
- Adjustable Inverse Cochlea settings (resonators, synthesis type, denoising)
- Play preview before exporting
- Export clean WAV

Run:
python eeg2audio_studio.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import soundfile as sf
import mne
import os
import tempfile
from scipy import signal
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ====================== CORE RAJAPINTA ENGINE ======================

def generate_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
            primes.append(num)
        num += 1
    return np.array(primes)

class PrimeLogAntiCochlea:
    def __init__(self, n_resonators=32, sfreq=256, audio_fs=16000, synth_type="sine"):
        self.n = n_resonators
        self.sfreq = sfreq
        self.audio_fs = audio_fs
        self.synth_type = synth_type  # "sine", "saw", "hybrid"
        
        primes = generate_primes(self.n)
        log_p = np.log(primes)
        log_p_norm = (log_p - log_p.min()) / (log_p.max() - log_p.min())
        
        self.brain_freqs = 0.5 + log_p_norm * 44.5
        self.audio_freqs = 80.0 * np.exp(log_p_norm * np.log(5000.0 / 80.0))
        
        self.phases = np.zeros(self.n)
        self.coherence_buffer = np.zeros(self.n)
        self.audio_phases = np.zeros(self.n)

    def process_and_synthesize(self, eeg_chunk, audio_samples_per_chunk):
        dt = 1.0 / self.sfreq
        energy = np.zeros(self.n)
        
        for sample in eeg_chunk:
            self.phases = (self.phases + 2 * np.pi * self.brain_freqs * dt) % (2 * np.pi)
            match = sample * np.sin(self.phases)
            self.coherence_buffer = self.coherence_buffer * 0.98 + match * 0.02
            energy = np.abs(self.coherence_buffer)
        
        if np.max(energy) > 1e-4:
            energy = energy / np.max(energy)
        
        audio = np.zeros(audio_samples_per_chunk)
        dt_audio = 1.0 / self.audio_fs
        phase_inc = 2 * np.pi * self.audio_freqs * dt_audio
        
        for i in range(self.n):
            if energy[i] > 0.01:
                amp = energy[i] * 0.28
                phase = self.audio_phases[i]
                inc = phase_inc[i]
                
                for j in range(audio_samples_per_chunk):
                    phase += inc
                    if self.synth_type == "sine":
                        audio[j] += amp * np.sin(phase)
                    elif self.synth_type == "saw":
                        audio[j] += amp * (2 * (phase % (2*np.pi)) / (2*np.pi) - 1)
                    else:  # hybrid
                        audio[j] += amp * (0.7 * np.sin(phase) + 0.3 * (2 * (phase % (2*np.pi)) / (2*np.pi) - 1))
                
                self.audio_phases[i] = phase % (2 * np.pi)
        
        return audio

def spectral_subtraction_denoise(audio, sr=16000, noise_percentile=12, over_sub=2.2, floor=0.08):
    nperseg = 512
    noverlap = 384
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    frame_energy = np.mean(mag, axis=0)
    noise_threshold = np.percentile(frame_energy, noise_percentile)
    quiet_mask = frame_energy < noise_threshold
    
    if np.sum(quiet_mask) > 3:
        noise_profile = np.mean(mag[:, quiet_mask], axis=1, keepdims=True)
    else:
        noise_profile = np.median(mag, axis=1, keepdims=True)
    
    mag_clean = mag - over_sub * noise_profile
    mag_clean = np.maximum(mag_clean, floor * mag)
    
    Zxx_clean = mag_clean * np.exp(1j * phase)
    _, audio_clean = signal.istft(Zxx_clean, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    
    if len(audio_clean) > len(audio):
        audio_clean = audio_clean[:len(audio)]
    elif len(audio_clean) < len(audio):
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
    
    return audio_clean.astype(np.float32)

# ====================== GUI ======================

class RajapintaStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("RAJAPINTA EEG 2 AUDIO STUDIO")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")
        
        self.raw = None
        self.fs = None
        self.channels = []
        self.preview_audio = None
        self.temp_wav = None
        
        self._build_ui()
    
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg="#16213e", height=50)
        top.pack(fill="x", side="top")
        
        tk.Label(top, text="RAJAPINTA EEG → AUDIO STUDIO", font=("Segoe UI", 16, "bold"), 
                 fg="#00ffcc", bg="#16213e").pack(side="left", padx=15, pady=8)
        
        # Main container
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill="both", expand=True, padx=10, pady=5)
        
        # LEFT PANEL - Controls
        left = tk.Frame(main, bg="#0f0f23", width=320)
        left.pack(side="left", fill="y", padx=(0, 8))
        
        # === EEG Loader ===
        ttk.Label(left, text="EEG FILE", font=("Segoe UI", 11, "bold"), foreground="#00ffcc").pack(pady=(10, 2))
        
        self.load_btn = ttk.Button(left, text="📂 LOAD EDF FILE", command=self.load_edf, width=28)
        self.load_btn.pack(pady=4)
        
        self.file_label = ttk.Label(left, text="No file loaded", foreground="#888888", wraplength=280)
        self.file_label.pack(pady=2)
        
        # Channel
        ttk.Label(left, text="CHANNEL / ELECTRODE", font=("Segoe UI", 10, "bold"), foreground="#00ffcc").pack(pady=(12, 2))
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(left, textvariable=self.channel_var, state="disabled", width=26)
        self.channel_combo.pack(pady=2)
        
        # Time range
        ttk.Label(left, text="TIME RANGE (seconds)", font=("Segoe UI", 10, "bold"), foreground="#00ffcc").pack(pady=(12, 2))
        
        time_frame = tk.Frame(left, bg="#0f0f23")
        time_frame.pack()
        
        ttk.Label(time_frame, text="Start:").grid(row=0, column=0, sticky="w")
        self.start_var = tk.DoubleVar(value=0.0)
        ttk.Entry(time_frame, textvariable=self.start_var, width=8).grid(row=0, column=1, padx=4)
        
        ttk.Label(time_frame, text="End:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.end_var = tk.DoubleVar(value=30.0)
        ttk.Entry(time_frame, textvariable=self.end_var, width=8).grid(row=0, column=3, padx=4)
        
        # Speed
        ttk.Label(left, text="SPEEDUP", font=("Segoe UI", 10, "bold"), foreground="#00ffcc").pack(pady=(10, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(left, from_=0.25, to=4.0, variable=self.speed_var, orient="horizontal", length=260)
        speed_scale.pack()
        self.speed_label = ttk.Label(left, text="1.0×")
        self.speed_label.pack()
        speed_scale.configure(command=lambda v: self.speed_label.config(text=f"{float(v):.2f}×"))
        
        # === INVERSE COCHLEA SETTINGS ===
        ttk.Label(left, text="INVERSE COCHLEA", font=("Segoe UI", 11, "bold"), foreground="#ffaa00").pack(pady=(18, 4))
        
        ttk.Label(left, text="Resonators (prime-log arms)", foreground="#cccccc").pack(anchor="w", padx=8)
        self.res_var = tk.IntVar(value=32)
        res_scale = ttk.Scale(left, from_=8, to=64, variable=self.res_var, orient="horizontal", length=260)
        res_scale.pack()
        self.res_label = ttk.Label(left, text="32")
        self.res_label.pack()
        res_scale.configure(command=lambda v: self.res_label.config(text=str(int(float(v)))))
        
        ttk.Label(left, text="Synthesis Type", foreground="#cccccc").pack(anchor="w", padx=8, pady=(8, 2))
        
        self.synth_var = tk.StringVar(value="sine")
        synth_frame = tk.Frame(left, bg="#0f0f23")
        synth_frame.pack()
        
        ttk.Radiobutton(synth_frame, text="Sine (clean vocal)", variable=self.synth_var, value="sine").pack(anchor="w")
        ttk.Radiobutton(synth_frame, text="Sawtooth (80s robot)", variable=self.synth_var, value="saw").pack(anchor="w")
        ttk.Radiobutton(synth_frame, text="Hybrid (sine + saw)", variable=self.synth_var, value="hybrid").pack(anchor="w")
        
        ttk.Label(left, text="Denoise Strength", foreground="#cccccc").pack(anchor="w", padx=8, pady=(10, 2))
        self.denoise_var = tk.DoubleVar(value=2.2)
        denoise_scale = ttk.Scale(left, from_=0.0, to=4.0, variable=self.denoise_var, orient="horizontal", length=260)
        denoise_scale.pack()
        self.denoise_label = ttk.Label(left, text="2.2")
        self.denoise_label.pack()
        denoise_scale.configure(command=lambda v: self.denoise_label.config(text=f"{float(v):.1f}"))
        
        # === ACTION BUTTONS ===
        action_frame = tk.Frame(left, bg="#0f0f23")
        action_frame.pack(pady=20, fill="x")
        
        self.preview_btn = ttk.Button(action_frame, text="🔄 GENERATE PREVIEW", 
                                       command=self.generate_preview, width=28, state="disabled")
        self.preview_btn.pack(pady=4)
        
        self.play_btn = ttk.Button(action_frame, text="▶ PLAY PREVIEW", 
                                    command=self.play_preview, width=28, state="disabled")
        self.play_btn.pack(pady=4)
        
        self.save_btn = ttk.Button(action_frame, text="💾 SAVE AS WAV...", 
                                    command=self.save_wav, width=28, state="disabled")
        self.save_btn.pack(pady=4)
        
        # CENTER - Preview plots
        center = tk.Frame(main, bg="#1a1a2e")
        center.pack(side="left", fill="both", expand=True)
        
        self.fig = Figure(figsize=(9, 6), facecolor="#0f0f23")
        self.fig.patch.set_facecolor("#0f0f23")
        
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_facecolor("#0f0f23")
        self.ax1.set_title("Waveform Preview", color="#00ffcc", fontsize=11)
        self.ax1.tick_params(colors="#888888")
        
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_facecolor("#0f0f23")
        self.ax2.set_title("Spectrogram Preview", color="#00ffcc", fontsize=11)
        self.ax2.tick_params(colors="#888888")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Status bar
        self.status = tk.Label(self.root, text="Ready — Load an EDF file to begin", 
                               bg="#16213e", fg="#00ffcc", anchor="w", font=("Segoe UI", 9))
        self.status.pack(fill="x", side="bottom")
    
    def load_edf(self):
        path = filedialog.askopenfilename(
            title="Select EDF file",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if not path:
            return
        
        try:
            self.status.config(text="Loading EDF...")
            self.root.update()
            
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            self.raw = raw
            self.fs = int(raw.info["sfreq"])
            self.channels = raw.ch_names
            
            self.file_label.config(text=os.path.basename(path))
            self.channel_combo.config(state="normal", values=self.channels)
            self.channel_var.set(self.channels[0] if self.channels else "")
            
            # Auto-set reasonable end time
            total_sec = len(raw.times)
            self.end_var.set(min(60.0, total_sec))
            
            self.preview_btn.config(state="normal")
            self.status.config(text=f"Loaded: {os.path.basename(path)}  |  {len(self.channels)} channels  |  {total_sec:.1f}s")
            
        except Exception as e:
            messagebox.showerror("Error loading EDF", str(e))
            self.status.config(text="Error loading file")
    
    def generate_preview(self):
        if self.raw is None:
            messagebox.showwarning("No EEG loaded", "Please load an EDF file first.")
            return
        
        try:
            ch = self.channel_var.get()
            start = float(self.start_var.get())
            end = float(self.end_var.get())
            speed = float(self.speed_var.get())
            n_res = int(self.res_var.get())
            synth = self.synth_var.get()
            denoise = float(self.denoise_var.get())
            
            if start >= end:
                messagebox.showerror("Invalid time", "Start time must be before end time.")
                return
            
            self.status.config(text="Synthesizing... please wait")
            self.root.update()
            
            # Extract data
            start_idx = int(start * self.fs)
            end_idx = int(end * self.fs)
            data = self.raw.get_data(picks=[ch], start=start_idx, stop=end_idx)[0].astype(np.float32)
            
            if len(data) == 0:
                messagebox.showerror("Empty slice", "No data in selected time range.")
                return
            
            # Z-score
            cal_len = min(1000, len(data))
            data = (data - np.mean(data[:cal_len])) / (np.std(data[:cal_len]) + 1e-6)
            
            # Synthesize
            audio_fs = 16000
            chunk_size = int(self.fs * 0.05)
            audio_chunk = int((audio_fs * 0.05) / speed)
            
            synth_engine = PrimeLogAntiCochlea(n_res, self.fs, audio_fs, synth)
            
            blocks = []
            for pos in range(0, len(data), chunk_size):
                chunk = data[pos:pos+chunk_size]
                if len(chunk) < chunk_size:
                    break
                block = synth_engine.process_and_synthesize(chunk, audio_chunk)
                blocks.append(block)
            
            raw_audio = np.concatenate(blocks)
            
            # Denoise if requested
            if denoise > 0.05:
                clean_audio = spectral_subtraction_denoise(raw_audio, sr=audio_fs, over_sub=denoise)
            else:
                clean_audio = raw_audio
            
            # Normalize
            peak = np.max(np.abs(clean_audio)) + 1e-6
            self.preview_audio = clean_audio / peak * 0.96
            
            # Update plots
            self._update_plots(self.preview_audio, audio_fs)
            
            self.play_btn.config(state="normal")
            self.save_btn.config(state="normal")
            self.status.config(text=f"Preview ready — {len(self.preview_audio)/audio_fs:.1f}s audio generated")
            
        except Exception as e:
            messagebox.showerror("Synthesis error", str(e))
            self.status.config(text="Error during synthesis")
    
    def _update_plots(self, audio, sr):
        self.ax1.clear()
        self.ax2.clear()
        
        t = np.linspace(0, len(audio)/sr, len(audio))
        
        # Waveform
        self.ax1.plot(t, audio, color="#00ffcc", linewidth=0.6)
        self.ax1.set_facecolor("#0f0f23")
        self.ax1.set_title("Waveform Preview", color="#00ffcc", fontsize=11)
        self.ax1.set_xlabel("Time (s)", color="#888888")
        self.ax1.tick_params(colors="#888888")
        self.ax1.grid(True, alpha=0.2, color="#444444")
        
        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(audio, fs=sr, nperseg=512, noverlap=384)
        self.ax2.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
        self.ax2.set_facecolor("#0f0f23")
        self.ax2.set_title("Spectrogram Preview", color="#00ffcc", fontsize=11)
        self.ax2.set_xlabel("Time (s)", color="#888888")
        self.ax2.set_ylabel("Frequency (Hz)", color="#888888")
        self.ax2.tick_params(colors="#888888")
        self.ax2.set_ylim(0, 5000)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def play_preview(self):
        if self.preview_audio is None:
            return
        
        try:
            # Save temp WAV
            if self.temp_wav and os.path.exists(self.temp_wav):
                os.remove(self.temp_wav)
            
            self.temp_wav = os.path.join(tempfile.gettempdir(), "rajapinta_preview.wav")
            sf.write(self.temp_wav, self.preview_audio, 16000)
            
            # Play (Windows)
            if os.name == "nt":
                os.startfile(self.temp_wav)
            else:
                import subprocess
                subprocess.Popen(["aplay", self.temp_wav] if os.path.exists("/usr/bin/aplay") else ["xdg-open", self.temp_wav])
            
            self.status.config(text="Playing preview...")
            
        except Exception as e:
            messagebox.showerror("Playback error", str(e))
    
    def save_wav(self):
        if self.preview_audio is None:
            return
        
        path = filedialog.asksaveasfilename(
            title="Save as WAV",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        if not path:
            return
        
        try:
            sf.write(path, self.preview_audio, 16000)
            messagebox.showinfo("Saved", f"Audio saved to:\n{path}")
            self.status.config(text=f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

# ====================== MAIN ======================

if __name__ == "__main__":
    root = tk.Tk()
    app = RajapintaStudio(root)
    root.mainloop()