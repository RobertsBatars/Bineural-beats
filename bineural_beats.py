import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import threading

class BinauralBeatsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Binaural Beats Generator")
        self.is_playing = False
        self.stream = None
        self.thread = None
        self.sr = 44100
        self.duration = None  # None means play until stopped
        self.volume = tk.DoubleVar(value=0.5)
        self.base_freq = tk.DoubleVar(value=200.0)
        self.beat_freq = tk.DoubleVar(value=10.0)
        self.mode = tk.StringVar(value="constant")
        self._cached_params = None  # To store cached values for audio thread
        self._volume_lock = threading.Lock()
        self._thread_volume = self.volume.get()
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid()
        ttk.Label(frm, text="Base Frequency (Hz):").grid(column=0, row=0, sticky="w")
        ttk.Entry(frm, textvariable=self.base_freq, width=10).grid(column=1, row=0)
        ttk.Label(frm, text="Beat Frequency (Hz):").grid(column=0, row=1, sticky="w")
        ttk.Entry(frm, textvariable=self.beat_freq, width=10).grid(column=1, row=1)
        ttk.Label(frm, text="Volume:").grid(column=0, row=2, sticky="w")
        self.volume_slider = ttk.Scale(frm, from_=0, to=1, orient="horizontal", variable=self.volume)
        self.volume_slider.grid(column=1, row=2)
        self.volume_slider.bind('<ButtonRelease-1>', self.on_volume_release)
        ttk.Label(frm, text="Mode:").grid(column=0, row=3, sticky="w")
        ttk.Radiobutton(frm, text="Constant Delta", variable=self.mode, value="constant").grid(column=1, row=3, sticky="w")
        ttk.Radiobutton(frm, text="Alternating Delta", variable=self.mode, value="alternating").grid(column=1, row=4, sticky="w")
        self.toggle_btn = ttk.Button(frm, text="Start", command=self.toggle_play)
        self.toggle_btn.grid(column=0, row=5, columnspan=2, pady=10)

    def on_volume_release(self, event):
        # Update thread-safe volume for audio thread
        with self._volume_lock:
            self._thread_volume = self.volume.get()

    def toggle_play(self):
        if self.is_playing:
            self.stop()
        else:
            self.start()
        self.update_toggle_btn()

    def update_toggle_btn(self):
        if self.is_playing:
            self.toggle_btn.config(text="Stop")
        else:
            self.toggle_btn.config(text="Start")

    def start(self):
        if self.is_playing:
            return
        # Cache all needed values for the audio thread
        self._cached_params = {
            'base': float(self.base_freq.get()),
            'beat': float(self.beat_freq.get()),
            'mode': str(self.mode.get()),
            'sr': self.sr,
            'duration': self.duration  # Will be None for continuous
        }
        with self._volume_lock:
            self._thread_volume = self.volume.get()
        self.is_playing = True
        self.thread = threading.Thread(target=self._play)
        self.thread.start()
        self.update_toggle_btn()

    def stop(self):
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.update_toggle_btn()

    def _play(self):
        # Use cached parameters to avoid accessing Tkinter variables from audio thread
        params = self._cached_params
        base = params['base']
        beat = params['beat']
        mode = params['mode']
        sr = params['sr']
        duration = params['duration']
        frames_total = None if duration is None else int(sr * duration)
        self.phase_left = 0.0
        self.phase_right = 0.0
        self.t_sample = 0

        def callback(outdata, frames, time, status):
            # Only read thread-safe volume variable
            with self._volume_lock:
                vol = self._thread_volume
            if not self.is_playing:
                raise sd.CallbackStop()
            if mode == "constant":
                freq_left = base - beat/2
                freq_right = base + beat/2
                phase_inc_left = 2 * np.pi * freq_left / sr
                phase_inc_right = 2 * np.pi * freq_right / sr
                idx = np.arange(frames)
                left = np.sin(self.phase_left + phase_inc_left * idx)
                right = np.sin(self.phase_right + phase_inc_right * idx)
                self.phase_left += phase_inc_left * frames
                self.phase_right += phase_inc_right * frames
                self.phase_left = np.mod(self.phase_left, 2 * np.pi)
                self.phase_right = np.mod(self.phase_right, 2 * np.pi)
            elif mode == "alternating":
                t = (np.arange(frames) + self.t_sample)
                t_sec = t / sr
                mod = np.sin(2 * np.pi * beat * t_sec)
                # Instantaneous frequencies
                freq_left = base - (mod * beat/2)
                freq_right = base + (mod * beat/2)
                # Integrate phase for each channel
                phase_left = np.cumsum(2 * np.pi * freq_left / sr)
                phase_right = np.cumsum(2 * np.pi * freq_right / sr)
                left = np.sin(self.phase_left + phase_left)
                right = np.sin(self.phase_right + phase_right)
                # Update phase accumulators for continuity
                self.phase_left += phase_left[-1]
                self.phase_right += phase_right[-1]
                self.phase_left = np.mod(self.phase_left, 2 * np.pi)
                self.phase_right = np.mod(self.phase_right, 2 * np.pi)
            else:
                left = right = np.zeros(frames)
            stereo = np.stack([left, right], axis=1) * vol
            outdata[:frames] = stereo.astype(np.float32)
            self.t_sample += frames

        self.stream = sd.OutputStream(
            samplerate=sr, channels=2, callback=callback, blocksize=1024
        )
        with self.stream:
            # If frames_total is None, play until stopped
            if frames_total is None:
                while self.is_playing:
                    sd.sleep(100)
            else:
                while self.is_playing and self.t_sample < frames_total:
                    sd.sleep(100)
        self.t_sample = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = BinauralBeatsApp(root)
    root.mainloop()
