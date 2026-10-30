"""单输出流播放器：A/B 共用样本位置，切换不会重新装载或从头播放。"""
import threading
import numpy as np
import sounddevice as sd


class ComparisonPlayer:
    def __init__(self):
        self.tracks = []
        self.sr = 44100
        self.position = 0
        self.selected = 0
        self.volume = .7
        self.playing = False
        self.stream = None
        self.lock = threading.Lock()
        self.underflows = 0
        self.frames_written = 0
        self.levels = np.zeros(2, dtype=np.float32)
        self.loop = None          # (起点帧, 终点帧)；播放到终点回到起点
        self.delta = False        # 差值监听：当前版本减去修音前

    @property
    def frames(self):
        return len(self.tracks[0]) if self.tracks else 0

    def load(self, tracks, sr):
        self.clear()
        self.tracks = tracks
        self.sr = sr

    def clear(self):
        self.pause()
        if self.stream:
            self.stream.close()
            self.stream = None
        self.tracks = []
        self.position = 0
        self.selected = 0
        self.loop = None
        self.delta = False

    def select(self, index):
        with self.lock:
            self.selected = index

    def replace_segment(self, index, start, samples):
        """原位替换草稿片段，短淡化避免切换时出现突变。"""
        with self.lock:
            if not self.tracks or index >= len(self.tracks):
                return
            track = self.tracks[index]
            start = max(0, min(int(start), len(track)))
            count = min(len(samples), len(track) - start)
            if count <= 0:
                return
            updated = np.asarray(samples[:count], dtype=np.float32).copy()
            fade = min(round(self.sr * .02), count // 2)
            if fade:
                ramp = np.linspace(0., 1., fade, dtype=np.float32)[:, None]
                updated[:fade] = track[start:start+fade] * (1-ramp) + updated[:fade] * ramp
                updated[-fade:] = track[start+count-fade:start+count] * ramp + updated[-fade:] * (1-ramp)
            track[start:start+count] = updated

    def seek(self, seconds):
        with self.lock:
            self.position = min(self.frames, max(0, round(seconds*self.sr)))

    def set_loop(self, start, end):
        with self.lock:
            a, b = (min(self.frames, max(0, round(x*self.sr))) for x in (start, end))
            self.loop = (a, b) if b - a >= self.sr * .05 else None

    def play(self):
        if not self.tracks:
            return
        if self.stream is None:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype='float32',
                                          latency='high', callback=self._callback)
        with self.lock:
            if self.position >= self.frames:
                self.position = 0
            self.playing = True
        try:
            self.stream.start()
        except Exception:
            self.playing = False
            self.stream.close()
            self.stream = None
            raise

    def pause(self):
        self.playing = False
        self.levels = np.zeros(2, dtype=np.float32)
        if self.stream:
            self.stream.abort()

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        if status.output_underflow:
            self.underflows += 1
        with self.lock:
            if not self.playing:
                return
            written = 0
            while written < frames:
                end = self.frames
                if self.loop and self.loop[0] <= self.position <= self.loop[1]:
                    if self.position == self.loop[1]:
                        self.position = self.loop[0]
                    end = self.loop[1]
                count = min(frames-written, end-self.position)
                if count <= 0:
                    self.playing = False
                    break
                part = self.tracks[self.selected][self.position:self.position+count]
                if self.delta and self.selected:
                    part = part - self.tracks[0][self.position:self.position+count]
                np.multiply(part, self.volume, out=outdata[written:written+count])
                self.position += count
                written += count
            # 电平表读取实际送往声卡的样本峰值（含音量）。
            self.levels = np.abs(outdata[:written]).max(axis=0) if written else np.zeros(2, dtype=np.float32)
            self.frames_written += written
