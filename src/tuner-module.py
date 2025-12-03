import numpy as np

SAMPLE_RATE = 44100
CHANNELS = 2
BLOCK_SIZE = 4096 # amount of audio data read from stream per iteration.
# block size makes the tuner more granular in frequency detection. Due to the nature of FFT, we need more block size for higher granularity.
# This issue has been fixed with the YIN method and specifically their parabolic interpolation of the CMNDF minima. 
DEVICE_INDEX = 'default'
DTYPE = 'int16'

# for the YIN method and Pitch tracker
MAX_TAU = SAMPLE_RATE // 40
ALPHA = 0.25

class PitchTrackerEMA:
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.stable_pitch = 0.0

    def update(self, new_pitch):
        if new_pitch <= 0.0:
            self.stable_pitch *= (1.0 - self.alpha * 0.5)
            if self.stable_pitch  < 1.0:
                self.stable_pitch = 0.0
            return self.stable_pitch
        if self.stable_pitch == 0.0:
            self.stable_pitch = new_pitch
        else:
            self.stable_pitch = self.alpha * new_pitch + (1 - self.alpha) * self.stable_pitch

        return self.stable_pitch

class PitchDetector:
    def __init__(self, sample_rate, window_size, min_freq, tau_thres=0.1):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.min_freq = min_freq
        self.tau_thres = tau_thres
        self.max_tau = self.sample_rate // self.min_freq

    # method for getting the pitch with Fourier transform (basic, intuitive, but restricted by window size)
    def get_pitch_fft(self, audio_block):
        fft_data = np.fft.fft(audio_block)
        magnitude = np.abs(fft_data)
        nyquist_adjs_mag = magnitude[:len(magnitude)//2]
        return (np.argmax(nyquist_adjs_mag[1:]) + 1) * self.sample_rate / self.window_size
    
    def _get_acf_fft(self, audio_block):
        padded_length = 2 ** int(np.ceil(np.log2(2 * self.window_size)))
        x_fft = np.fft.rfft(audio_block, n=padded_length)
        spectral_prod = x_fft * np.conj(x_fft)
        acf_full = np.fft.irfft(spectral_prod, n=padded_length)
        tau_max = min(self.window_size // 2, self.max_tau)
        return acf_full[1:tau_max + 1]

    def _get_absolute_threshold(self, cmn_df):
        tau = 1
        while tau < self.max_tau:
            if cmn_df[tau] < self.tau_thres:
                while tau + 1 < self.max_tau and cmn_df[tau + 1] < cmn_df[tau]:
                    tau += 1
                return tau
            tau += 1
        return -1

    def _find_global_min(self, cmn_df):
        return np.argmin(cmn_df)
    
    def _interpolate(self, cmn_df, tau):
        p = cmn_df[tau - 1]
        q = cmn_df[tau]
        r = cmn_df[tau + 1] if tau + 1 < self.max_tau else q
        return tau + ((p - r) / (2 * (p - 2 * q + r)))

    def get_pitch_yin(self, audio_block):
        tau_max = min(self.window_size // 2, self.max_tau)
        audio_block_sq = audio_block ** 2
        sum_sq = np.cumsum(audio_block_sq)
        raw_df = np.zeros(self.max_tau)
        acf_taus = self._get_acf_fft(audio_block)
        for tau in range(1, self.max_tau):
            energy_0 = sum_sq[self.window_size - tau - 1] - sum_sq[0]
            energy_tau = sum_sq[self.window_size - 1] - sum_sq[tau - 1]
            raw_df[tau] = energy_0 + energy_tau - 2 * acf_taus[tau - 1]
        cmn_df = np.ones(self.max_tau)
        cmn_df[1:self.max_tau] = raw_df[1:] * np.arange(1, self.max_tau) / np.cumsum(raw_df[1:]).astype(float)
        tau_candidate = self._get_absolute_threshold(cmn_df)
        global_min_tau = self._find_global_min(cmn_df)
        tau_final = tau_candidate if tau_candidate != -1 else global_min_tau
        interpolated_tau = self._interpolate(cmn_df, tau_final)

        return self.sample_rate / interpolated_tau

def generate_signal(tgt_freq, duration, decay_rate=-1):
    # duration in seconds
    num_samples = int(SAMPLE_RATE * duration)
    x = np.linspace(0, duration, num=num_samples, endpoint=False)
    signal = np.sin(np.pi * 2 * tgt_freq * x)
    if decay_rate < 0:
        return signal
    else:
        envelope = np.exp(-decay_rate * (x / duration))
        return envelope * signal
 

if __name__ == '__main__':

    tuner = PitchDetector(SAMPLE_RATE, BLOCK_SIZE, 40)
    signal = generate_signal(440, 2, 3)
    print(tuner.get_pitch_yin(signal[:BLOCK_SIZE]))





