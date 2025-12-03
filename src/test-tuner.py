import sounddevice as sd
import numpy as np

# sounddevice config
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


def list_devices():
    print(sd.query_devices())
    

def ascertain_pitch_fft(audio_block):
    fft_data = np.fft.fft(audio_block)
    magnitude = np.abs(fft_data)
    nyquist_adjusted_magnitude = magnitude[:len(magnitude)//2]
    return (np.argmax(nyquist_adjusted_magnitude[1:]) + 1) * SAMPLE_RATE / BLOCK_SIZE


def acf_fft(audio_block):
    padded_length = 2 ** int(np.ceil(np.log2(2 * BLOCK_SIZE)))
    x_fft = np.fft.rfft(audio_block, n=padded_length)
    spectral_prod = x_fft * np.conj(x_fft)
    acf_full = np.fft.irfft(spectral_prod, n=padded_length)
    tau_max = min(BLOCK_SIZE // 2, MAX_TAU)

    return acf_full[1:tau_max + 1]


def absolute_threshold(cmn_df, tau_thres=0.1):
    tau = 1
    while tau < MAX_TAU:
        if cmn_df[tau] < tau_thres:
            # if the next tau isn't out of bounds and the cmn_df value for next tau is smaller, advance
            while tau + 1 < MAX_TAU and cmn_df[tau + 1] < cmn_df[tau]:
                tau += 1
            return tau
        tau += 1
    return -1

def find_global_min(cmn_df):
    return np.argmin(cmn_df)

def parabolic_interpolation(cmn_df, tau):
    p = cmn_df[tau - 1]
    q = cmn_df[tau]
    r = cmn_df[tau + 1] if tau + 1 < MAX_TAU else q

    return tau + ((p - r) / (2 * (p - 2 * q + r)))


def ascertain_pitch_yin(audio_block):
    # some range of lag values tau
    tau_max = min(BLOCK_SIZE // 2, MAX_TAU)
    window = BLOCK_SIZE # why was this divided by 2?
    # calculate df values
    audio_block_sq = audio_block ** 2
    sum_sq = np.cumsum(audio_block_sq)
    raw_df = np.zeros(MAX_TAU)
    acf_taus = acf_fft(audio_block) # yields acf values from tau values 1 ... MAX_TAU
    for tau in range(1, MAX_TAU):
        # energy_zero = audio_block_sq[1:window - tau] 
        energy_zero = sum_sq[window - tau - 1] - sum_sq[0]
        # energy_tau = audio_block_sq[tau:window]
        energy_tau = sum_sq[window - 1] - sum_sq[tau - 1]
        raw_df[tau] = energy_zero + energy_tau - 2 * acf_taus[tau - 1]
    # cumulative mean normalization
    cmn_df = np.ones(MAX_TAU)
    cmn_df[1:MAX_TAU] = raw_df[1:] * np.arange(1, MAX_TAU) / np.cumsum(raw_df[1:]).astype(float)
    # threshold search
    tau_candidate = absolute_threshold(cmn_df)
    global_min_tau = find_global_min(cmn_df)
    tau_final = tau_candidate if tau_candidate != -1 else global_min_tau
    # parabolic interpolation
    interpolated_tau = parabolic_interpolation(cmn_df, tau_final)

    # maybe do best local estimate

    return SAMPLE_RATE / interpolated_tau


def start_tuning():
    '''
    Test if input is working as desired
    input switching: DAC input vs microphone input
    run the tuner (probably need a gui for this shit)
    implement closest note finder to display on screen ex) A=440hz
    '''
    list_devices()

    buffers = []
    pitch_his = PitchTrackerEMA()
    with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=DEVICE_INDEX,
            channels=CHANNELS,
            dtype=DTYPE
    ) as stream:
        while True:
            audio_data, overflow = stream.read(BLOCK_SIZE)
            if overflow:
                print("audio overflow detected; block_size may be too small, processing may be too slow")

            audio_data_mono = audio_data.mean(axis=1)
            freq = ascertain_pitch_yin(audio_data_mono)
            smoothed_freq = pitch_his.update(freq)
            print(f'{smoothed_freq:8.2f}Hz', end='\r')

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
    

def test_tuner(tgt_freq, duration):
    signal = generate_signal(tgt_freq, duration)
    freq = ascertain_pitch_yin(signal[:BLOCK_SIZE])
    print(f'detected frequency: {freq}Hz, target freq is {tgt_freq}')



if __name__ == '__main__':
    if sd.query_devices() is None:
        print("no input device detected")
        exit()
    start_tuning()
    # test_tuner(670, 4)


