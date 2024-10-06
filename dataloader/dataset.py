import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
from whisper import whisper
from whisper.whisper.tokenizer import get_tokenizer
import torchaudio
import torchaudio.transforms as at
import random
import glob
import wave

import soundfile as sf
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import pad
from utils.data_augmentation.simulate_clipping import simulate_clipping_tensor
from scipy.signal import oaconvolve as ao

from pydub import AudioSegment, silence
from pydub.generators import WhiteNoise
# util

def load_wave(wave_path, sample_rate: int = 16000, pad_ms: int = 0) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)

    # Calculate the number of samples to pad based on the sample rate and padding in milliseconds
    pad_samples = int(sample_rate * (pad_ms / 1000.0))

    # Pad the waveform on both sides with the calculated number of samples
    # The pad function takes a tuple that represents the padding on each side of the last dimension
    # Since audio waveforms are typically shaped (channels, samples), we want to pad the second dimension
    if pad_samples > 0:
        waveform = pad(waveform, (pad_samples, pad_samples), "constant", 0)

    return waveform

def convert_floats(value):
    try:
        if '.' in value and float(value).is_integer():
            return str(int(float(value)))
        else:
            return value
    except ValueError:
        return value


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, db_file="fb_enhanced.csv", 
                 rand=True, 
                 text_max_length=120, 
                 supported_snrs=['Q', '8', '6', '4', '2'], 
                 enhancement_loss=False, 
                 train=False,
                 packet_loss=False,
                 clipping=False,
                 reverb=False, 
                 white=False, 
                 pub=False, 
                 rir_dirs=["/home/datasets/public/sim_rir_16k/simulated_rirs_16k"]): #, "/home/data/shua/simulated_RIRs"]):
        self.train = train
        if not self.train:
            supported_snrs=['Q', '8', '6', '4', '2']
        # load list of files
        if db_file.endswith(".parquet"):
            db_df = pd.read_parquet(db_file)
        elif db_file.endswith(".csv"):
            db_df = pd.read_csv(db_file)
        else:
            print("unsupported data format")
            exit(1)
        self.enhancement_loss = enhancement_loss
        supported_snrs_list = [item for sublist in [s.split() for s in supported_snrs] for item in sublist]

        self.sample_rate = 16000    
        self.db_df = db_df.copy()
        self.db_df['SNR'] = self.db_df['SNR'].apply(convert_floats)
        self.db_df = self.db_df[self.db_df['target'].str.len() <= text_max_length]
        # Filter rows based on supported SNRs, including 'Q'
        self.db_df = self.db_df[self.db_df['SNR'].astype(str).isin(supported_snrs_list)]

        self.rand = rand
        self.gpt2_tokenizer = get_tokenizer(multilingual=False)
        woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
        self.multilingual_tokenizer = get_tokenizer(multilingual=True, language="en", task=woptions.task)
        # self.drop_frequency = 0.2  # 5% chance of frame drop at any given frame
        self.span_distribution = torch.tensor([0.4, 0.2, 0.1, 0.1, 0.07, 0.05, 0.03, 0.02, 0.02, 0.01])  # Probability distribution for the span
        self.packet_loss = packet_loss
        self.clipping    = clipping
        self.reverb      = reverb
        self.white       = white
        self.pub         = pub
        self.rirs = []
        for dir_path in rir_dirs:
            # Use glob to list all .wav files recursively in each augmentation directory
            self.rirs.extend(glob.glob(os.path.join(dir_path, '**', '*.wav'), recursive=True))
        # Ensure only files are listed (this is already ensured by the '*.wav' pattern)
        self.rirs = [f for f in self.rirs if os.path.isfile(f)]
        self.loaded_noise_samples = []
        self.load_noise_samples("/home/datasets/public/youtube/ambient_noise")

    def __len__(self):
        return self.db_df.shape[0]


    # Function to load noise samples
    def load_noise_samples(self, noise_directory):
        for filename in os.listdir(noise_directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(noise_directory, filename)
                noise_waveform = load_wave(filepath)
                self.loaded_noise_samples.append(noise_waveform)
        # print(f"Loaded {len(loaded_noise_samples)} noise files.")


    def get_random_noise_sample(self, length):
        """
        Get a random noise sample of the specified length from the loaded noise files.
        This function now correctly handles multi-channel (stereo, etc.) audio.
        """
        if not self.loaded_noise_samples:
            raise ValueError("No noise samples loaded. Please load samples first.")

        # Randomly select a noise sample
        selected_sample = random.choice(self.loaded_noise_samples)

        # Ensure the selected sample is long enough by repeating it if necessary
        while selected_sample.shape[1] < length:  # Check the second dimension for the length
            selected_sample = torch.cat([selected_sample, selected_sample], dim=1)  # Concatenate along the time axis

        # Randomly select a start index for the noise segment
        start_index = random.randint(0, selected_sample.shape[1] - length)
        # Extract the segment ensuring all channels are included
        return selected_sample[:, start_index:start_index + length]

    def pad_or_trim(self, tokens, max_length=50, padding_token_id=0):
        """Pad or trim the token sequence to the desired length."""
        if len(tokens) > max_length:
            return tokens[:max_length]
        elif len(tokens) < max_length:
            return tokens + [padding_token_id] * (max_length - len(tokens))
        else:
            return tokens
        
    def simulate_packet_loss_pytorch(self, mel_spectrum, drop_frequency):
        """
        Simulate packet loss in a mel spectrum tensor.
        
        :param mel_spectrum: A PyTorch tensor of shape (80, 3000) representing the mel spectrum.
        :param drop_frequency: Probability of a frame drop occurring at any given frame.
        :param span_distribution: A list or array representing the probability distribution of the span of frame drops.
        :return: Modified mel spectrum with simulated packet loss.
        """
        num_channels, num_frames = mel_spectrum.shape

        for frame in range(num_frames):
            if torch.rand(1).item() < drop_frequency:
                drop_span = torch.multinomial(self.span_distribution, 1).item()
                end_frame = min(frame + drop_span, num_frames)
                mel_spectrum[:, frame:end_frame] = 0

        return mel_spectrum

    def __getitem__(self, i):
        # Noise order: Reverberation -> white & or Pub -> Clipping -> Packet-loss
        row = self.db_df.iloc[i]
        audio = load_wave(row['audio'], sample_rate=self.sample_rate)
        # print(audio.shape)
        # Calculate the power (squared RMS) of the target audio
        audio_power = audio_power = torch.mean(audio.pow(2))
        if self.reverb:
            if random.choices([0,1], weights=[60,40]):
                # Load the Room Impulse Response (RIR) audio
                rir_file = random.choice(self.rirs)
                room, _   = sf.read(rir_file)
                reverberated_audio = ao(audio.numpy()[0], room, mode='full')
                audio = torch.from_numpy(reverberated_audio).float()

        if self.train and (self.white or self.pub):
            if random.choices([0,1], weights=[60,40]):
                rng = np.random.default_rng()  # Using default random generator which is recommended.
                if self.white:
                    noise_samples = torch.from_numpy(rng.normal(0, 1, audio.shape[-1])).float()
                elif self.pub:
                    noise_samples = self.get_random_noise_sample(audio.shape[-1])
                noise_power = torch.mean(noise_samples.pow(2))
                snr_ratio = float(random.choices([2.0, 4.0, 6.0, 8.0], weights=[25, 25, 25, 25])[0])  # Ensure SNR ratio is a float
                # Calculate the desired noise power based on the SNR
                desired_noise_power = 1
                if snr_ratio >= 0:
                    # Desired noise power for positive SNR: reduce noise power
                    desired_noise_power = audio_power / (10 ** (snr_ratio / 10))
                else:
                    # Desired noise power for negative SNR: increase noise power
                    # We use the absolute value of snr_ratio to calculate the increase
                    desired_noise_power = audio_power * (10 ** (abs(snr_ratio) / 10))

                # Calculate the scaling factor for adjusting the noise
                scaling_factor = desired_noise_power / noise_power

                # Apply the scaling factor to adjust the noise level
                adjusted_noise = noise_samples * torch.sqrt(scaling_factor)
                # Add the scaled noise to the original audio
                audio = audio + adjusted_noise

        if self.clipping:
            clipping_rate = random.choices([0, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99], weights=[72, 2, 2, 2, 2, 10, 10])[0]
            audio = simulate_clipping_tensor(audio, clipping_rate)

        audio = whisper.pad_or_trim(audio.flatten())
        mels = whisper.log_mel_spectrogram(audio).squeeze()
        if self.packet_loss:
            dropout_rate = random.choices([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], weights=[60, 5, 5, 15, 5, 5, 5])[0]
            mels = self.simulate_packet_loss_pytorch(mels, dropout_rate)
        # mels = whisper.log_mel_spectrogram(row['audio'])    
        trans = row['target']
        gpt2_tokens = self.gpt2_tokenizer.encode(trans)
        # multilingual_tokens = self.multilingual_tokenizer.encode(trans)
        fn = row['audio'].split('/')[-1]
        if self.enhancement_loss and 'orig_audio' in self.db_df.columns:
            clean_audio = load_wave(row['orig_audio'], sample_rate=self.sample_rate, pad_ms=500)
            clean_audio = whisper.pad_or_trim(clean_audio.flatten())
            clean_mels = whisper.log_mel_spectrogram(clean_audio).squeeze()
        elif self.packet_loss and self.enhancement_loss:
            clean_audio = load_wave(row['audio'], sample_rate=self.sample_rate)
            clean_audio = whisper.pad_or_trim(clean_audio.flatten())
            clean_mels = whisper.log_mel_spectrogram(clean_audio).squeeze()
        else:
            clean_mels = mels
        multilingual_tokens = [*self.multilingual_tokenizer.sot_sequence_including_notimestamps] + self.multilingual_tokenizer.encode(trans)
        labels = multilingual_tokens[1:] + [self.multilingual_tokenizer.eot]
        # Pad or trim token sequences
        # gpt2_tokens = self.pad_or_trim(gpt2_tokens)
        # multilingual_tokens = self.pad_or_trim(multilingual_tokens)
        
        return {"mels": mels.unsqueeze(dim=0),
                "clean_mels": clean_mels.unsqueeze(dim=0),
                "gpt2_tokens": gpt2_tokens, 
                "labels": labels, 
                "multilingual_tokens": multilingual_tokens}

class WhisperDataCollatorWithPadding:
    def __call__(sefl, features):
        mels, clean_mels, labels, multilingual_tokens = [], [], [], []
        for f in features:
            mels.append(f["mels"])
            clean_mels.append(f["clean_mels"])
            labels.append(f["labels"])
            multilingual_tokens.append(f["multilingual_tokens"])

        mels = torch.concat([mel[None, :] for mel in mels])
        clean_mels = torch.concat([mel[None, :] for mel in clean_mels])
        
        label_lengths = [len(lab) for lab in labels]
        multilingual_tokens_length = [len(e) for e in multilingual_tokens]
        max_label_len = max(label_lengths+multilingual_tokens_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        multilingual_tokens = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(multilingual_tokens, multilingual_tokens_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "multilingual_tokens": multilingual_tokens
        }
        
        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["mels"] = mels
        batch["clean_mels"] = clean_mels

        return batch
