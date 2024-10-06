import os
from pydub import AudioSegment, silence
from pydub.generators import WhiteNoise
import math
import numpy as np
from tqdm import tqdm
import wave
import random
from data_augmentation.simulate_reveberation import apply_reverberation_with_rt60_and_random_conditions

# Global variable to store loaded noise samples
loaded_noise_samples = []

def load_noise_samples(directory):
    """
    Load all WAV files from the specified directory into memory.
    """
    global loaded_noise_samples
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            with wave.open(filepath, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                noise_sample = np.frombuffer(frames, dtype=np.int16)
                loaded_noise_samples.append(noise_sample)
    print(f"Loaded {len(loaded_noise_samples)} noise files.")

def get_random_noise_sample(length):
    """
    Get a random noise sample of the specified length from the loaded noise files.
    """
    if not loaded_noise_samples:
        raise ValueError("No noise samples loaded. Please load samples first.")

    # Randomly select a noise sample
    selected_sample = random.choice(loaded_noise_samples)

    # If the selected sample is shorter than the required length, repeat it
    while len(selected_sample) < length:
        selected_sample = np.tile(selected_sample, 2)

    # Extract a random segment of the required length
    start_index = random.randint(0, len(selected_sample) - length)
    return selected_sample[start_index:start_index + length]

def mix_audio_with_noise(audio_path, output_dir, snr, pre_silence_ms, post_silence_ms, type="white"):
    if snr != 'Q':
        # Calculate the scaling factor for noise to achieve the desired SNR
        snr = int(snr) 
    output_path = os.path.join(output_dir, os.path.basename(audio_path))#).replace('.wav', f'_SNR{snr}.wav'))
    if os.path.exists(output_path):
        return
    try:
        audio = AudioSegment.from_file(audio_path)
        # Generate random noise to match the length of the audio
        rng = np.random.default_rng()  # Using default random generator which is recommended.
        if type == "white":
            noise_samples = rng.normal(0, 1, len(audio.get_array_of_samples()))
        elif type == "pub":
            noise_samples = get_random_noise_sample(len(audio.get_array_of_samples()))
        else: 
            raise ValueError(f"Unsupported noise type {type}")
        # Convert the generated noise into an audio segment
        noise = AudioSegment(
            noise_samples.tobytes(), 
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width, 
            channels=1
        )
        
        if snr != 'Q':
            # Calculate the scaling factor for noise to achieve the desired SNR
            snr = float(snr)  # Convert SNR to float if it is not 'Q'
            audio_rms = audio.rms
            noise_rms = noise.rms if noise.rms > 0 else 1
            required_noise_rms = audio_rms / (10 ** (snr / 20))
            scaling_factor = required_noise_rms / noise_rms
            noise = noise.apply_gain(20 * math.log10(scaling_factor))
        else:
            # If 'Q' is provided as SNR, use audio without any noise.
            noise = AudioSegment.silent(duration=len(audio), frame_rate=audio.frame_rate)

        # Combining the audio and noise
        mixed = audio.overlay(noise)
        # Adding silence before and after audio
        mixed = AudioSegment.silent(duration=pre_silence_ms) + mixed + AudioSegment.silent(duration=post_silence_ms)

        # Saving the result
        mixed.export(output_path, format="wav")
    except:
        return

def process_directory(input_dir, output_dir, snr_list, pre_silence_ms, post_silence_ms, noise_type="white"):
    for dirpath, dirnames, filenames in tqdm(list(os.walk(input_dir))):
        for file in filenames:
            if file.endswith(".wav"):
                file_path = os.path.join(dirpath, file)

                # Create a corresponding path in the output directory.
                relative_path = os.path.relpath(dirpath, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                # Ensure the directory exists; if not, create it.
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                for snr in snr_list:
                    mix_audio_with_noise(file_path, output_subdir, snr, pre_silence_ms, post_silence_ms, noise_type)

def process_directory_pub_test(input_dir, output_dir, pre_silence_ms, post_silence_ms, noise_type="pub"):
    for dirpath, dirnames, filenames in tqdm(list(os.walk(input_dir))):
        for file in filenames:
            if file.endswith(".wav"):
                file_path = os.path.join(dirpath, file)

                # Create a corresponding path in the output directory.
                relative_path = os.path.relpath(dirpath, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                # Ensure the directory exists; if not, create it.
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                snr = file.split('.wav')[0].split("SNR")[1].replace("n", "-").replace('p', '').upper()
                mix_audio_with_noise(file_path, output_subdir, snr, pre_silence_ms, post_silence_ms, noise_type)

def reverb_directory(input_dir, output_dir, RT60_list):
    for dirpath, dirnames, filenames in tqdm(list(os.walk(input_dir))):
        for file in filenames:
            if file.endswith(".wav"):
                file_path = os.path.join(dirpath, file)
                # import ipdb; ipdb.set_trace()
                # Create a corresponding path in the output directory.
                relative_path = os.path.relpath(dirpath, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                # Ensure the directory exists; if not, create it.
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                for RT60 in RT60_list:
                    apply_reverberation_with_rt60_and_random_conditions(file_path, RT60, output_subdir)
            

if __name__ == "__main__":
    noise_type = "pub" # "reverb" "pub" "white" 
    if noise_type in ["pub", "white"]:
        input_directory = "/home/datasets/public/librispeech" #input("Enter the path to the input directory: ")
        # input_directory = "/home/datasets/public/voxpopuli/Shua" #input("Enter the path to the input directory: ")
        # input_directory = "/home/datasets/public/FLEURS/Shua/all" #input("Enter the path to the input directory: ")
        # noise_file = "Data/LTASmatched_noise.wav" #input("Enter the path to the noise file: ")
        output_directory = "/home/datasets/public/librispeech_random_noise" #input("Enter the path to the output directory: ")
        # output_directory = "/home/datasets/public/FLEURS/Shua/fleurs_random_noise" #input("Enter the path to the output directory: ")
        
        input_directory = "/home/mlspeech/shua/home/Shua/recipies/Whisper_denoiser/origin_data_16k/"
        output_directory = "/home/data/shua/DB_manipulations/allsstar_pub" 
        train_or_test = "test"
        if train_or_test == "train":
            noise_dir = "/home/datasets/public/youtube/ambient_noise"
        elif train_or_test == "test":
            noise_dir = "/home/datasets/public/youtube/pub"
        snr_levels = [0]# 2, 'Q', 4, 6, 8] #-4, -2, 0,
        pre_silence_duration = 0  # in milliseconds
        post_silence_duration = 0  # in milliseconds
        if noise_type == "pub":
            load_noise_samples(noise_dir)
            process_directory_pub_test(input_directory, output_directory, pre_silence_duration, post_silence_duration, noise_type)
        # process_directory(input_directory, output_directory, snr_levels, pre_silence_duration, post_silence_duration, noise_type)
    if noise_type == "reverb":
        input_directory = "/home/mlspeech/shua/home/Shua/recipies/Whisper_denoiser/origin_data_16k/"
        output_directory = "/home/data/shua/DB_manipulations/allsstar_reverb" 
        RT60_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        reverb_directory(input_directory, output_directory, RT60_list)