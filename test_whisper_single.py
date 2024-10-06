#%%
from whisper import whisper
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import torchmetrics
import re
import string 
import torch
from models.unet_whisper_lit import LitUnetWhisperModel
import os
import seaborn as sns 
import matplotlib.pyplot as plt
import evaluate
from whisper.whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from utils.tplcnet_utils import create_dropped_audio
from utils.data_augmentation.simulate_clipping import simulate_clipping_numpy
import scipy.io.wavfile 
import jiwer

asr = "0_50_distribution"
decode = True
run_unet = True
run_orig = False
packet_loss = False
clipping = False
reverb = False
white_noise = False
pub_noise = False
finetuned = False
generate_plots = False
testing_forward = False
allstar = False
multilingual = False
calculate_hallucination_rate = True
chime6 = True

exp = "multi_noise" #"libri_enhancement_loss_large" #
ckpt = "epoch=22-step=51390-v1"
model_name = "base" # "large-v2" # 
experiment_name = f"{model_name}_PL_ce1_l19_q_{ckpt}" #ce1_l19_q8642_epoch=5-step=201215"
lang = 'en'
device = "cuda" 
torch.manual_seed(42) 

def calculate_zeroed_out_percentage(mel_spectrum):
    """
    Calculate the percentage of zeroed-out rows in a mel spectrum tensor.

    :param mel_spectrum: A PyTorch tensor representing the mel spectrum.
    :return: Percentage of rows that are completely zeroed out.
    """
    num_rows = mel_spectrum.size(1)
    zeroed_rows = (mel_spectrum.sum(dim=0) == 0).sum().item()
    percentage = (zeroed_rows / num_rows) * 100
    return percentage

def simulate_packet_loss_pytorch(mel_spectrum, drop_frequency=0.4, span_distribution=torch.tensor([0.0, 0.4, 0.2, 0.1, 0.1, 0.07, 0.05, 0.03, 0.02, 0.02, 0.01])):
    """
    Simulate packet loss in a mel spectrum tensor such that the overall dropout rate approximates drop_frequency.
    
    :param mel_spectrum: A PyTorch tensor of shape (80, 3000) representing the mel spectrum.
    :param drop_frequency: Probability of a frame drop occurring at any given frame.
    :param span_distribution: A list or array representing the probability distribution of the span of frame drops.
    :return: Modified mel spectrum with simulated packet loss.
    """
    num_channels, num_frames = mel_spectrum.shape
    total_frames_to_drop = int(drop_frequency * num_frames)

    dropped_frames = torch.zeros(num_frames, dtype=torch.bool)

    while dropped_frames.sum() < total_frames_to_drop:
        frame = torch.randint(0, num_frames, (1,)).item()
        if not dropped_frames[frame]:  # Check if the frame is not already dropped
            drop_span = torch.multinomial(span_distribution, 1).item()
            end_frame = min(frame + drop_span, num_frames)
            dropped_frames[frame:end_frame] = True

    mel_spectrum[:, dropped_frames] = 0

    return mel_spectrum, dropped_frames

supported_snrs = ['Q', '8', '6', '4', '2', '0', '-2', '-4']
supported_RT60s = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# if white_noise or pub_noise: 
SNR_type = "SNR"
if reverb:
    SNR_type = "RT60"
if run_unet or finetuned:
    unet_w = LitUnetWhisperModel.load_from_checkpoint(f".exp/{exp}/ckpt/{ckpt}.ckpt").to(device)
    unet_w.eval()
    model = whisper.load_model(model_name).to(device) #unet_w.whisper
else:
    model = whisper.load_model(model_name).to(device)

compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
baseline_df = pd.read_csv("Data/intelligibility/whisper_openai_base_baseline.csv")
# baseline_df = pd.read_csv("Data/tiny_dbg.csv")
# baseline_df = pd.read_csv("Data/intelligibility/whisper_openai_large-v2_baseline.csv")
# baseline_df = pd.read_csv("Data/fleurs_all_test.csv")
# baseline_df = pd.read_csv("Data/fb_enhanced.csv")
# baseline_df = pd.read_csv("Data/PLC_challenge_blind.csv")
if pub_noise or packet_loss or clipping:
    baseline_df = pd.read_csv("Data/allstar_origin_16k.csv")
if reverb:
    baseline_df = pd.read_csv("Data/allsstar_rt60.csv")
if chime6:
    baseline_df = pd.read_csv("chime6_data/segments_fixed.csv")

baseline_df = baseline_df.copy()
# baseline_df = baseline_df.drop_duplicates(subset='lang_id')
if 'audio' not in baseline_df.columns:
    baseline_df.rename(columns={'path': 'audio'}, inplace=True)
if 'lang_id' not in baseline_df.columns:
    baseline_df['lang_id'] = 'en'
if 'SNR' not in baseline_df.columns:
    baseline_df['SNR'] = 'Q'
if 'RT60' not in baseline_df.columns:
    baseline_df['RT60'] = '0.0'

# baseline_df = baseline_df[baseline_df["lang_id"] == "fr_fr"] # "es_419", ru_ru, de_de
# print("fr_fr")
replace_dict = {    
    "rain coat":"raincoat",
    "mail man":"mailman",
    "police man":"policeman",
    "matchboxes":"match boxes",
    "hand stand":"handstand",
    "goal post":"goalpost",
    "fire truck":"firetruck",
    "saucepan":"sauce pan",
    "(^|\s)1($|\s)":" one ",
    "(^|\s)2($|\s)":" two ",
    "(^|\s)3($|\s)":" three ",
    "(^|\s)4($|\s)":" four ",
    "(^|\s)5($|\s)":" five ",
    "(^|\s)6($|\s)":" six ",
    "(^|\s)7($|\s)":" seven ",
    "(^|\s)8($|\s)":" eight ",
    "(^|\s)9($|\s)":" nine ",
    r"\s+":" "
}

contraction_dict = {
    r"(^|\s)its($|\s)": " it's ",
    r"(^|\s)were($|\s)": " we're ",
    r"(^|\s)youre($|\s)": " you're ",
    r"(^|\s)theyre($|\s)":  " they're ",
    r"(^|\s)im\'?($|\s)":  " i'm ",
    r"(^|\s)hes\'?($|\s)":  " he's ",
    r"(^|\s)she\'?($|\s)": " she's "
}



pub_noise_audio = whisper.load_audio("Data/restaurant08.wav")
# Calculate the power (squared RMS) of the noise audio
noise_power = np.mean(np.square(pub_noise_audio))
pub_noise_audio = whisper.pad_or_trim(pub_noise_audio)

z_count = 0
drop_frequencies = [0.0]
if packet_loss:
    drop_frequencies = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #[0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99] # 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99
if clipping:
    drop_frequencies = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
for drop_freq in drop_frequencies:
    actual_drop_freq = []
    now = datetime.now().strftime("%d_%m")
    if decode:
        file_name = f'whisper_openai_{model_name}_{now}_{experiment_name}.csv'
        for language, group_df in tqdm(baseline_df.groupby('lang_id'), desc="Processing languages"):
            for index, row in tqdm(group_df.iterrows(), total=group_df.shape[0]):
                if str(row['SNR']) not in supported_snrs:
                    print(f"skipping {row['SNR']}")
                    continue                
                if float(row['RT60']) not in supported_RT60s and reverb:
                    continue
                audio = whisper.load_audio(row['audio'])
                # Calculate the power (squared RMS) of the target audio
                audio_power = np.mean(np.square(audio))
                # if(np.all(audio==0)):
                #     z_count += 1
                #     print(z_count, "zeroed")
                #     continue
                # audio_path = os.path.join(f"/home/data/shua/DB_manipulations/allstar_after_tplc_16k/{str(drop_freq)}/", os.path.basename(row['audio']))
                # audio = whisper.load_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                if clipping:
                    audio = simulate_clipping_numpy(audio, drop_freq)
                if pub_noise:
                    snr_ratio = row['audio'].split('.wav')[0].split("SNR")[1].replace("n", "-").replace('p', '')    
                    baseline_df.at[index, 'SNR'] = snr_ratio.upper()               
                    if snr_ratio.lower() != 'q':
                        snr_ratio = float(snr_ratio)  # Ensure SNR ratio is a float
                        # Calculate the desired noise power based on the SNR
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
                        adjusted_noise = pub_noise_audio * np.sqrt(scaling_factor)
                        # Add the scaled noise to the original audio
                        audio = audio + adjusted_noise

                # make log-Mel spectrogram and move to the same device as the model
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                if packet_loss:
                    mel, dropped_frames = simulate_packet_loss_pytorch(mel, drop_frequency=drop_freq)
                    # create_dropped_audio(row['audio'], dropped_frames, drop_freq, '/home/data/shua/DB_manipulations/allstar_zeroed_16k')
                    actual_drop_freq.append(calculate_zeroed_out_percentage(mel))
                if 'language' in baseline_df.columns:
                    lang = row['language']
                elif 'lang_id' in baseline_df.columns:
                    lang = row['lang_id']
                lang = lang.lower()
                if lang not in LANGUAGES and lang not in TO_LANGUAGE_CODE:
                    print(f"unsupported language {lang}")
                    continue

                options = whisper.DecodingOptions(language=lang, without_timestamps=True, beam_size=5)     
                if finetuned:
                    result = whisper.decode(model, mel, options).text
                    baseline_df.at[index, "asr"] = result.replace(',', '')
                if run_orig:
                    result = whisper.decode(model, mel, options).text
                    baseline_df.at[index, "baseline"] = result.replace(',', '')
                if run_unet:
                    with torch.no_grad():
                        mel = unet_w.unet.forward(mel.unsqueeze(0).unsqueeze(0)).squeeze()
                        # decode the audio
                        result = whisper.decode(model, mel, options).text
                        baseline_df.at[index, "asr"] = result.replace(',', '')          
                # result = whisper.transcribe(model=model, )
                if testing_forward:
                    woptions = whisper.DecodingOptions(language=lang, without_timestamps=True)
                    multilingual_tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=lang, task=woptions.task)
                    multilingual_tokens = torch.tensor([*multilingual_tokenizer.sot_sequence_including_notimestamps]).unsqueeze(0).to(device)
                    audio_features = model.encoder(mel.unsqueeze(0))
                    # out = unet_w.whisper.decoder(multilingual_tokens, audio_features)
                    decoded_list = []  # List to accumulate decoder outputs
                    
                    max_seq_len = 50  # Maximum sequence length
                    while multilingual_tokens.shape[-1] < max_seq_len:
                        # Generate output distribution from the decoder
                        out_distribution = model.decoder(multilingual_tokens, audio_features)
                        # Convert the output distribution to a discrete token
                        next_token = torch.argmax(out_distribution, dim=2)
                        # Check if the last token is the End Of Text token
                        if next_token[:, -1] == multilingual_tokenizer.eot:
                            break
                        # Concatenate the next token to the existing sequence
                        multilingual_tokens = torch.cat([multilingual_tokens, next_token[:, -1:]], dim=1)
                        # Optionally, store each generated token in a list (if needed)
                        decoded_list.append(next_token[:, -1:].detach())
                    txt = multilingual_tokenizer.decode(decoded_list)
                    baseline_df.at[index, "forward"] = txt.replace(',', '')
                    
            baseline_df.at[index, "model"] = asr
        baseline_df.to_csv(f"Data/Model_outputs/{file_name}")
    else:
        file_name = f'whisper_openai_base_22_05_base_PL_ce1_l19_q_epoch=83-step=190091.csv'

    def unify(row, dictionary):
        for key in dictionary.keys():
            row = re.sub(key, dictionary[key], row)
        return row.strip()

    def contractions(target, row, dictionary):
        for key in dictionary.keys():
            if (key.split(")")[1].split("(")[0] in row) and (dictionary[key].strip() in target):
                row = re.sub(key, dictionary[key], row)
        return row.strip().capitalize()

    # Function to remove punctuation from a string
    def remove_punctuation(text):
        return ''.join([char for char in text if char not in string.punctuation])

    def clean_text(text):
        return re.sub(r'[^a-zA-Z0-9 ]', '', text).lower().strip()

    def calculate_hallucination_rate_per_snr(data, target_col, response_col, SNR_type="SNR"):
        # Initialize a dictionary to hold total hallucinations and target lengths for each SNR value
        snr_stats = {}

        for _, row in data.iterrows():
            target = clean_text(row[target_col])
            response = clean_text(row[response_col])
            snr = row[SNR_type]
            if SNR_type == "SNR":
                if str(snr) not in supported_snrs:
                    continue
            if SNR_type == "RT60":
                if float(snr) not in supported_RT60s:
                    continue
            target_words = set(target.split())
            response_words = set(response.split())

            hallucinations = len(response_words.difference(target_words))
            target_length = len(target_words)

            if snr not in snr_stats:
                snr_stats[snr] = {'total_hallucinations': 0, 'total_target_length': 0}

            snr_stats[snr]['total_hallucinations'] += hallucinations
            snr_stats[snr]['total_target_length'] += target_length

        # Calculate and print hallucination rate for each SNR
        for snr, stats in snr_stats.items():
            total_hallucination_rate = (stats['total_hallucinations'] / stats['total_target_length'] * 100) if stats['total_target_length'] > 0 else 0
            print(f"{SNR_type}: {snr}, Hallucination Rate: {total_hallucination_rate:.2f}%")

    # Initialize the WordErrorRate metric from torchmetrics
    wer_metric = torchmetrics.WordErrorRate()
    metrics_wer = evaluate.load("wer")
    data = pd.read_csv(f"Data/Model_outputs/{file_name}", index_col=0)
    # Filter out rows with empty 'target' values
    data = data.dropna(subset=['target'])
    data = data[data['target'].astype(str).str.strip() != '']
    # print(data.columns)
    unnamed_cols = [col for col in data.columns if "Unnamed" in col]
    if len(unnamed_cols) > 0:
        data = data.drop(columns= unnamed_cols)
    
    data.reset_index(inplace=True)
    data["target"] = data["target"].apply(lambda x: x.lower().capitalize())
    if allstar:
        data["response"] = data["response"].apply(lambda x: unify(str(x).lower(), replace_dict))
        data["response"] = data.apply(lambda x: contractions(x["target"].lower(), x["response"].lower(), contraction_dict), axis=1)
        data["human_wer"] = data.apply(lambda x: np.round(wer_metric(x["response"].lower(), x["target"].lower()).item(), 2), axis=1)
    if run_unet or finetuned:
        data["asr"] = data["asr"].apply(lambda x: unify(str(x).lower(), replace_dict))
        data["asr"] = data.apply(lambda x: contractions(x["target"].lower(), x["asr"].lower(), contraction_dict), axis=1)
        data["asr_wer"] = data.apply(lambda x: np.round(wer_metric(x["asr"].lower(), x["target"].lower()).item(), 2), axis=1)
    if run_orig:
        data["baseline"] = data["baseline"].apply(lambda x: unify(str(x).lower(), replace_dict))
        data["baseline"] = data.apply(lambda x: contractions(x["target"].lower(), x["baseline"].lower(), contraction_dict), axis=1)
        data["baseline_wer"] = data.apply(lambda x: np.round(wer_metric(x["baseline"].lower(), x["target"].lower()).item(), 2), axis=1)

    now = datetime.now().strftime("%d.%m")
    data.to_csv(f"Data/Model_outputs/{file_name}", index=False)

    def compute_wer_for_snr(dataframe, system, SNR_type="SNR"):
        # Dictionary to store WER values for each SNR
        snr_wer = {}
        snr_wer2 = {}
        snr_jiwer = {}
        total_targets = []
        total_responses = []
        # Iterate through each unique SNR value
        for snr in dataframe[SNR_type].unique():
            if SNR_type == "SNR":
                if str(snr) not in supported_snrs:
                    continue
            if SNR_type == "RT60":
                if float(snr) not in supported_RT60s:
                    continue
            
            snr_df = dataframe[dataframe[SNR_type] == snr]

            all_targets = []
            all_responses = []

            # Accumulate targets and responses for each row in the filtered dataframe
            for _, row in snr_df.iterrows():
                target = remove_punctuation(row['target']).lower()
                response = remove_punctuation(row[system]).lower()

                all_targets.append(target)
                all_responses.append(response)

                total_targets.append(target)
                total_responses.append(response)

            # Compute WER for the accumulated targets and responses for the current SNR
            snr_wer[snr] = wer_metric(all_responses, all_targets).item()
            snr_wer2[snr] = metrics_wer.compute(predictions=all_responses, references=all_targets)
            snr_jiwer[snr] = jiwer.compute_measures(all_targets, all_responses)
        total_wer = wer_metric(total_responses, total_targets)
        total_jiwer = jiwer.compute_measures(total_targets, total_responses)
        print('Total WER%', total_wer)
        print('Total ji WER%', ", ".join([f"{key}: {total_jiwer[key]}" for key in ["wer", "substitutions", "deletions", "insertions"]]))
        return snr_wer, snr_wer2, snr_jiwer

    print(experiment_name)
    print(drop_freq)
    if len(actual_drop_freq) != 0:
        print(sum(actual_drop_freq) / len(actual_drop_freq) )
    if run_orig:
        noisy_wer, wer2, ji_wer = compute_wer_for_snr(data, 'baseline', SNR_type)
        print("baseline_asr WER for noisy files:", noisy_wer)
        for snr in ji_wer:
            print(f"baseline_asr ji WER for noisy files {snr}:", ", ".join([f"{key}: {ji_wer[snr][key]}" for key in ["wer", "substitutions", "deletions", "insertions"]]))
       
        if calculate_hallucination_rate:
            calculate_hallucination_rate_per_snr(data, 'target', 'baseline', SNR_type)

    if run_unet or finetuned:
        noisy_wer, wer2, ji_wer = compute_wer_for_snr(data, 'asr', SNR_type)
        print("unet asr WER for noisy files:", noisy_wer)
        for snr in ji_wer:
            print(f"unet asr ji WER for noisy files {snr}:", ", ".join([f"{key}: {ji_wer[snr][key]}" for key in ["wer", "substitutions", "deletions", "insertions"]]))
        if calculate_hallucination_rate:
            calculate_hallucination_rate_per_snr(data, 'target', 'asr', SNR_type)

    if multilingual:
        for lang in data.lang_id.unique():
            if run_orig:
                noisy_wer, wer2, ji_wer = compute_wer_for_snr(data[data['lang_id'] == lang], 'baseline', SNR_type)
                print(f"baseline_asr WER for {lang} files:", noisy_wer)
            if run_unet or finetuned:
                noisy_wer, wer2, ji_wer = compute_wer_for_snr(data[data['lang_id'] == lang], 'asr', SNR_type)
                print(f"Unet asr WER for {lang} files:", noisy_wer)

    if testing_forward:
        noisy_wer, wer2 = compute_wer_for_snr(data, "forward")
        print("Forward WER for noisy files forward:", noisy_wer)

if generate_plots:
    data = pd.read_csv(os.path.join(f"Data/Model_outputs/{file_name}"), index_col=0)
    # print(data)
    data["asr"] = data["asr"].fillna("")
    data["wer"] = data.apply(lambda x: np.round(wer_metric(x["asr"].lower(), x["target"].lower()).item(), 2), axis=1)
    data['wer'] = data['wer'].clip(upper=2)

    # data["wer_asr"] = data["wer_asr"].apply(lambda x: 1.0 if x > 1.0 else x)

    sns.relplot(
        data=data, kind="line",
        x="SNR", y="wer", hue="speaker", palette="deep")
    plt.title(f"Relation between WER of {experiment_name} prediction and SNR by speaker")
    sns.set(rc={'figure.figsize':(20,15)})
    plt.savefig(os.path.join(f'Plots/{experiment_name}.png'),bbox_inches='tight', dpi=300)

