import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from tqdm import tqdm
from datasets import audio

def is_mulaw_quantize(input_type):
    return input_type == 'mulaw-quantize'

def is_mulaw(input_type):
    return input_type == 'mulaw'

def mulaw(x, quantize_channels):
    mu = quantize_channels - 1
    safe_x = np.minimum(np.abs(x), 1.0)
    f = np.sign(x) * np.log1p(mu * safe_x) / np.log1p(mu)
    return f

def mulaw_quantize(x, quantize_channels):
    mu = quantize_channels - 1
    y = mulaw(x, quantize_channels)
    return ((y + 1) / 2 * mu + 0.5).astype(np.int)

def build_from_path(hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omitted
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    for input_dir in input_dirs:
        with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                print("Debug: parts =", parts)
                if len(parts) < 3:
                    print("Skipping line:", line)
                    continue
                basename = parts[0]
                wav_path = os.path.join(input_dir, 'wavs', '{}'.format(basename))
                print(f"Debug: Checking if file exists: {wav_path}")
                if not os.path.exists(wav_path):
                    print(f"File {wav_path} does not exist, skipping!")
                    continue
                text = parts[2]
                futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
                index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]

def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
    """
    Preprocesses a single utterance wav/text pair

    This writes the mel scale spectrogram to disk and returns a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectrograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectrogram filename
        - wav_path: path to the audio file containing the speech input
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
        return None

    # Trim lead/trail silences
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)

    # Pre-emphasize
    preem_wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

    # Rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
        preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

        # Assert all audio is in [-1, 1]
        if (wav > 1.).any() or (wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))
        if (preem_wav > 1.).any() or (preem_wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = mulaw_quantize(wav, hparams.quantize_channels)

        # Trim silences
        start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start: end]
        preem_wav = preem_wav[start: end]
        out = out[start: end]

        constant_values = mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16

    elif is_mulaw(hparams.input_type):  # [-1, 1]
        out = mulaw(wav, hparams.quantize_channels)
        constant_values = mulaw(0., hparams.quantize_channels)
        out_dtype = np.float32

    else:
        # [-1, 1]
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(preem_wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio.linearspectrogram(preem_wav, hparams).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]

    # Sanity check
    assert linear_frames == mel_frames

    if hparams.use_lws:
        # Ensure time resolution adjustment between audio and mel-spectrogram
        fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
        l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

        # Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        # Ensure time resolution adjustment between audio and mel-spectrogram
        l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams), hparams.wavenet_pad_sides)

        # Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    # Time resolution adjustment
    # Ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)

    # Write the spectrogram and audio to disk
    audio_filename = 'audio-{}.npy'.format(index)
    mel_filename = 'mel-{}.npy'.format(index)
    linear_filename = 'linear-{}.npy'.format(index)
    np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)
