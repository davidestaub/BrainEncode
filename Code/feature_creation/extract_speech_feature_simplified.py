#!/usr/bin/env python3

"""
Feature extraction for ASR models supported by Hugging Face.
"""

import argparse
import collections
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional
import configparser

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor, PreTrainedModel, WhisperModel
import time

# Resample to this sample rate. 16kHz is used by most models.
#TODO: check for whisper large
TARGET_SAMPLE_RATE = 16000
SCRIPT_DIR = Path(__file__).resolve().parent

def extract_speech_features(model: PreTrainedModel, model_config: dict, wav: torch.Tensor,
                            chunksz_sec: float, contextsz_sec: float,
                            num_sel_frames = 1, frame_skip = 5, sel_layers: Optional[List[int]]=None,
                            batchsz: int = 1,
                            return_numpy: bool = True, move_to_cpu: bool = True,
                            disable_tqdm: bool = False, feature_extractor=None,
                            sampling_rate: int = TARGET_SAMPLE_RATE, require_full_context: bool = False,
                            stereo: bool = False):
    assert (num_sel_frames == 1), f"'num_sel_frames must be 1 to ensure causal feature extraction, but got {num_sel_frames}. "\
        "This option will be deprecated in the future."
    if stereo:
        raise NotImplementedError("stereo not implemented")
    else:
        assert wav.ndim == 1, f"input wav must be 1-D but got {wav.ndim}"
    if return_numpy: assert move_to_cpu, "'move_to_cpu' must be true if returning numpy arrays"

    # Whisper needs special handling
    is_whisper_model = isinstance(model, WhisperModel)

    #TODO: del gpu memory in loop (if can be saved on cpu)
    #TODO: run with torch no grad

    # Compute chunks & context sizes in terms of samples & context
    chunksz_samples = int(chunksz_sec * sampling_rate)
    contextsz_samples = int(contextsz_sec * sampling_rate)

    # snippet_ends has the last (exclusive) sample for each snippet
    snippet_ends = []
    if not require_full_context:
        # Add all snippets that are _less_ than the total input size
        # (context+chunk)
        snippet_ends.append(torch.arange(chunksz_samples, contextsz_samples+chunksz_samples, chunksz_samples))

    # Add all snippets that are exactly the length of the requested input
    # (Tensor.unfold is basically a sliding window).
    if wav.shape[0] >= chunksz_samples+contextsz_samples:
        snippet_ends.append(
            torch.arange(wav.shape[0]).unfold(0, chunksz_samples+contextsz_samples, chunksz_samples)[:,-1]+1
        )

    snippet_ends = torch.cat(snippet_ends, dim=0) # shape: (num_snippets,)

    if snippet_ends.shape[0] == 0:
        raise ValueError(f"No snippets possible! Stimulus is probably too short ({wav.shape[0]} samples). Consider reducing context size or setting require_full_context=True")

    # 2-D array where [i,0] and [i,1] are the start and end, respectively,
    # of snippet i in samples. Shape: (num_snippets, 2)
    snippet_times = torch.stack([torch.maximum(torch.zeros_like(snippet_ends),
                                               snippet_ends-(contextsz_samples+chunksz_samples)),
                                 snippet_ends], dim=1)

    # Remove snippets that are not long enough.
    if 'min_input_length' in model_config:
        min_length_samples = model_config['min_input_length']
    elif 'win_ms' in model.config:
        min_length_samples = model.config['win_ms'] / 1000. * TARGET_SAMPLE_RATE

    snippet_times = snippet_times[(snippet_times[:,1] - snippet_times[:,0]) >= min_length_samples]
    snippet_times_sec = snippet_times / sampling_rate # snippet_times, but in sec.

    module_features = collections.defaultdict(list)
    out_features = [] # the final output of the model
    times = [] # times are shared across all layers

    frame_len_sec = model_config['stride'] / TARGET_SAMPLE_RATE # length of an output frame (sec.)

    snippet_length_samples = snippet_times[:,1] - snippet_times[:,0] # shape: (num_snippets,)
    if require_full_context:
        assert all(snippet_length_samples == snippet_length_samples[0]), "uneven snippet lengths!"
        snippet_length_samples = snippet_length_samples[0]
        assert snippet_length_samples.ndim == 0

    # Set up the iterator over batches of snippets
    if require_full_context:
        snippet_batches = snippet_times.T.split(batchsz, dim=1)
    else:
        snippet_batches = snippet_times.tensor_split(torch.where(snippet_length_samples.diff() != 0)[0]+1, dim=0)
        snippet_iter = []
        for batch in snippet_batches:
            if batch.shape[0] > batchsz:
                snippet_iter += batch.T.split(batchsz,dim=1)
            else:
                snippet_iter += [batch.T]
        snippet_batches = snippet_iter

    snippet_iter = snippet_batches
    if not disable_tqdm:
        snippet_iter = tqdm(snippet_iter, desc='snippet batches', leave=False)
    snippet_iter = enumerate(snippet_iter)
    #print("memory status pre loop:", torch.cuda.memory_summary())
    # Iterate with a sliding window. stride = chunk_sz
    for batch_idx, (snippet_starts, snippet_ends) in snippet_iter:
        #print("batch_idx",batch_idx,"memory status:",torch.cuda.memory_summary())
        if ((snippet_ends - snippet_starts) < (contextsz_samples + chunksz_samples)).any() and require_full_context:
            raise ValueError("This shouldn't happen with require_full_context")

        if (snippet_ends - snippet_starts < min_length_samples).any():
            print('If this is true for any, then you might be losing more snippets than just the offending (too short) snippet')
            assert False

        # Construct the input waveforms for the batch
        batched_wav_in_list = []
        for batch_snippet_idx, (snippet_start, snippet_end) in enumerate(zip(snippet_starts, snippet_ends)):
            batched_wav_in_list.append(wav[snippet_start:snippet_end])
        batched_wav_in = torch.stack(batched_wav_in_list, dim=0)

        if (snippet_starts.shape[0] != batched_wav_in.shape[0]) and (snippet_starts.shape[0] != batchsz):
            batched_wav_in = batched_wav_in[:snippet_starts.shape[0]]

        output_inds = np.array([-1 - frame_skip*i for i in reversed(range(num_sel_frames))])

        if feature_extractor is not None:
            if stereo: raise NotImplementedError("Support handling multi-channel audio with feature extractor")

            feature_extractor_kwargs = {}
            if is_whisper_model:
                features_key = 'input_features'
                feature_extractor_kwargs['return_attention_mask'] = True
            else:
                features_key = 'input_values'

            preprocessed_snippets = feature_extractor(list(batched_wav_in.cpu().numpy()),
                                                      return_tensors='pt',
                                                      sampling_rate=sampling_rate,
                                                      **feature_extractor_kwargs)
            if is_whisper_model:

                #FAILS HERE
                chunk_features = model.encoder(preprocessed_snippets[features_key].to(model.device))

                #print("CHUNK FEATURES = ",chunk_features )


                contributing_outs = preprocessed_snippets.attention_mask
                contributing_outs = contributing_outs[0].unsqueeze(0)

                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv1.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv1.stride,
                                                               padding=model.encoder.conv1.padding,
                                                               dilation=model.encoder.conv1.dilation,
                                                               groups=model.encoder.conv1.groups)
                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv2.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv2.stride,
                                                               padding=model.encoder.conv2.padding,
                                                               dilation=model.encoder.conv2.dilation,
                                                               groups=model.encoder.conv2.groups)

                final_output = contributing_outs[0].nonzero().squeeze(-1).max()
            else:
                assert sampling_rate == TARGET_SAMPLE_RATE, f"sampling rate mismatch! {sampling_rate} != {TARGET_SAMPLE_RATE}"

                chunk_features = model(preprocessed_snippets[features_key].to(model.device))
        else:
            chunk_features = model(batched_wav_in)

        if(chunk_features['last_hidden_state'].shape[1] < (num_sel_frames-1) * frame_skip - 1):
            print("Skipping:", batch_idx, "only had", chunk_features['last_hidden_state'].shape[1],
                    "outputs, whereas", (num_sel_frames-1) * frame_skip - 1, "were needed.")
            continue

        assert len(output_inds) == 1, "Only one output per evaluation is supported for Hugging Face"

        if is_whisper_model:
            output_inds = [final_output]

        for out_idx, output_offset in enumerate(output_inds):
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))

            output_representation = chunk_features['last_hidden_state'][:, output_offset, :]
            if move_to_cpu: output_representation = output_representation.cpu()
            if return_numpy: output_representation = output_representation.numpy()
            out_features.append(output_representation)

            for layer_idx, layer_activations in enumerate(chunk_features['hidden_states']):
                #print("LAYER IDX = ",layer_idx)
                #print("LAYER ACTIVATIONS =",layer_activations)
                #print(sel_layers)
                #print(layer_idx not in sel_layers)

                #TODO: This is kind of ugly as it neccessitates that the user knows
                # how manny hidden layers there are in the model
                if sel_layers:
                    if layer_idx not in sel_layers: continue

                layer_representation = layer_activations[:, output_offset, :]
                if move_to_cpu: layer_representation = layer_representation.cpu()
                if return_numpy: layer_representation = layer_representation.numpy()

                if is_whisper_model:
                    module_name = f"encoder.{layer_idx}"
                else:
                    module_name = f"layer.{layer_idx}"

                module_features[module_name].append(layer_representation)

        #No further improvements on this scale (A100 1 small story)
        #del chunk_features
        #torch.cuda.empty_cache()

    #print("MODULE FEATURES = ",module_features)
    out_features = np.concatenate(out_features, axis=0) if return_numpy else torch.cat(out_features, dim=0)
    module_features = {name: (np.concatenate(features, axis=0) if return_numpy else torch.cat(features, dim=0))\
                       for name, features in module_features.items()}

    assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
        "Missing timesteps in the module activations!! (possible PyTorch bug)"
    times = torch.cat(times, dim=0) / TARGET_SAMPLE_RATE # convert samples --> seconds. shape: (timesteps,)
    if return_numpy: times = times.numpy()

    del chunk_features # possible memory leak. remove if unneeded
    return {'final_outputs': out_features, 'times': times,
            'module_features': module_features}


if __name__ == "__main__":

    # Parse the input arguments
    parser = argparse.ArgumentParser(description="Extract speech features for a single audio file.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input stimulus file")
    args = parser.parse_args()

    t0 = time.time()

    print(torch.device)
    # Load the configuration file
    config = configparser.ConfigParser()
    config.read(SCRIPT_DIR / 'speech_feature_arguments.ini')

    # Access the values
    stimulus_dir = Path(config['DEFAULT']['stimulus_dir'])
    output_dir = Path(config['DEFAULT']['output_dir'])
    model_name = config['DEFAULT']['model']
    use_featext = config.getboolean('DEFAULT', 'use_featext')
    batchsz = config.getint('DEFAULT', 'batchsz')
    chunksz = config.getfloat('DEFAULT', 'chunksz')
    contextsz = config.getfloat('DEFAULT', 'contextsz')
    layers = [int(x) for x in config['DEFAULT']['layers'].split(',')]
    full_context = config.getboolean('DEFAULT', 'full_context')
    resample = config.getboolean('DEFAULT', 'resample')
    stride = config['DEFAULT']['stride']
    pad_silence = config.getboolean('DEFAULT', 'pad_silence')
    recursive = config.getboolean('DEFAULT', 'recursive')
    overwrite = config.getboolean('DEFAULT', 'overwrite')

    if stride == "None":
        stride = None
    else:
        stride = float(stride)

    # Print loaded config for debugging
    print(f"Loaded config: {config['DEFAULT']}")

    # Load the model configuration
    with open(SCRIPT_DIR / 'speech_model_configs.json', 'r') as f:
        model_config = json.load(f)[model_name]
    model_hf_path = model_config['huggingface_hub']

    print(f'Loading model {model_name} from the Hugging Face Hub...')
    model = AutoModel.from_pretrained(model_hf_path, output_hidden_states=True).cuda()

    torch.set_grad_enabled(False)  # VERY important! (for memory)
    model.eval()

    print(model.device)

    feature_extractor = None

    if use_featext:
        print("using feature extractor")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)

    # Use the provided input stimulus file
    stimulus_local_path = Path(args.input)

    wav, sample_rate = torchaudio.load(stimulus_local_path)
    if not resample:
        assert wav.shape[0] == 1, f"stimulus '{stimulus_local_path}' is not mono-channel"
        assert sample_rate == TARGET_SAMPLE_RATE
    else:
        if wav.shape[0] != 1: wav = wav.mean(0, keepdims=True)
        if sample_rate != TARGET_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

    wav.squeeze_(0)

    assert sample_rate == TARGET_SAMPLE_RATE, f"Expected sample rate {TARGET_SAMPLE_RATE} but got {sample_rate}"

    stimulus_name = stimulus_local_path.stem

    output_dir = output_dir / f"features_cnk{chunksz:0.1f}_ctx{contextsz:0.1f}" / model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    features_save_path = output_dir / f"{stimulus_name}-v3.npz"
    times_save_path = output_dir / f"{stimulus_name}_times-v3.npz"

    if not overwrite and features_save_path.exists() and times_save_path.exists():
        print(f"Skipping {stimulus_name}, features found at {features_save_path}")
    else:

        extract_features_kwargs = {
            'model': model, 'model_config': model_config,
            'wav': wav.to(model.device),
            'chunksz_sec': chunksz / 1000., 'contextsz_sec': contextsz / 1000.,
            'sel_layers': layers, 'feature_extractor': feature_extractor,
            'require_full_context': full_context or pad_silence,
            'batchsz': batchsz, 'return_numpy': False
        }

        if stride:
            extract_features_kwargs['contextsz_sec'] = chunksz / 1000. + contextsz / 1000. - stride
            extract_features_kwargs['chunksz_sec'] = stride

        if pad_silence:
            wav = torch.cat([torch.zeros(int(extract_features_kwargs['contextsz_sec'] * TARGET_SAMPLE_RATE)), wav],
                            axis=0)
            extract_features_kwargs['wav'] = wav.to(model.device)

        extracted_features = extract_speech_features(**extract_features_kwargs)

        out_features, times, module_features = [extracted_features[k] for k in \
                                                ['final_outputs', 'times', 'module_features']]
        del extracted_features

        if pad_silence:
            times = torch.clip(times - extract_features_kwargs['contextsz_sec'], 0, torch.inf)
            assert torch.all(times >= 0), "padding is smaller than the correction (subtraction)!"
            assert torch.all(times[:, 1] > 0), f"insufficient padding for require_full_context !"

        np.savez_compressed(features_save_path, features=out_features.numpy())
        np.savez_compressed(times_save_path, times=times.numpy())

        module_save_paths = {module: output_dir / module / f"{stimulus_name}.npz" for module in module_features.keys()}

        for module_name, features in module_features.items():
            features_save_path = module_save_paths[module_name]
            features_save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(features_save_path, features=features.numpy())

    t1 = time.time()
    print(f"Extracting features using {model_name} and {layers} as layers took {t1-t0} seconds / {(t1-t0)/60} Minutes")
