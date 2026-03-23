#!/usr/bin/env python3

"""
Feature extraction for text using Hugging Face language models, aligned with the user's implementation.
Includes optional “eventboundary_log_prob” extraction based on an event‐boundary prompt, computed over audio‐based chunks.
"""
from tqdm import tqdm
import uroman as ur
import torchaudio
import torchaudio.transforms as T
import sys
print(sys.executable, flush=True)
from pathlib import Path
import ridge_utils.textgrid as textgrid
import numpy as np
from ridge_utils.dsutils import make_word_ds
from ridge_utils.DataSequence import DataSequence
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from ridge_utils.tokenization_helpers import (
    generate_efficient_feat_dicts_llama_NEW,
    convert_to_feature_mats_llama_NEW
)
import torch
import os
import pickle as pkl
import argparse, time, configparser, re, string
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

# ─────────────────────────────────────────────────────────────
# 2)  BOUNDARY DETECTION UTILS
# ─────────────────────────────────────────────────────────────
# Prompts
CTX_LIMIT_DEFAULT   = 2048
EVENT_MARKER        = "¶"

SYSTEM_PROMPT = (
    "Ein Event ist eine fortlaufende Situation. " 
    "Du bekommst gleich einen Text. "
    "Deine Aufgabe:\n"
    " Kopiere den Text wort‑für‑wort.\n"
    " Unterteile den Text in Events."
    " Füge ausschließlich den Event‑Marker ¶ ein – "
    " genau dann (und nur dann), wenn ein Event endet und ein neues beginnt.\n"
    " Halte die Zahl der Marker so klein wie möglich (<150 Marker pro 10000 Wörter).\n"
    "\n"
    "Wichtig\n"
    "• Gib nur den modifizierten Text zurück – keine Überschriften, Einleitungen, "
    "Erklärungen, Entschuldigungen oder sonstigen Kommentare.\n"
    "• Ändere keinerlei Rechtschreibung, Zeichensetzung oder Wortreihenfolge außer "
    "dem Einfügen von „¶“.\n"
    "• Setze den Marker ohne zusätzliche Leer‑ oder Sonderzeichen (also nicht „**¶**“).\n"
)

#NOT USED ANYMORE
PROMPT_CORE = (
    "Ein Event ist eine fortlaufende Situation. "
    "Die folgende Geschichte muss kopiert und in möglichst wenige Events unterteilt werden."
    "Kopiere die folgende Geschichte wort für Wort und füge ausschlieslich den Event-Marker"
    f"{EVENT_MARKER} ein, dann und nur dann, wenn ein Event endet und ein neuer beginnt. "
    "\n\n"
    "Dies ist die Geschichte:\n\n"
)

uroman_obj = ur.Uroman()

def build_word_index_mapping(var_words: list[str],
                             ref_words: list[str]) -> dict[int,int]:
    """
    Returns a dict mapping each index in var_words (0‑based) to the
    matching index in ref_words (0‑based), wherever the SequenceMatcher
    sees exact equality runs.
    """
    sm = SequenceMatcher(None, var_words, ref_words, autojunk=False)
    mapping = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    return mapping

def align_indices(ref_words: list[str],
                  var_words: list[str],
                  var_EB: list[int]) -> list[int]:
    """
    Translate a list of event‑boundary indices var_EB, which are 1‑based
    positions in var_words, into the corresponding 1‑based positions in ref_words.
    """
    # build the 0‑based mapping
    mapping = build_word_index_mapping(var_words, ref_words)

    aligned = []
    for w in var_EB:
        var_i = w - 1
        ref_i = mapping.get(var_i)
        if ref_i is not None:
            # back to 1‑based
            aligned.append(ref_i + 1)
    return aligned
_WS_RE = re.compile(r"\s+")
def strip_eb_markers(txt: str):
    parts = txt.split(EVENT_MARKER)
    eb, words = [], []
    for i, seg in enumerate(parts):
        seg_norm = _WS_RE.sub(" ", seg.strip())
        seg_words = seg_norm.split(" ") if seg_norm else []
        words.extend(seg_words)
        if i < len(parts) - 1:
            eb.append(len(words))
    return words, eb

##############################################################################
# Long‑story helper: handles stories that exceed the context window
##############################################################################
def greedy_copy_with_marker_smart(
    story: str,
    tok,
    model,
    *,
    device: str = "cuda",
    max_slack: int = 256,      # safety buffer for extra markers
    overlap: int = 50,         # tokens of look‑back on each new window
    event_marker: str = "¶",
):
    """
    Greedy copy of `story` with EVENT_MARKER inserted at event boundaries.
    Works for arbitrarily long stories by sliding a window over the text.

    Returns
    -------
    full_gen : str          # the story incl. inserted markers
    eb       : List[int]    # word indices (0‑based, whole story) AFTER which
                            # EVENT_MARKER was inserted
    """
    import math, re

    # ------------------------------------------------------------------ helpers
    def _generate_one_chunk(chunk_text: str):
        """Chat‑template wrapper for a single chunk (no length limit here)."""
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content":  chunk_text},
        ]
        ids_in = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        n_story_tok   = len(tok(chunk_text, add_special_tokens=False).input_ids)
        approx_markers = max(1, n_story_tok // 15)
        max_new       = n_story_tok + approx_markers + max_slack

        with torch.no_grad():
            out = model.generate(
                ids_in,
                do_sample=False,
                max_new_tokens=max_new,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][ids_in.size(1):], skip_special_tokens=False)

        # ----- extract marker offsets for this chunk -------------------------
        parts, eb_loc, w = re.split(f"({re.escape(event_marker)})", gen), [], 0
        for p in parts:
            if p == event_marker:
                eb_loc.append(w)          # boundary after word w‑1
            elif p.strip():
                w += len(p.split())
        return gen, eb_loc

    # ----------------------------------------------------------------- prework
    ctx_limit = getattr(model.config, "max_position_embeddings", 8192)

    # template length once (story inserted later, so use dummy)
    tmp = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": ""},
    ]
    tmpl_len = tok.apply_chat_template(
        tmp, add_generation_prompt=True, return_tensors="pt"
    ).size(1)

    room_for_story = ctx_limit - tmpl_len - max_slack
    if room_for_story <= 0:
        raise ValueError("Prompt itself already fills the context window.")

    # full story in tokens
    story_ids = tok(story, add_special_tokens=False).input_ids
    n_story_tokens = len(story_ids)

    # ---------------------------------------------------------------- dispatch
    if n_story_tokens <= room_for_story:             # short story → one pass
        return _generate_one_chunk(story)

    # ---------------------------------------------------------------- long case
    stride      = room_for_story - overlap
    n_chunks    = math.ceil((n_story_tokens - room_for_story) / stride) + 1

    words       = story.split()
    gen_chunks  = []
    eb_global   = []
    word_cursor = 0

    for ci in range(n_chunks):
        s_tok = ci * stride
        e_tok = min(s_tok + room_for_story, n_story_tokens)
        chunk_text = tok.decode(story_ids[s_tok:e_tok], skip_special_tokens=True)

        gen_chunk, eb_chunk = _generate_one_chunk(chunk_text)

        # keep markers outside trailing overlap (except last chunk)
        valid_limit = math.inf if ci == n_chunks - 1 else len(chunk_text.split()) - overlap
        eb_global.extend([word_cursor + w for w in eb_chunk if w < valid_limit])

        # merge generated text
        if ci < n_chunks - 1:
            keep = len(chunk_text.split()) - overlap
            gen_chunks.append(" ".join(gen_chunk.split()[:keep]))
            word_cursor += keep
        else:
            gen_chunks.append(gen_chunk)

    return " ".join(gen_chunks), eb_global

# ── 2) Revised boundary_lp with LOOK-AHEAD ───────────────────────────────

# ─────────────────────────────────────────────────────────────
# 2)  BOUNDARY‑LP WITH TRUE LOOK‑AHEAD   (replace the old function)
# ─────────────────────────────────────────────────────────────
def boundary_lp(ds, tok, model, lookahead: int, device="cuda"):
    """
    lp_word[i]  =  log p(EVENT_MARKER | context that includes *lookahead*
                 future tokens beyond word i)
    """
    enc          = tok(ds.data.tolist(),
                       is_split_into_words=True,
                       add_special_tokens=False,
                       return_tensors="pt")
    seq          = enc.input_ids[0].tolist()
    word_for_tok = enc.word_ids()

    lp_tok  = np.full(len(seq),     np.nan, np.float32)
    lp_word = np.full(len(ds.data), np.nan, np.float32)

    # ---- build the chat prefix once
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": PROMPT_CORE},
    ]
    chat_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt"
    ).tolist()[0]

    ctx_lim = getattr(model.config, "max_position_embeddings",
                      CTX_LIMIT_DEFAULT)
    room = ctx_lim - len(chat_ids) - 2                 # ‑2 safety margin
    if room <= lookahead:
        raise ValueError("look‑ahead larger than available context room")

    # ---- sliding window
    for s in range(0, len(seq), room - lookahead):
        e_main = min(s + room - lookahead, len(seq) - lookahead)
        block  = seq[s : e_main + lookahead]           # +future
        ids    = torch.tensor([chat_ids + block], device=device)

        with torch.no_grad():
            logits = model(ids,use_cache=False).logits[0]
        lps = torch.log_softmax(logits, -1).cpu().numpy()
        off = len(chat_ids)

        #
        # logits row j predicts token j+1
        # → to score boundary before token  j+lookahead,
        #   read logits at (j+lookahead‑1)
        #
        for j in range(0, len(block) - lookahead):  # j = position of w_t
            tgt_tok_idx = s + j  # absolute index of w_t
            #
            # prefix length that produced row = off + j
            # ⇒ row that predicts w_t is   off + (j - lookahead)
            #
            src_row = off + j #- lookahead
            score = lps[src_row, marker_id]
            lp_tok[tgt_tok_idx] = score
            w = word_for_tok[tgt_tok_idx]
            if w is not None:
                lp_word[w] = score

    mask = np.isnan(lp_word)
    if mask.all():
        raise RuntimeError("All scores are NaN — check look‑ahead logic.")
    if mask.any():
        idx_valid = np.where(~mask)[0]
        print("nan values at indices = ",np.where(mask)[0])
        # forward/backward fill the extremes
        first, last = idx_valid[0], idx_valid[-1]
        lp_word[:first] = lp_word[first]
        lp_word[last + 1:] = lp_word[last]
        # linear interpolation for interior gaps
        lp_word[mask] = np.interp(
            np.where(mask)[0], idx_valid, lp_word[idx_valid]
        )
        # lp_tok: simple nearest fill is enough for plotting
        mask_tok = np.isnan(lp_tok)
        if mask_tok.any():
            idx_valid = np.where(~mask_tok)[0]
            first, last = idx_valid[0], idx_valid[-1]
            lp_tok[:first] = lp_tok[first]
            lp_tok[last + 1:] = lp_tok[last]
            lp_tok[mask_tok] = np.interp(
                np.where(mask_tok)[0], idx_valid, lp_tok[idx_valid]
            )

    # ---- down‑sample word‑level to TR‑bins
    ds_lp = DataSequence(lp_word, ds.split_inds,
                         ds.data_times, ds.tr_times)
    lp_tr = np.asarray(
        ds_lp.chunksums("lanczos", window=1).data, dtype=np.float32
    )


    return lp_tr, lp_word, lp_tok


class Simple_TR_File:
    def __init__(self, tr_time_list, avg_tr):
        self.avgtr = avg_tr
        self.tr_list = tr_time_list

    def get_reltriggertimes(self):
        return self.tr_list


def shift_extracted_features(audio_start_time, times):
    shifted_times = times + audio_start_time  # Align the times with the BOLD signal
    return shifted_times


### NEW — punctuation helpers
PUNCT_MARKS = {".", ",", ":", ";"}

def _token_is_punct(tok, tok_id: int) -> bool:
    """
    Robustly checks whether a single token represents one of . , : ;
    Works with GPT-2/BPE (Ġ space) and SentencePiece (▁ space).
    """
    try:
        s = tok.convert_ids_to_tokens(int(tok_id))
    except Exception:
        s = None
    if s is None:
        s = tok.decode([int(tok_id)], skip_special_tokens=True)
    # strip tokenizer's 'space' markers and whitespace
    s = (s or "").replace("Ġ", "").replace("▁", "").strip()
    return s in PUNCT_MARKS

def _tiny_smooth_1d(x: np.ndarray,
                    kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Minimal smoothing to avoid delta functions. Uses a 3-tap triangular kernel.
    """
    if kernel is None:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size < 2:
        return x
    pad = len(kernel) // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xp, kernel, mode="valid")
    return y.astype(np.float32)



def normalize_uroman(text: str) -> str:
    import re, unicodedata
    text = text.encode('utf-8').decode('utf-8')
    text = text.lower().replace("’", "'")
    text = unicodedata.normalize('NFC', text)
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def romanize_and_clean(word: str) -> str:
    import unicodedata
    # try whichever romanize method your version has
    if hasattr(uroman_obj, "romanize_string"):
        rom = uroman_obj.romanize_string(word)
    elif hasattr(uroman_obj, "romanize"):
        rom = uroman_obj.romanize(word)
    else:
        # fallback: strip diacritics
        rom = unicodedata.normalize('NFKD', word)
        rom = rom.encode('ascii', 'ignore').decode('ascii')
    return normalize_uroman(rom)


def generate_textgrid_for_story(
    FA_model, FA_aligner, FA_tokenizer, story_name, text_raw, audio_path, output_dir, bundle, device
):
    """Generates TextGrid file for a given story."""
    import torch

    # Initialize uroman and normalize text
    text_roman = uroman_obj.romanize_string(text_raw)
    text_normalized = normalize_uroman(text_roman)
    transcript = text_normalized.split()

    # Load waveform
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if necessary
    if sample_rate != bundle.sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)
        sample_rate = bundle.sample_rate

    assert sample_rate == bundle.sample_rate

    # Try processing the whole waveform at once
    try:
        with torch.inference_mode():
            emission, _ = FA_model(waveform.to(device))
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            # If out of memory, process in chunks
            print("CUDA out of memory, processing in chunks...", flush=True)
            # Define chunk parameters
            chunk_duration = 1200  # 20 minutes in seconds
            overlap_duration = 10  # 10 seconds

            chunk_size = int(chunk_duration * sample_rate)
            overlap_size = int(overlap_duration * sample_rate)

            waveform_length = waveform.size(-1)
            chunks = []
            start = 0
            while start < waveform_length:
                end = min(start + chunk_size, waveform_length)
                chunk = waveform[:, start:end]
                chunks.append((chunk, start))
                print(f"start = {start}, end = {end}, waveform length = {waveform_length}", flush=True)

                if end == waveform_length:
                    break
                else:
                    start = end - overlap_size  # Move start with overlap
            emission_chunks = []
            with torch.inference_mode():
                for idx, (chunk, offset) in enumerate(chunks):
                    emission_chunk, _ = FA_model(chunk.to(device))
                    print("working on chunk", idx, flush=True)
                    free, total = torch.cuda.mem_get_info()
                    print(f"Free memory: {free / 1e6:.2f} MB", flush=True)
                    print(f"Total memory: {total / 1e6:.2f} MB", flush=True)

                    # Handle overlapping emissions
                    if idx > 0:
                        # Calculate frames per second for this chunk
                        chunk_duration_in_sec = chunk.size(1) / sample_rate
                        frames_per_second = emission_chunk.size(1) / chunk_duration_in_sec
                        # Calculate the number of overlapping frames to discard
                        overlap_frames = int(frames_per_second * overlap_duration)
                        emission_chunk = emission_chunk[:, overlap_frames:]

                    emission_chunk = emission_chunk.cpu()
                    emission_chunks.append(emission_chunk)
                    del emission_chunk  # Free up GPU memory
                    del chunk
                    torch.cuda.empty_cache()
                # Concatenate emission chunks
                emission = torch.cat(emission_chunks, dim=1)
                del emission_chunks  # Free up memory
            torch.cuda.empty_cache()
        else:
            # Re-raise if it's a different error
            raise e
    else:
        # If successful, proceed without moving emission to CPU
        pass

    # Proceed with alignment
    token_spans = FA_aligner(emission[0], FA_tokenizer(transcript))
    num_frames = emission.size(1)
    waveform_length = waveform.size(1)
    total_duration = waveform_length / sample_rate  # Total duration in seconds
    frames_per_second = num_frames / total_duration

    # Optionally, delete emission if no longer needed
    del emission
    torch.cuda.empty_cache()

    # Create TextGrid
    output_textgrid = output_dir / f"{story_name}.TextGrid"
    num_words = len(transcript)

    textgrid_lines = []
    textgrid_lines.append('File type = "ooTextFile"')
    textgrid_lines.append('Object class = "TextGrid"')
    textgrid_lines.append("")
    textgrid_lines.append(f"xmin = 0 ")
    textgrid_lines.append(f"xmax = {total_duration}")
    textgrid_lines.append("tiers? <exists>")
    textgrid_lines.append("size = 1")
    textgrid_lines.append("item []:")
    textgrid_lines.append("    item [1]:")
    textgrid_lines.append('        class = "IntervalTier"')
    textgrid_lines.append('        name = "words"')
    textgrid_lines.append(f"        xmin = 0")
    textgrid_lines.append(f"        xmax = {total_duration}")
    textgrid_lines.append(f"        intervals: size = {num_words}")

    # Iterate over each word and create an interval
    for idx, (spans, word) in enumerate(zip(token_spans, transcript)):
        start_frame = spans[0].start
        end_frame = spans[-1].end
        start_sec = start_frame / frames_per_second
        end_sec = end_frame / frames_per_second

        textgrid_lines.append(f"        intervals [{idx + 1}]:")
        textgrid_lines.append(f"            xmin = {start_sec}")
        textgrid_lines.append(f"            xmax = {end_sec}")
        textgrid_lines.append(f"            text = \"{word}\"")

    # Write to file
    with open(output_textgrid, "w", encoding="utf-8") as f:
        f.write("\n".join(textgrid_lines))

    del waveform
    torch.cuda.empty_cache()


def extract_text_features(
        model, tokenizer, wordds, layers, device='cuda', disable_tqdm=False,
        audio_start_times=None, feature_types=None, lookback1=256, lookback2=512
):
    """
    Extract features from text data using a language model, including:
      - hidden_states, hidden_state_diff, kl_div_next, log_prob_actual, max_log_prob,
        entropy_logits, surprisal, perplexity, anticipation_gap.
    Returns a dict mapping:
      (feature_type, layer) -> { story -> DataSequence } for hidden_states/hidden_state_diff
      feature_type -> { story -> DataSequence } for others.
    """
    import torch
    import numpy as np

    if feature_types is None:
        feature_types = ['hidden_states']
    elif isinstance(feature_types, str):
        feature_types = [feature_types]

    # 1) Prepare outer dict for final DataSequence objects
    all_featureseqs = {}
    for ft in feature_types:
        if ft == 'hidden_states':
            for layer in layers:
                all_featureseqs[(ft, layer)] = {}
        elif ft == 'hidden_state_diff':
            for layer in layers:
                all_featureseqs[(ft, layer)] = {}
        elif ft == 'kl_div_next':
            all_featureseqs['kl_div_next'] = {}
        else:
            all_featureseqs[ft] = {}

    # 2) Iterate over each story
    for story in wordds.keys():
        print(f"Processing story: {story}")

        # Build the forced-alignment "chunk" dicts
        text_dict, text_dict2, _ = generate_efficient_feat_dicts_llama_NEW(
            {story: wordds[story]}, tokenizer, lookback1, lookback2
        )

        # 3) Init per-chunk feature dicts
        feature_dicts = {}
        for ft in feature_types:
            if ft == 'hidden_states':
                for layer in layers:
                    feature_dicts[(ft, layer)] = {}
            elif ft == 'hidden_state_diff':
                for layer in layers:
                    feature_dicts[(ft, layer)] = {}
            else:
                feature_dicts[ft] = {}

        model.eval()

        # 4) Loop over all forced-alignment chunks
        for phrase in tqdm(text_dict2, desc=f"Extracting features for {story}", disable=disable_tqdm):
            if not text_dict2[phrase]:
                continue

            # Build inputs for this chunk
            input_ids_list = text_dict[phrase]
            inputs = {
                'input_ids': torch.tensor([input_ids_list], dtype=torch.int64).to(device),
                'attention_mask': torch.ones((1, len(input_ids_list)), dtype=torch.int64).to(device),
            }
            outputs = model(**inputs, output_hidden_states=True)

            # The entire chunk key and sequence length
            input_ids = inputs['input_ids'][0].cpu().numpy()
            this_key = tuple(input_ids)
            seq_len = len(input_ids)

            # 4a) Per-prefix features (except final token)
            for ei in range(seq_len - 1):
                key_prefix = this_key[:ei + 1]
                token_id = input_ids[ei + 1]

                logits = outputs.logits[0][ei].cpu().detach()
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                token_log_prob = log_probs[token_id].item()
                max_log_prob   = torch.max(log_probs).item()
                entropy        = -torch.sum(torch.exp(log_probs) * log_probs).item()
                surprisal_val  = -token_log_prob
                perplexity_val = float(np.exp(surprisal_val))
                anticipation_gap_val = max_log_prob - token_log_prob

                for ft in feature_types:
                    if ft == 'hidden_states':
                        for layer in layers:
                            h = outputs.hidden_states[layer][0][ei].cpu().detach().numpy()
                            feature_dicts[(ft, layer)][key_prefix] = h
                    elif ft == 'hidden_state_diff':
                        for layer in layers:
                            if ei == 0:
                                h0 = outputs.hidden_states[layer][0][0].cpu().detach().numpy()
                                diff_vec = np.zeros_like(h0)
                            else:
                                h_curr = outputs.hidden_states[layer][0][ei].cpu().detach()
                                h_prev = outputs.hidden_states[layer][0][ei - 1].cpu().detach()
                                diff_vec = (h_curr - h_prev).numpy()
                            feature_dicts[(ft, layer)][key_prefix] = diff_vec
                    elif ft == 'log_prob_actual':
                        feature_dicts[ft][key_prefix] = np.array([token_log_prob])
                    elif ft == 'max_log_prob':
                        feature_dicts[ft][key_prefix] = np.array([max_log_prob])
                    elif ft == 'entropy_logits':
                        feature_dicts[ft][key_prefix] = np.array([entropy])
                    elif ft == 'surprisal':
                        feature_dicts[ft][key_prefix] = np.array([surprisal_val])
                    elif ft == 'perplexity':
                        feature_dicts[ft][key_prefix] = np.array([perplexity_val])
                    elif ft == 'anticipation_gap':
                        feature_dicts[ft][key_prefix] = np.array([anticipation_gap_val])


                    elif ft == 'punctuation':
                        # Mark a 1.0 impulse when the NEXT token is punctuation (.,:;)
                        is_p = 1.0 if _token_is_punct(tokenizer, token_id) else 0.0
                        feature_dicts[ft][key_prefix] = np.array([is_p], dtype=np.float32)

                    elif ft == 'kl_div_next':
                        if ei < seq_len - 2:
                            next_logits    = outputs.logits[0][ei + 1].cpu().detach()
                            next_log_probs = torch.nn.functional.log_softmax(next_logits, dim=-1)
                            p_next         = torch.exp(next_log_probs)
                            kl_val         = torch.sum(p_next * (next_log_probs - log_probs)).item()
                        else:
                            kl_val = 0.0
                        feature_dicts[ft][key_prefix] = np.array([kl_val])

            # 4b) Final token in the chunk
            ei = seq_len - 1
            key_prefix = this_key[:ei + 1]
            for ft in feature_types:
                if ft == 'hidden_states':
                    for layer in layers:
                        h_last = outputs.hidden_states[layer][0][ei].cpu().detach().numpy()
                        feature_dicts[(ft, layer)][key_prefix] = h_last
                elif ft == 'hidden_state_diff':
                    for layer in layers:
                        if ei == 0:
                            h0 = outputs.hidden_states[layer][0][0].cpu().detach().numpy()
                            diff_vec = np.zeros_like(h0)
                        else:
                            h_curr = outputs.hidden_states[layer][0][ei].cpu().detach()
                            h_prev = outputs.hidden_states[layer][0][ei - 1].cpu().detach()
                            diff_vec = (h_curr - h_prev).numpy()
                        feature_dicts[(ft, layer)][key_prefix] = diff_vec

                elif ft == 'punctuation':
                    final_is_p = 1.0 if _token_is_punct(tokenizer, input_ids[ei]) else 0.0
                    feature_dicts[ft][key_prefix] = np.array([final_is_p], dtype=np.float32)

            if seq_len >= 2:
                final_token_idx      = seq_len - 1
                logits_second_last   = outputs.logits[0][final_token_idx - 1].cpu().detach()
                final_log_probs      = torch.nn.functional.log_softmax(logits_second_last, dim=-1)
                final_token_id       = input_ids[final_token_idx]
                final_token_log_prob = final_log_probs[final_token_id].item()
                final_max_log_prob   = torch.max(final_log_probs).item()
                final_entropy        = -torch.sum(torch.exp(final_log_probs) * final_log_probs).item()
                final_surprisal_val  = -final_token_log_prob
                final_perplexity_val = float(np.exp(final_surprisal_val))
                final_anticipation_gap_val = final_max_log_prob - final_token_log_prob

                for ft in feature_types:
                    if ft == 'log_prob_actual':
                        feature_dicts[ft][key_prefix] = np.array([final_token_log_prob])
                    elif ft == 'max_log_prob':
                        feature_dicts[ft][key_prefix] = np.array([final_max_log_prob])
                    elif ft == 'entropy_logits':
                        feature_dicts[ft][key_prefix] = np.array([final_entropy])
                    elif ft == 'surprisal':
                        feature_dicts[ft][key_prefix] = np.array([final_surprisal_val])
                    elif ft == 'perplexity':
                        feature_dicts[ft][key_prefix] = np.array([final_perplexity_val])
                    elif ft == 'anticipation_gap':
                        feature_dicts[ft][key_prefix] = np.array([final_anticipation_gap_val])
                    elif ft == 'kl_div_next':
                        feature_dicts[ft][key_prefix] = np.array([0.0])

        # 5) Convert each feature type into DataSequence and store
        audio_start_time = audio_start_times.get(story, 0.0)
        for ft in feature_types:
            if ft == 'hidden_states':
                for layer in layers:
                    feats = convert_to_feature_mats_llama_NEW(
                        {story: wordds[story]},
                        tokenizer, lookback1, lookback2,
                        feature_dicts[(ft, layer)], audio_start_time,
                    )
                    all_featureseqs[(ft, layer)][story] = feats[story]
            elif ft == 'hidden_state_diff':
                for layer in layers:
                    feats = convert_to_feature_mats_llama_NEW(
                        {story: wordds[story]},
                        tokenizer, lookback1, lookback2,
                        feature_dicts[(ft, layer)], audio_start_time,
                    )
                    all_featureseqs[(ft, layer)][story] = feats[story]

            elif ft == 'punctuation':
                # 1) Read raw, punctuated text
                raw_text = (text_dir / f"{story}.txt").read_text(encoding="utf8").replace("\n", " ")

                # 2) Split so that ., : ; are standalone tokens
                toks = re.findall(r"[A-Za-zÀ-ÿ']+|[.,:;]", raw_text)

                # 3) Build model-word list and record punctuation after word count "wc"
                model_words = []
                EB_model = []  # 1-based word indices *before* each punctuation mark
                wc = 0
                for t in toks:
                    if t in PUNCT_MARKS:
                        if wc > 0:
                            EB_model.append(wc)
                    else:
                        model_words.append(t)
                        wc += 1

                # 4) Clean & align model words → FA (spoken) words
                spoken_ds = wordds[story]
                spoken_words = list(spoken_ds.data)  # FA words (lower, no punctuation)
                model_clean = [romanize_and_clean(w) for w in model_words]
                spoken_clean = [normalize_uroman(w) for w in spoken_words]

                fa_EB = align_indices(
                    ref_words=spoken_clean,  # target grid (FA)
                    var_words=model_clean,  # source grid (raw model words)
                    var_EB=EB_model  # 1-based indices in model words
                )

                # 5) Word-level impulse vector on FA grid
                p_word = np.zeros(len(spoken_clean), dtype=np.float32)
                for w in fa_EB:
                    if 1 <= w <= len(p_word):
                        p_word[w - 1] = 1.0

                # 6) Shift to fMRI clock and downsample to TRs
                sh = audio_start_times.get(story, 0.0)
                shifted_word_times = spoken_ds.data_times + sh
                shifted_tr_times = spoken_ds.tr_times + sh

                ds_punct_word = DataSequence(
                    p_word, spoken_ds.split_inds, shifted_word_times, shifted_tr_times
                )

                # Some versions return DataSequence, others return np.ndarray
                res = ds_punct_word.chunksums("lanczos", window=1)

                if isinstance(res, DataSequence):
                    y_tr = np.asarray(res.data, dtype=np.float32)
                    split_inds = res.split_inds
                    data_times = res.data_times
                    tr_times = res.tr_times
                else:
                    y_tr = np.asarray(res, dtype=np.float32)  # TR-length vector
                    tr_times = ds_punct_word.tr_times  # use TR axis
                    data_times = tr_times
                    split_inds = list(range(len(y_tr)))  # harmless placeholder

                # 7) Tiny smoothing at TR level so it’s not a spike train
                y_tr_sm = _tiny_smooth_1d(y_tr)

                all_featureseqs['punctuation'][story] = DataSequence(
                    y_tr_sm, split_inds, data_times, tr_times
                )

                # (optional sanity)
                nz = int((y_tr_sm > 0).sum())
                print(f"[punctuation] {story}: nonzero TRs = {nz}/{len(y_tr_sm)}")




            else:
                feats = convert_to_feature_mats_llama_NEW(
                    {story: wordds[story]},
                    tokenizer, lookback1, lookback2,
                    feature_dicts[ft], audio_start_time,
                )
                all_featureseqs[ft][story] = feats[story]

    return all_featureseqs


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse the input arguments
    if str(device) == 'cpu':
        print("using cpu")
        parser = argparse.ArgumentParser(description="Extract text features for stories.")
        parser.add_argument('--stories', nargs='+', required=True,
                            help="List of stories to process")
        parser.add_argument('--config', type=str,
                            default='text_features_arguments_local.ini',
                            help="Path to the configuration file")
        parser.add_argument('--stories_config', type=str,
                            default='stories.ini',
                            help="Path to the stories configuration file")
        parser.add_argument('--feature_types', type=str,
                            default='hidden_states',
                            help="Comma-separated list of feature types to extract. "
                                 "Options: hidden_states, hidden_state_diff, kl_div_next, "
                                 "log_prob_actual, max_log_prob, entropy_logits, surprisal, "
                                 "perplexity, anticipation_gap, eventboundary_log_prob, eventboundary_log_prob_cs1, punctuation")
        parser.add_argument('--layers', type=str, default=None,
                            help="Comma-separated list of layer indices to extract hidden states from.")
        parser.add_argument('--lookback1', type=int, default=256,
                            help="Initial context size for feature extraction.")
        parser.add_argument('--lookback2', type=int, default=512,
                            help="Maximum context size for feature extraction before resetting.")
        args = parser.parse_args()
    else:
        parser = argparse.ArgumentParser(description="Extract text features for stories.")
        parser.add_argument('--stories', nargs='+', required=True,
                            help="List of stories to process")
        parser.add_argument('--config', type=str,
                            default='encoding-model-scaling-laws/text_features_arguments.ini',
                            help="Path to the configuration file")
        parser.add_argument('--stories_config', type=str,
                            default='encoding-model-scaling-laws/stories.ini',
                            help="Path to the stories configuration file")
        parser.add_argument('--feature_types', type=str,
                            default='hidden_states',
                            help="Comma-separated list of feature types to extract. "
                                 "Options: hidden_states, hidden_state_diff, kl_div_next, "
                                 "log_prob_actual, max_log_prob, entropy_logits, surprisal, "
                                 "perplexity, anticipation_gap, eventboundary_log_prob, eventboundary_log_prob_cs1.")
        parser.add_argument('--layers', type=str, default=None,
                            help="Comma-separated list of layer indices to extract hidden states from.")
        parser.add_argument('--lookback1', type=int, default=256,
                            help="Initial context size for feature extraction.")
        parser.add_argument('--lookback2', type=int, default=512,
                            help="Maximum context size for feature extraction before resetting.")
        args = parser.parse_args()

    t0 = time.time()
    print(f"Using device: {device}")

    # Load config files
    config = configparser.ConfigParser()
    config.read(args.config)
    stories_config = configparser.ConfigParser()
    stories_config.read(args.stories_config)

    # Params from config
    model_name    = config['DEFAULT']['model_name']
    hf_token      = config['DEFAULT'].get('hf_token', None)
    output_dir    = Path(config['DEFAULT']['output_dir'])
    textgrid_dir  = Path(config['DEFAULT']['textgrid_dir'])
    audio_dir     = Path(config['DEFAULT']['audio_dir'])
    text_dir      = Path(config['DEFAULT']['text_dir'])
    avg_tr        = float(config['DEFAULT']['TR'])

    # Feature types
    feature_types = [ft.strip() for ft in args.feature_types.split(',')]
    print(f"Feature types to extract: {feature_types}")

    # Layers
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]
    else:
        if 'layers' in config['DEFAULT']:
            layers = [int(x.strip()) for x in config['DEFAULT']['layers'].split(',')]
        else:
            layers = [-1]
    print(f"Layers to process: {layers}")

    # Audio start times & TR counts
    audio_start_times = {}
    num_TRs           = {}
    for story in args.stories:
        if story in stories_config:
            audio_start_times[story] = float(stories_config[story]['audio_start_time'])
            num_TRs[story]           = int(stories_config[story]['num_trs'])
        else:
            raise ValueError(f"Story '{story}' not found in stories configuration file.")

    # Forced-alignment model
    bundle      = torchaudio.pipelines.MMS_FA
    FA_model    = bundle.get_model().to(device)
    FA_tokenizer= bundle.get_tokenizer()
    FA_aligner  = bundle.get_aligner()

    # Load or generate TextGrids
    grids   = {}
    trfiles = {}
    wordds  = {}
    for story in args.stories:
        print(f"Processing story: {story}", flush=True)
        text_file     = text_dir / f"{story}.txt"
        audio_file    = audio_dir / f"{story}.wav"
        textgrid_path = textgrid_dir / f"{story}.TextGrid"

        if textgrid_path.exists():
            print(f"TextGrid for {story} already exists. Skipping forced alignment.", flush=True)
        else:
            print(f"starting FA for {story}", flush=True)
            with open(text_file, "r") as f:
                text_raw = f.read().replace("\n", " ")
            generate_textgrid_for_story(
                FA_model, FA_aligner, FA_tokenizer,
                story, text_raw, audio_file, textgrid_dir,
                bundle, device=device
            )

        # Load generated TextGrid
        with open(textgrid_path, 'r') as f:
            grids[story] = textgrid.TextGrid(f.read())
        tr_times = np.arange(0, num_TRs[story] * avg_tr, avg_tr)
        trfiles[story] = Simple_TR_File(tr_times, avg_tr)

    # Create word datasets
    wordds = make_word_ds(grids, trfiles)

    # Free FA resources
    del bundle, FA_model
    torch.cuda.empty_cache()

    # Load LLM
    print(f'Loading model {model_name}...')
    #TODO:turn off do_sample False when using non nelwinelogprob features
    bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
        cache_dir=os.getenv("HF_HOME"), quantization_config=bnb,
        low_cpu_mem_usage=False, do_sample=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
        cache_dir=os.getenv("HF_HOME"))
    #model.to(device)
    torch.set_grad_enabled(False)
    #model.eval()
    print(f"Model loaded on device: {model.device}")

    # 1) Standard features via sliding-window
    #    (handles everything *except* eventboundary_log_prob)
    # Split off the “base” features (everything except eventboundary_log_prob)
    base_feature_types = [ft for ft in feature_types if ((ft != 'eventboundary_log_prob') and ft!= ('eventboundary_log_prob_cs1'))]

    # Only run the sliding-window extractor if there’s something to do
    if base_feature_types:
        featureseqs = extract_text_features(
            model=model,
            tokenizer=tokenizer,
            wordds=wordds,
            layers=layers,
            device=device,
            disable_tqdm=False,
            audio_start_times=audio_start_times,
            feature_types=base_feature_types,
            lookback1=args.lookback1,
            lookback2=args.lookback2
        )
    else:
        featureseqs = {}   # nothing else to extract

    # 2) Optional: eventboundary_log_prob via audio-based chunks
    # eventboundary_log_prob via audio-chunks
    maupassant_finn = """Guy de Maupassant Die Hand. ¶ Man drängte sich um den Untersuchungsrichter Bermutier, der seine Ansicht äußerte über den mysteriösen Fall in Saint Cloud. ¶Seit einem Monat entsetzte dies unerklärliche Verbrechen Paris. Niemand konnte es erklären. ¶Herr Bermutier stand, den Rücken gegen den Kamin gelehnt da, sprach, sichtete die Beweisstücke, kritisierte die verschiedenen Ansichten darüber, aber er selbst gab kein Urteil ab. Ein paar Damen waren aufgestanden, um näher zu sein, blieben vor ihm stehen, indem sie an den glattrasierten Lippen des Beamten hingen, denen so ernste Worte entströmten. Sie zitterten und schauerten ein wenig zusammen in neugieriger Angst und dem glühenden unersättlichen Wunsch nach Grauenhaftem, der ihre Seelen quälte und peinigte. Eine von ihnen, bleicher als die anderen, sagte während eines Augenblicks Stillschweigen : – Das ist ja schrecklich! Es ist wie etwas Übernatürliches dabei. Man wird die Wahrheit nie erfahren. Der Beamte wandte sich zu ihr : – Ja, gnädige Frau, wahrscheinlich wird man es nicht erfahren, aber wenn Sie von Übernatürlichem sprechen, so ist davon nicht die Rede. Wir stehen vor einem sehr geschickt ausgedachten und ungemein geschickt ausgeführten Verbrechen, das so mit dem Schleier des Rätselhaften umhüllt ist, daß wir die unbekannten Nebenumstände nicht zu entschleiern vermögen. Aber ich habe früher einmal selbst einen ähnlichen Fall zu bearbeiten gehabt, in den sich auch etwas Phantastisches zu mischen schien. Übrigens mußte man das Verfahren einstellen, da man der Sache nicht auf die Spur kam. Mehrere Damen sagten zu gleicher Zeit, so schnell, daß ihre Stimmen zusammenklangen : – Ach Gott, erzählen Sie uns das! Der Beamte lächelte ernst, wie ein Untersuchungsrichter lächeln muß, und sagte : – Glauben Sie ja nicht, daß ich auch nur einen Augenblick gemeint habe, bei der Sache wäre etwas Übernatürliches. Es geht meiner Ansicht nach alles mit rechten Dingen zu. Aber wenn sie statt ›übernatürlich‹ für das was wir nicht verstehen, einfach ›unaufklärbar‹ sagen, so wäre das viel besser. Jedenfalls interessierten mich bei dem Fall, den ich Ihnen erzählen werde, mehr die Nebenumstände. Es handelte sich etwa um folgendes : ¶ Ich war damals Untersuchungsrichter in Ajaccio, einer kleinen weißen Stadt an einem wundervollen Golf, der rings von hohen Bergen umstanden ist. Ich hatte dort hauptsächlich Vendetta - Fälle zu verfolgen. ¶Es giebt wundervolle, so tragisch wie nur möglich, wild und leidenschaftlich. Dort kommen die schönsten Rächerakte vor, die man sich nur träumen kann, Jahrhunderte alter Haß, nur etwas verblaßt, aber nie erloschen. Unglaubliche Listen, Mordfälle, die zu wahren Massakren, sogar beinahe zu herrlichen Thaten ausarten. ¶Seit zwei Jahren hörte ich nur immer von der Blutrache, diesem furchtbaren, korsischen Vorurteil, das die Menschen zwingt, Beleidigungen nicht bloß an der Person, die sie gethan, zu rächen, sondern auch an den Kindern und Verwandten. Ich hatte ihm Greise, Kinder, Vettern zum Opfer fallen sehen, ich steckte ganz voll solcher Geschichten. ¶ Da erfuhr ich eines Tages, daß ein Engländer auf mehrere Jahre eine im Hintergrund des Golfes gelegene Villa gemietet. Er hatte einen französischen Diener mitgebracht, den er in Marseille gemietet. Bald sprach alle Welt von diesem merkwürdigen Manne, der in dem Haus allein lebte und nur zu Jagd und Fischfang ausging. Er redete mit niemand, kam nie in die Stadt, und jeden Morgen übte er sich ein oder zwei Stunden im Pistolen - oder Karabiner - Schießen. Allerlei Legenden bildeten sich um den Mann. Es wurde behauptet, er wäre eine vornehme Persönlichkeit, die aus politischen Gründen aus seinem Vaterlande entflohen. Dann ging das Gerücht, daß er sich nach einem furchtbaren Verbrechen hier versteckt hielt ; man erzählte sogar grauenvolle Einzelheiten. ¶Ich wollte in meiner Eigenschaft als Untersuchungsrichter etwas über den Mann erfahren, aber es war mir nicht möglich. Er ließ sich Sir John Rowell nennen. Ich begnügte mich also damit, ihn näher zu beobachten, und ich kann nur sagen, daß man mir nichts irgendwie Verdächtiges mitteilen konnte. Aber da die Gerüchte über ihn fortgingen, immer seltsamer wurden und sich immer mehr verbreiteten, so entschloß ich mich, einmal den Fremden selbst zu sehen, und ich begann regelmäßig in der Nähe seines Besitztums auf die Jagd zu gehen. Ich wartete lange auf eine Gelegenheit. ¶ Endlich bot sie sich mir dadurch, daß ich dem Engländer ein Rebhuhn vor der Nase wegschoß. Mein Hund brachte es mir, ich nahm es auf, entschuldigte mich Sir John Rowell gegenüber und bat ihn artig, die Beute anzunehmen. ¶Er war ein großer, rothaariger Mann, mit rotem Bart, sehr breit und kräftig, eine Art ruhiger, höflicher Herkules. Er hatte nichts von der sprüchwörtlichen englischen Steifheit und dankte mir lebhaft für meine Aufmerksamkeit in einem englisch gefärbten Französisch. ¶ Nach vier Wochen hatten wir fünf oder sechs Mal zusammen gesprochen, und ¶ eines Abends, als ich an seiner Thür vorüberkam, sah ich ihn, wie er in seinem Garten rittlings auf einem Stuhl saß und die Pfeife rauchte. Ich grüßte, und er lud mich zu einem Glase Bier ein. Das ließ ich mir nicht zweimal sagen. Er empfing mich mit aller peinlichen englischen Artigkeit, sprach am höchsten Lobeston von Frankreich, von Korsika, und erklärte, er hätte dieses Eiland zu gern. ¶ Da stellte ich ihm mit größter Vorsicht, indem ich lebhaftes Interesse heuchelte, einige Fragen über sein Leben und über seine Absichten. Ohne Verlegenheit antwortete er mir, erzählte mir, er sei sehr viel gereist, in Afrika, Indien und Amerika und fügte lachend hinzu : – O, ich haben viele Abenteuer gehabt, o yes! ¶ Dann sprach ich weiter von der Jagd, und er erzählte mir interessante Einzelheiten über die Nilpferd -, Tiger -, Elephanten - und sogar Gorilla - Jagd. Ich sagte : – Alle diese Tiere sind gefährlich! Er lächelte : – O no, die schlimmste ist die Mensch! Er lachte gemütlich, in seiner behäbigen englischen Art und sagte : – Ich habe auch viel die Mensch gejagt! ¶Dann sprach er von Waffen und forderte mich auf, bei ihm einzutreten, um ein paar Gewehre verschiedener Systeme zu besehen. Das Wohnzimmer war mit schwarzer, gestickter Seide ausgeschlagen, große, gelbe Blumen schlängelten sich über den dunklen Stoff und leuchteten wie Feuer. Er sagte : – Das ist japanische Stickerei! Aber mitten auf der größten Wand zog ein eigentümlicher Gegenstand meine Blicke auf sich. Von vier Ecken mit rotem Sammet umgeben, hob sich etwas Seltsames ab. Ich trat näher. Es war eine Hand. Eine menschliche Hand. Nicht die Hand eines Skelettes mit gebleichten, reinlich präparierten Knochen, sondern eine schwarze, vertrocknete Hand mit gelben Nägeln, bloßliegenden Muskeln und alten Blutspuren von dem glatt abgeschnittenen Knochen, als wäre er mitten im Unterarm mit einem Beile abgehackt. An dem Handgelenk war eine Riesen - Eisenkette befestigt, die mit einem so starken Ring, als wolle man einen Elephant daran binden, die Hand an der Mauer hielt. ¶ Ich fragte : – Was ist denn das? Der Engländer antwortete ganz ruhig : – ¶ Das war meine beste Feind ; sie kam von Amerika. Das ist mit die Säbel abgeschlagen und die Haut mit scharfe Kiesel abgekratzt und acht Tage in die Sonne getrocknet. Aho, sehr fein für mir! ¶ Ich faßte diese menschlichen Überreste, die einem Koloß angehört haben mußten, an. Diese Hand war gräßlich zu sehen, und unwillkürlich drängte sich mir der Gedanke an einen fürchterlichen Racheakt auf. Ich sagte : – Dieser Mann muß sehr stark gewesen sein! Der Engländer antworte ganz weich : – O yes, aber ich war stärker, ich hatte die Kette angebunden, sie zu halten. Ich meinte, er scherze und sagte : – Nun, diese Kette ist ja jetzt unnütz, die Hand wird ja nicht davon laufen. Sir John Rowell antwortete ernst : – Er wollte immer fortlaufen, die Kette war nötig. ¶Mein Blick ruhte fragend auf seinem Gesicht, und ich sagte mir : Ist der Kerl verrückt, oder ist es ein schlechter Witz? Aber sein Gesicht blieb unbeweglich ruhig, voller Wohlwollen, er sprach von anderen Dingen, und ich bewunderte seine Gewehre. Aber ich bemerkte, daß geladene Revolver hier und da auf den Tischen lagen, als ob er in ständiger Furcht vor einem Angriff lebte. ¶ Ich besuchte ihn noch ein paar Mal, dann nicht mehr, man hatte sich an seine Anwesenheit gewöhnt, er war uns allen uninteressant geworden. ¶ Ein ganzes Jahr verstrich, da weckte mich eines Morgens, Ende September, mein Diener mit der Meldung, Sir John Rowell wäre in der Nacht ermordet worden. ¶ Eine halbe Stunde später betrat ich mit dem Gendarmerie - Hauptmann das Haus des Engländers. Der Diener stand ganz verzweifelt vor der Thür und weinte. Ich hatte zuerst den Mann in Verdacht, aber er war unschuldig. ¶Den Schuldigen hat man nie entdecken können. ¶ Als ich in das Wohnzimmer des Sir John Rowell. trat, sah ich auf den ersten Blick mitten in dem Raum die Leiche auf dem Rücken liegen. Die Weste war zerrissen, ein Ärmel hing herab, alles deutete darauf hin, daß ein furchtbarer Kampf stattgefunden hatte. Der Engländer war erwürgt worden, sein schwarzes, gedunsenes Gesicht hatte etwas Gräßliches und schien ein furchtbares Entsetzen auszudrücken. Zwischen den zusammengebissenen Zähnen steckte etwas und sein blutiger Hals war von fünf Löchern durchbohrt, als wären fünf Eisenspitzen dort eingedrungen. ¶ Ein Arzt folgte uns, er betrachtete lange die Fingerspuren im Fleisch und that die seltsame Äußerung : – Das ist ja, als ob er von einem Skelett erwürgt worden wäre. ¶ Ein Schauder lief mir über den Rücken, und ich blickte zur Wand, auf die Stelle, wo ich sonst die entsetzliche Hand gesehen. Sie war nicht mehr da, die Kette hing zerbrochen herab. Da beugte ich mich zu dem Toten nieder und fand in seinem verzerrten Mund einen der Finger dieser verschwundenen Hand. Gerade am zweiten Glied von den Zähnen abgebissen, oder vielmehr abgesägt. ¶Die Untersuchung wurde eingeleitet, man fand nichts, keine Thür war aufgebrochen worden, kein Fenster, kein Möbel. Die beiden Wachthunde waren nicht wach geworden. Die Aussage des Dieners war etwa folgende : ¶ Seit einem Monat schien sein Herr sehr erregt, er hatte viele Briefe bekommen, aber sie sofort wieder verbrannt. Oft nahm er in einem Wutanfall, fast tobsuchtartig, eine Reitpeische und schlug ein auf diese vertrocknete Hand, die an die Mauer geschmiedet und, man weiß nicht wie, zur Stunde, als das Verbrechen geschehen, geraubt worden war. Er ging sehr spät zu Bett und schloß sich jedesmal sorgfältig ein. Er hatte immer Waffen bei der Hand, manchmal sprach er Nachts laut, als zankte er sich mit jemandem. Diese Nacht hatte er aber zufällig keinen Lärm gemacht, und der Diener hatte Sir John erst ermordet vorgefunden, als er die Fenster öffnete. Er hatte niemandem im Verdacht. ¶ Was ich wußte, teilte ich dem Beamten und der Polizei mit, und auf der ganzen Insel wurde sorgfältig nachgeforscht – man entdeckte nichts. ¶Da hatte ich eine Nacht, ein Vierteljahr nach dem Verbrechen, einen furchtbaren Traum. Es war mir, als sähe ich die Hand, die entsetzliche Hand wie einen Skorpion, wie eine Spinne längs der Vorhänge hinhuschen. Dreimal wachte ich auf, dreimal schlief ich wieder ein, dreimal sah ich dieses entsetzliche Überbleibsel um mein Zimmer herumjagen, indem es die Finger wie Pfoten bewegte. ¶Am nächsten Tage brachte man mir die Hand, die man auf dem Kirchhof, wo Sir John Rowell begraben war, da man seine Familie nicht eruiert hatte, auf seinem Grabe gefunden hatte. Der Zeigefinger fehlte. ¶ Das, meine Damen, ist meine Geschichte, mehr weiß ich nicht. ¶Die Damen waren bleich geworden, zitterten, und eine von ihnen rief : – Aber das ist doch keine Lösung und keine Erklärung, wir können ja garnicht schlafen, wenn Sie uns nicht sagen, was Ihrer Ansicht nach passiert ist. Der Beamte lächelte ernst : – O meine Damen, ich will Sie gewiß nicht um Ihre schönsten Träume bringen, ich denke ganz einfach, daß der Besitzer dieser Hand gar nicht tot war und daß er einfach gekommen ist, um sie mit der Hand wieder zu holen, die ihm übrig geblieben war ; aber ich weiß nicht, wie er das angestellt hat. Das wird eine Art Vendetta sein. Eine der Damen flüsterte : – Nein, das kann nicht so gewesen sein! Und der Untersuchungsrichter schloß immer noch lächelnd : – Ich habe es Ihnen doch gesagt, daß meine Erklärung Ihnen nicht passen würde."""
    if 'eventboundary_log_prob' in feature_types or 'eventboundary_log_prob_cs1' in feature_types:
        LOOKAHEAD = 0
        EVENT_MARKER = "¶"
        marker_id = tokenizer(EVENT_MARKER, add_special_tokens=False).input_ids[0]
        punct_chars = string.punctuation + "„“”‚‘…–—"
        punct_re = re.compile(f"([{re.escape(punct_chars)}])")

        featureseqs['eventboundary_log_prob_cs1'] = {}

        for story in args.stories:
            print(f"\n=== Event‑boundary extraction for {story} ===", flush=True)

            # (A) Raw model transcript (punctuated)
            raw_text = (text_dir / f"{story}.txt").read_text(encoding="utf8")
            model_words = raw_text.replace("\n", " ").split()

            # (B) FA‐based DataSequence
            spoken_ds = wordds[story]
            spoken_words = list(spoken_ds.data)  # lowercased, no punctuation
            print("spoken words:", len(spoken_words),
                  "model words:", len(model_words))

            # (C) Run boundary_lp on model words (fake uniform times, real TRs)
            ds_model = DataSequence(
                np.array(model_words),
                list(range(len(model_words))),
                np.arange(len(model_words), dtype=float),
                spoken_ds.tr_times
            )
            lp_tr_full, lp_word_full, _ = boundary_lp(
                ds_model, tokenizer, model,
                lookahead=LOOKAHEAD, device=device
            )

            # (D) Clean & normalize for exact token match
            model_clean = [romanize_and_clean(w) for w in model_words]
            spoken_clean = [normalize_uroman(w) for w in spoken_words]

            # ─────────────────────────────────────────────────────────
            # (E) EXPLICIT REMAPPING: model→FA word align via align_indices
            # ─────────────────────────────────────────────────────────
            # Build a list of *all* original word indices (1‑based) so we
            # get a mapping for every model word that matches an FA word.
            wmap = build_word_index_mapping(model_clean, spoken_clean)
            print(f"Aligned {len(wmap)}/{len(model_clean)} model words → spoken words "
                  f"({100 * len(wmap) / len(model_clean):.1f}%)")

            for orig_i, fa_i in list(wmap.items())[:100]:
                print(f"model word #{orig_i:3d}='{model_clean[orig_i]}' → "
                      f"spoken #{fa_i:3d}='{spoken_clean[fa_i]}'")
            # 2) Remap the entire lp_word_full vector onto FA grid
            fa_lp_word = np.full(len(spoken_clean), np.nan, dtype=np.float32)
            for orig_i, fa_i in wmap.items():
                fa_lp_word[fa_i] = lp_word_full[orig_i]

            # 3) Fill NaNs by linear interpolation
            mask = np.isnan(fa_lp_word)
            if mask.any():
                valid = np.flatnonzero(~mask)
                first, last = valid[0], valid[-1]
                fa_lp_word[:first] = fa_lp_word[first]
                fa_lp_word[last + 1:] = fa_lp_word[last]
                miss = np.flatnonzero(
                    mask &
                    (np.arange(len(fa_lp_word)) > first) &
                    (np.arange(len(fa_lp_word)) < last)
                )
                fa_lp_word[miss] = np.interp(miss, valid, fa_lp_word[valid])
            assert not np.isnan(fa_lp_word).any()

            # (G) Build DataSequence for TR‐downsample (with real FA times)
            # shift everything into fMRI clock if needed:
            shifted_word_times = spoken_ds.data_times + audio_start_times[story]
            shifted_tr_times = spoken_ds.tr_times + audio_start_times[story]

            ds_boundary = DataSequence(
                fa_lp_word,
                spoken_ds.split_inds,
                shifted_word_times,
                shifted_tr_times
            )

            # Ensure featureseqs has keys for each new filter
            featureseqs.setdefault("eventboundary_log_prob_rect", {})
            featureseqs.setdefault("eventboundary_log_prob_sinc3", {})
            featureseqs.setdefault("eventboundary_log_prob_sinc1", {})
            featureseqs.setdefault("eventboundary_log_prob_chunksum3", {})
            featureseqs.setdefault("eventboundary_log_prob_max", {})
            featureseqs.setdefault("eventboundary_log_prob_mean", {})

            lp_tr = ds_boundary.chunksums("lanczos", window=1)
            featureseqs["eventboundary_log_prob_cs1"][story] = lp_tr

            # 1) Rectangular (sum) downsampling
            chunks = ds_boundary.chunks()
            lp_tr_rect = np.array([chunk.sum() if len(chunk) > 0 else 0.0 for chunk in chunks], dtype=np.float32)
            featureseqs["eventboundary_log_prob_rect"][story] = lp_tr_rect

            # 2) Sinc interpolation (3‐lobe)
            lp_tr_sinc3 = ds_boundary.chunksums("sinc", window=3)
            featureseqs["eventboundary_log_prob_sinc3"][story] = lp_tr_sinc3

            # 2b) Sinc interpolation (1‐lobe)
            lp_tr_sinc3 = ds_boundary.chunksums("sinc", window=1)
            featureseqs["eventboundary_log_prob_sinc1"][story] = lp_tr_sinc3

            # 3) Lanczos interpolation (3‐lobe) — "chunksum3"
            lp_tr_chunksum3 = ds_boundary.chunksums("lanczos", window=3)
            featureseqs["eventboundary_log_prob_chunksum3"][story] = lp_tr_chunksum3

            # 4) Max‐per‐TR aggregator
            #    (for each TR, take the maximum word‐level score in that chunk)
            chunks = ds_boundary.chunks()
            lp_tr_max = np.array([chunk.max() if len(chunk) > 0 else 0.0 for chunk in chunks], dtype=np.float32)
            featureseqs["eventboundary_log_prob_max"][story] = lp_tr_max


            # 5) Mewan‐per‐TR aggregator
            #    (for each TR, take the maximum word‐level score in that chunk)
            try:
                chunks = ds_boundary.chunks()
                lp_tr_mean = np.array([chunk.mean() if len(chunk) > 0 else 0.0 for chunk in chunks], dtype=np.float32)
                featureseqs["eventboundary_log_prob_mean"][story] = lp_tr_mean
            except Exception as e:
                print("exception: ",e)



            # (H) Plot + overlay Finn’s hand‐annotated markers
            #    first re‐extract Finn EB indices from the original
            #    (you already have maupassant_finn string above)
            _, EB_finn = strip_eb_markers(maupassant_finn)
            # map those 1‑based word IDs to Tru TR‐indices
            finnpred_tr = [
                np.argmin(np.abs(ds_boundary.tr_times - shifted_word_times[w - 1]))
                for w in EB_finn
                if 1 <= w <= len(shifted_word_times)
            ]

            # right after you finish fa_lp_word interpolation
            plt.figure(figsize=(12, 3))
            plt.plot(fa_lp_word, lw=1, label="fa_lp_word (word‑level)")
            # remap Finn EB → FA word indices (1‑based → 0‑based for plotting)
            _, EB_finn = strip_eb_markers(maupassant_finn)
            fa_EB_finn = align_indices(
                ref_words=spoken_clean,
                var_words=model_clean,
                var_EB=EB_finn
            )
            # ─────────────────────────────────────────────────────────
            # TWO‑STAGE FINN‑MARKER ALIGNMENT + SAVED PLOTS
            # ─────────────────────────────────────────────────────────

            # 1) Finn markers in original maupassant_finn → model words
            raw_finn_words, EB_finn_raw = strip_eb_markers(maupassant_finn)
            finn_clean = [romanize_and_clean(w) for w in raw_finn_words]
            EB_model = align_indices(
                ref_words=model_clean,
                var_words=finn_clean,
                var_EB=EB_finn_raw
            )

            # 2) Then model words → FA (spoken) words
            fa_EB_finn = align_indices(
                ref_words=spoken_clean,
                var_words=model_clean,
                var_EB=EB_model
            )

            story_dir = output_dir / story
            story_dir.mkdir(parents=True, exist_ok=True)

            finn_tr_idxs = [
                np.argmin(np.abs(shifted_tr_times - shifted_word_times[w - 1]))
                for w in fa_EB_finn
            ]

            # Write to pickle
            pkl_path_x = story_dir / "finn_markers_downsampled_x.pkl"
            pkl_path_y = story_dir / "finn_markers_downsampled_y.pkl"
            finn_x =shifted_tr_times[finn_tr_idxs]
            finn_y = lp_tr[finn_tr_idxs]
            with open(pkl_path_x, "wb") as pf:
                pkl.dump(finn_x, pf)
            print(f"Saved finn markers → {pkl_path_x}")
            with open(pkl_path_y, "wb") as pf:
                pkl.dump(finn_y, pf)
            print(f"Saved finn markers → {pkl_path_y}")

            # — WORD‑LEVEL PLOT —
            plt.figure(figsize=(12, 3))
            plt.plot(fa_lp_word, lw=1, label="fa_lp_word (word‑level)")
            xw = [i - 1 for i in fa_EB_finn]
            yw = [fa_lp_word[i - 1] for i in fa_EB_finn]
            plt.scatter(xw, yw, c="C1", marker="x", s=60, label="Finn EB")
            plt.xlabel("word index")
            plt.ylabel("score")
            plt.legend()
            plt.tight_layout()
            outpath = output_dir / story / f"{story}_boundary_wFINN_WORD_CS1.png"
            plt.savefig(outpath)

            # — TR‑LEVEL PLOT (real TRs) —
            plt.figure(figsize=(12, 2.5))
            plt.plot(shifted_tr_times, lp_tr, lw=1, label="lp_tr (TR‑level)")
            finn_tr_idxs = [
                np.argmin(np.abs(shifted_tr_times - shifted_word_times[w - 1]))
                for w in fa_EB_finn
            ]
            plt.scatter(
                shifted_tr_times[finn_tr_idxs],
                lp_tr[finn_tr_idxs],
                c="C1", marker="x", s=60, label="Finn EB"
            )
            plt.xlabel("time (s)")
            plt.ylabel("score")
            plt.legend()
            plt.tight_layout()
            outpath = output_dir / story / f"{story}_boundary_wFINN_TR_CS1.png"
            plt.savefig(outpath)

            # — OVERLAY FULL MODEL LOG‑P ON TR GRID (for comparison) —
            plt.figure(figsize=(14, 4))
            plt.plot(ds_boundary.tr_times, lp_tr, lw=1, label="model log p(¶)")
            plt.scatter(
                ds_boundary.tr_times[finn_tr_idxs],
                lp_tr[finn_tr_idxs],
                marker="x", s=100, c="C1", label="Finn EB"
            )
            plt.xlabel("time (s)")
            plt.ylabel("log p(¶)")
            plt.legend()
            plt.tight_layout()
            outpath = output_dir / story / f"{story}_TR_boundary_wFINN_remapped_CS1.png"
            plt.savefig(outpath, dpi=150)
            plt.close()

            print(f"  ↳ saved plots to {output_dir / story}")

    print("featureseqs = ", featureseqs)

    print("featureseqs = ", featureseqs)

    # SAVE TIMING INFO PER STORY (using wordds, not featureseqs)
    for story in args.stories:
        # wordds[story] is a DataSequence with the timing info you need
        ds_any = wordds[story]

        # make sure the output folder exists
        story_dir = output_dir / story
        story_dir.mkdir(parents=True, exist_ok=True)

        # extract timing arrays
        times = ds_any.data_times  # per-word or per-prefix timestamps
        tr_times = ds_any.tr_times  # TR onset times
        split_inds = np.array(ds_any.split_inds, dtype=int)

        # save them to a separate file
        tfile = story_dir / "times_info.npz"
        np.savez_compressed(
            tfile,
            times=times,
            tr_times=tr_times,
            split_inds=split_inds
        )
        print(f"Saved timing info → {tfile}")
    # Save all features
    for key in featureseqs:
        feature_type = key if isinstance(key, str) else key[0]
        layer = None if isinstance(key, str) else key[1]
        for story, ds in featureseqs[key].items():
            story_output_dir = Path(config['DEFAULT']['output_dir']) / story
            story_output_dir.mkdir(parents=True, exist_ok=True)

            # Extract numpy array
            feats = ds.data.numpy() if isinstance(ds.data, torch.Tensor) else np.array(ds.data)

            if layer is not None:
                filename       = f"final_outputs_{feature_type}_layer{layer}_context_{args.lookback1}_{args.lookback2}.npz"
            else:
                filename       = f"final_outputs_{feature_type}_context_{args.lookback1}_{args.lookback2}.npz"

            np.savez_compressed(story_output_dir / filename, features=feats)

    t1 = time.time()
    print(f"Feature extraction completed in {t1 - t0} seconds ({(t1 - t0) / 60:.2f} minutes).")
