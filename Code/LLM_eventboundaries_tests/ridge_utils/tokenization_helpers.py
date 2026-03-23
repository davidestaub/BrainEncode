import logging
import sys
import joblib
import matplotlib.pyplot as plt
from ridge_utils.DataSequence import DataSequence
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import string


def shift_extracted_features(audio_start_time,times):
    shifted_times = times + audio_start_time  # Align the times with the BOLD signal
    return shifted_times

def is_new_word_marker(tok_str: str) -> bool:
    """True iff the token string carries a beginning-of-word marker."""
    #Extend this if a new model comes out that labels newwords with a new token, this should work for all llama and GPT style models
    return tok_str.startswith("Ġ") or tok_str.startswith("▁")


def is_punctuation(word):
    """Return True if the word is composed solely of punctuation characters."""
    return all(ch in string.punctuation for ch in word)


def normalize_expected_words(ds, tokenizer):
    """
    Normalize the expected words from ds.data by encoding and decoding each word.
    This forces the forced‐alignment words to use the same subword splitting as the tokenizer.
    """
    normalized = []
    for w in ds.data:
        w = w.strip()
        if not w:
            continue
        toks = tokenizer.encode(w, add_special_tokens=False)
        detok = tokenizer.decode(toks).strip()
        normalized.append(detok.lower())
    return normalized


def compute_accumulator(ds, tokenizer):
    """
    Compute an accumulator (acc) from ds.data (a list of words from forced alignment)
    using the tokenizer. This accumulator is used to generate keys for feature extraction.
    We ignore tokens that decode solely to punctuation when deciding word boundaries.
    """
    full_text = " ".join(ds.data)
    inputs = tokenizer([full_text], return_tensors="pt")
    tokens = np.array(inputs['input_ids'][0])

    # Ensure our dummy boundary token (29947) does not appear in the original token stream.
    assert 29947 not in tokens, "Dummy token 29947 found in tokens."
    acc = [1]  # start with special START token

    boundaries = []
    not_boundaries = []

    force_next_as_boundary = False
    for token in tokens:
        token_str = tokenizer.convert_ids_to_tokens(torch.tensor([token]))[0]
        token_text = token_str[1:] if is_new_word_marker(token_str) else token_str

        if token in tokenizer.all_special_ids:
            continue

        if token_text.strip() == "":
            # It's whitespace — skip it and force the next token to be a boundary
            print("⚠️ whitespace-only token detected, skipping and forcing next as boundary")
            force_next_as_boundary = True
            continue

        if force_next_as_boundary or is_new_word_marker(token_str) or (len(acc) == 1 and not is_punctuation(token_text)):
            acc.append(29947)
            acc.append(token)
            #print("inserting boundary", token_text)
            boundaries.append(token_text)
            force_next_as_boundary = False  # reset
        else:
            #print("not a new word token", token_text)
            not_boundaries.append(token_text)
            acc.append(token)

    #if acc[-1] != 29947:
       # acc.append(29947)

   # print("boundaries:", boundaries)
   # print("not boundaries", not_boundaries)
    return acc


def merge_reconstructed_words(expected, reconstructed):
    """
    Given a list of expected words and a list of words reconstructed from the accumulator,
    try to merge adjacent tokens in 'reconstructed' so that the merged version equals the expected word.
    """
    merged = []
    i = 0
    while i < len(reconstructed):
        # Get the next expected word (if available)
        exp = expected[len(merged)].lower() if len(merged) < len(expected) else ""
        candidate = reconstructed[i].strip().lower()
        if candidate == exp:
            merged.append(candidate)
            i += 1
        else:
            j = i + 1
            found = False
            while j <= len(reconstructed):
                candidate = ' '.join(reconstructed[i:j]).strip().lower()
                if candidate == exp:
                    merged.append(candidate)
                    i = j
                    found = True
                    break
                j += 1
            if not found:
                merged.append(reconstructed[i].strip().lower())
                i += 1
    return merged


def split_merged_words(word_list):
    """
    Given a list of words, split any element that contains a whitespace
    into separate words.
    """
    new_list = []
    for word in word_list:
        if " " in word:
            parts = word.split()
            new_list.extend(parts)
        else:
            new_list.append(word)
    return new_list


def adjust_accumulator(acc, ds, tokenizer):
    """
    Reconstruct words from the accumulator (acc) by splitting on the dummy boundary (29947)
    and filtering out punctuation-only tokens. Then, split any merged words (i.e. those that
    contain whitespace) and compare with the normalized expected words. If the count is off by only a few words,
    extra boundaries are appended.
    """
    def reconstruct(acc):
        words = []
        current = []
        for token in acc:
            if token == 29947:
                if current:
                    decoded = tokenizer.decode(torch.tensor(current)).strip()
                    if decoded and not is_punctuation(decoded):
                        words.append(decoded.lower())
                    current = []
            else:
                current.append(token)
        if current:
            decoded = tokenizer.decode(torch.tensor(current)).strip()
            if decoded and not is_punctuation(decoded):
                words.append(decoded.lower())
        return words

    rec_words = reconstruct(acc)
    # Apply splitting on merged words.
    merged_rec = split_merged_words(rec_words)
    expected_words = normalize_expected_words(ds, tokenizer)
    print("Expected normalized words:", expected_words)
    print("Reconstructed words (pre-split):", rec_words)
    print("Reconstructed words (post-split):", merged_rec)
    rec_count = len(merged_rec)
    expected_count = len(expected_words)
    print(f"After merging & splitting, accumulator word count: {rec_count}, expected: {expected_count}")
    #if rec_count < expected_count and (expected_count - rec_count) < 10:
       # diff = expected_count - rec_count
        #print(f"Accumulator is short by {diff} words; appending extra boundaries.")
       # for _ in range(diff):
         #   acc.append(29947)
        #rec_words = reconstruct(acc)
        #merged_rec = split_merged_words(rec_words)
        #rec_count = len(merged_rec)
        #print(f"After appending, accumulator word count: {rec_count}")
    if rec_count != expected_count:
        print("recc count",merged_rec)
        print("expected count",expected_words)
        raise ValueError(f"After merging & splitting, accumulator word count {rec_count} does not match expected {expected_count}.")

   # if not acc or acc[-1] != 29947:
        #TODO check if this is correct
        #print(acc[-1])
       # acc.append(29947)
    return acc


def compute_correct_tokens_llama(acc, acc_lookback, acc_offset, total_len):
    new_tokens = [1]
    acc_count_all = 0
    first_word = max(0, acc_offset - acc_lookback)
    last_word = min(acc_offset + 1, total_len)

    #print("===" * 20)
    #print(f"acc_offset: {acc_offset}, acc_lookback: {acc_lookback}")
    #print(f"Computed first_word index: {first_word}, last_word index: {last_word}")
    expected_boundaries = last_word - first_word
    #print(f"Total expected boundaries in this segment: {expected_boundaries}")

    boundary_count = 0
    while boundary_count != (first_word + 1):
        if acc_count_all >= len(acc):
            raise ValueError("Reached end of accumulator while searching for the starting boundary.")
        if acc[acc_count_all] == 29947:
            boundary_count += 1
        acc_count_all += 1

    acc2 = acc[acc_count_all:]
    #print(f"Length of original acc: {len(acc)}")
    #print(f"Starting index in acc for this segment: {acc_count_all}")
    #print(f"Length of acc2 (sliced accumulator): {len(acc2)}")

    acc_boundaries = 0
    token_index = 0
    while acc_boundaries != expected_boundaries and token_index < len(acc2):
        if acc2[token_index] == 29947:
            acc_boundaries += 1
        else:
            new_tokens.append(acc2[token_index])
        token_index += 1

    #print(f"Collected boundaries: {acc_boundaries} (expected: {expected_boundaries})")
    #print(f"Total tokens processed in acc2: {token_index}")

    if acc_boundaries != expected_boundaries:
        missing = expected_boundaries - acc_boundaries
        if token_index == len(acc2) and missing <= 2:  # Temporarily allow 2 boundaries missing
            print(f" Warning: Missing {missing} boundary tokens. Appending automatically.")
            new_tokens.extend([29947] * missing)
            acc_boundaries += missing
        else:
            print(f"Critical boundary mismatch detected: missing {missing} boundaries")
            raise ValueError(
                f"Accumulator segment incomplete: expected {expected_boundaries} boundaries, "
                f"but found only {acc_boundaries}. Processed {token_index} tokens out of {len(acc2)} in acc2."
            )

    return new_tokens


def generate_efficient_feat_dicts_llama_NEW(wordseqs, tokenizer, lookback1, lookback2):
    import torch
    import numpy as np

    text_dict = {}
    text_dict2 = {}
    text_dict3 = {}

    for es, story in enumerate(wordseqs.keys()):
       # print("=" * 80)
       # print(f"Processing story: {story}, index: {es}")
        ds = wordseqs[story]
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert 29947 not in tokens, "Dummy token 29947 found in tokens."


        acc = compute_accumulator(ds, tokenizer)
       # print("acc post compute", acc,len(acc),acc.count(29947))
        acc = adjust_accumulator(acc, ds, tokenizer)
       # print("acc post adjust", acc, len(acc),acc.count(29947))

        # Diagnostics: Check word counts explicitly
        expected_words = normalize_expected_words(ds, tokenizer)
        actual_boundaries = acc.count(29947)

        print("Expected normalized words count:", len(expected_words),expected_words)
        print("Actual boundaries in acc:", actual_boundaries,acc)

        # Reconstruct words explicitly for comparison
        rec_words_list = []
        current_tokens = []
        for token in acc:
            if token == 29947:
                if current_tokens:
                    decoded = tokenizer.decode(torch.tensor(current_tokens)).strip().lower()
                    if decoded and not is_punctuation(decoded):
                        rec_words_list.append(decoded)
                    current_tokens = []
            else:
                current_tokens.append(token)

        # If tokens remain at end, decode them
        if current_tokens:
            decoded = tokenizer.decode(torch.tensor(current_tokens)).strip().lower()
            if decoded and not is_punctuation(decoded):
                rec_words_list.append(decoded)

        merged_rec = merge_reconstructed_words(expected_words, rec_words_list)
        merged_rec = split_merged_words(merged_rec)

        print(f"After merging & splitting, accumulator word count: {len(merged_rec)}, expected: {len(expected_words)}")

        if len(expected_words) != len(merged_rec):
            print(" Mismatch in expected words vs reconstructed words")
            for idx, (exp_word, rec_word) in enumerate(zip(expected_words, merged_rec)):
                if exp_word != rec_word:
                    print(f"Mismatch at idx {idx}: expected '{exp_word}', reconstructed '{rec_word}'")
            raise ValueError("Mismatch between expected words and reconstructed words!")

        if len(expected_words) != actual_boundaries:
            print("Mismatch in expected words vs accumulator boundaries ")
            print(f"Difference: {len(expected_words) - actual_boundaries} words")
            # Print example mismatch if boundary counts mismatch
            min_len = min(len(expected_words), len(merged_rec))
            for idx in range(min_len):
                if expected_words[idx] != merged_rec[idx]:
                    print(
                        f"First mismatch at idx {idx}: expected '{expected_words[idx]}', reconstructed '{merged_rec[idx]}'")
                    break
            raise ValueError("Mismatch between expected words and accumulator boundaries!")

        # Build feature dictionary keys using sliding-window logic
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [1]

        for i, w in enumerate(ds.data):
            if w.strip() != '':
                if acc_lookback < lookback1 or (lookback2 > acc_lookback >= lookback1):
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    text_dict3[tuple(new_tokens)] = False
                else:
                    raise AssertionError("Unexpected acc_lookback value!")
            else:
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                text_dict3[tuple(new_tokens)] = False
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True

    return text_dict, text_dict2, text_dict3


def convert_to_feature_mats_llama_NEW(wordseqs, tokenizer, lookback1, lookback2, text_dict3, audio_start_time):
    """
    Convert feature dictionaries into matrices.
    This function uses the same accumulator (computed and adjusted) to extract feature vectors
    and then shifts times by the audio start time.
    """
    featureseqs = {}
    for story in wordseqs.keys():
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert 29947 not in tokens, "Dummy token 29947 found in tokens."

        acc = compute_accumulator(ds, tokenizer)
        acc = adjust_accumulator(acc, ds, tokenizer)

        # Diagnostic: reconstruct accumulator words.
        rec_words_list = []
        current_tokens = []
        for token in acc:
            if token == 29947:
                if current_tokens:
                    decoded = tokenizer.decode(torch.tensor(current_tokens)).strip()
                    if decoded and not is_punctuation(decoded):
                        rec_words_list.append(decoded.lower())
                    current_tokens = []
            else:
                current_tokens.append(token)
        if current_tokens:
            decoded = tokenizer.decode(torch.tensor(current_tokens)).strip()
            if decoded and not is_punctuation(decoded):
                rec_words_list.append(decoded.lower())
       # print("Final reconstructed accumulator words (for conversion):")
       # for idx, word in enumerate(rec_words_list):
        #    print(f"  {idx}: '{word}'")
        merged_rec = merge_reconstructed_words(normalize_expected_words(ds, tokenizer), rec_words_list)
        #print("merged rec:",merged_rec)
        #print("normalize_expected_words(ds, tokenizer)",normalize_expected_words(ds, tokenizer))
        rec_count = len(merged_rec)
        expected_count = len(normalize_expected_words(ds, tokenizer))
       # print(f"Total annotations (merged) from acc: {rec_count}")
       # print(f"Total expected normalized words: {expected_count}")
        if rec_count != expected_count:
            raise ValueError(f"Mismatch in reconstructed words: {rec_count} != {expected_count}")

        # Build feature vectors using the computed accumulator.
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [1]
        feature_dim = None
        print_idx = 0
        for i, w in enumerate(ds.data):
            if w.strip() != '':
                if acc_lookback < lookback1 or (lookback2 > acc_lookback >= lookback1):
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    feature_vector = text_dict3.get(tuple(new_tokens))
                  #  if print_idx % 1000 == 0:
                    #    print("text_dict 3 = ",text_dict3,"tuple = ",tuple(new_tokens),"new tokens = ",new_tokens)
                    #    print("text_dict3.get(tuple(new_tokens)) aka feature vector",feature_vector)
                    #    print_idx = print_idx+1

                    if feature_vector is not None:
                        newdata.append(feature_vector)
                        if feature_dim is None:
                            feature_dim = feature_vector.shape[0]
                    else:
                        if feature_dim is None:
                            raise ValueError("Feature vector missing and feature dimension unknown.")
                        placeholder_vector = np.full((feature_dim,), np.nan)
                        newdata.append(placeholder_vector)
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    feature_vector = text_dict3.get(tuple(new_tokens))
                    if feature_vector is not None:
                        newdata.append(feature_vector)
                        if feature_dim is None:
                            feature_dim = feature_vector.shape[0]
                    else:
                        if feature_dim is None:
                            raise ValueError("Feature vector missing and feature dimension unknown.")
                        placeholder_vector = np.full((feature_dim,), np.nan)
                        newdata.append(placeholder_vector)
                else:
                    raise AssertionError("Unexpected acc_lookback value")
            else:
                feature_vector = text_dict3.get(tuple(new_tokens))
                if feature_vector is not None:
                    newdata.append(feature_vector)
                    if feature_dim is None:
                        feature_dim = feature_vector.shape[0]
                else:
                    if feature_dim is None:
                        raise ValueError("Feature vector missing and feature dimension unknown.")
                    placeholder_vector = np.full((feature_dim,), np.nan)
                    newdata.append(placeholder_vector)
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1

        shifted_times = shift_extracted_features(audio_start_time, ds.data_times)
        featureseqs[story] = DataSequence(np.array(newdata), ds.split_inds, shifted_times, ds.tr_times)
    downsampled_featureseqs = {}
    for story in featureseqs:
        downsampled_featureseqs[story] = featureseqs[story].chunksums('lanczos', window=3)
    return downsampled_featureseqs





