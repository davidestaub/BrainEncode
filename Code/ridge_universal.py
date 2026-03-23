#### Dependencies ####
import sys
print(sys.executable, flush=True)
import numpy as np
import time
from ridge_utils.ridge import bootstrap_ridge
import ridge_utils.npp
from ridge_utils.util import make_delayed
import pickle
from pathlib import Path
from ridge_utils.interpdata import lanczosinterp2D
import configparser
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def compute_null_corrs(i, chunks, pred):
    # Shuffle a copy of the chunks to avoid modifying the original
    shuffled_chunks = chunks.copy()
    np.random.shuffle(shuffled_chunks)
    shuffled_response = np.concatenate(shuffled_chunks)
    shuffled_response = np.expand_dims(shuffled_response, axis=0)
    _, _, _, corrs_unnorm = spe_and_cc_norm(shuffled_response, pred, max_flooring=0.25)
    return corrs_unnorm


def compute_null_corrs_single(chunks, pred_ts):
    # make a shallow copy of the list of arrays
    chunks_copy = chunks.copy()
    # shuffle in‑place without coercing to one big ndarray
    np.random.shuffle(chunks_copy)
    # concatenate and compute Pearson r
    shuffled = np.concatenate(chunks_copy)
    return np.corrcoef(shuffled, pred_ts)[0, 1]


def shift_extracted_features(audio_start_time, times):
    shifted_times = times + audio_start_time  # Align the times with the BOLD signal
    return shifted_times


def spe_and_cc_norm(orig_data, data_pred, data_norm=True, max_flooring=None):
    '''
    Computes the signal power explained and the cc_norm of a model given the observed and predicted values
    Assumes normalization unless data_norm is set to False

    orig_data: 3D numpy array (trials, timepoints, voxels)

    data_pred: 2D numpy array (timepoints, voxels)

    data_norm: bool -> Set to False if not pre-normalized

    max_flooring: None/float (0-1) -> If not None, compute cc_norm in an alternate way that floors cc_max by max_flooring.
    This is helpful to clean up bad voxels that are not at all language selective.

    According to Schoppe: https://www.frontiersin.org/articles/10.3389/fncom.2016.00010/full
    '''
    y = np.mean(orig_data, axis=0)
    num_trials = len(orig_data)
    if num_trials == 1:
        print("Correct it is only one trial")
        y_flip = np.swapaxes(y, axis1=0, axis2=1)
        data_flip = np.swapaxes(data_pred, axis1=0, axis2=1)
        corrs = np.zeros(y_flip.shape[0])
        for i, row in enumerate(y_flip):
            corrs[i] = np.corrcoef(y_flip[i], data_flip[i])[0][1]
        SPE, cc_norm, cc_max = 0, 0, 0
    else:
        print("This is not good, ", num_trials)

        if not data_norm:
            variance_across_time = np.var(orig_data, axis=1, ddof=1)
            TP = np.mean(variance_across_time, axis=0)
        else:
            TP = np.zeros(orig_data.shape[2]) + 1

        SP = (1 / (num_trials - 1)) * ((num_trials * np.var(y, axis=0, ddof=1)) - TP)

        SPE_num = (np.var(y, axis=0, ddof=1) - np.var(y - data_pred, axis=0, ddof=1))
        SPE = (np.var(y, axis=0, ddof=1) - np.var(y - data_pred, axis=0, ddof=1)) / SP

        y_flip = np.swapaxes(y, axis1=0, axis2=1)
        data_flip = np.swapaxes(data_pred, axis1=0, axis2=1)
        covs = np.zeros(y_flip.shape[0])
        for i, row in enumerate(y_flip):
            covs[i] = np.cov(y_flip[i], data_flip[i])[0][1]
        cc_norm = np.sqrt(1 / SP) * (covs / np.sqrt(np.var(data_pred, axis=0, ddof=1)))
        cc_max = None
        if max_flooring is not None:
            cc_max = np.nan_to_num(1 / (np.sqrt(1 + ((1 / num_trials) * ((TP / SP) - 1)))))
            # cc_max = np.maximum(cc_max, np.zeros(cc_max.shape) + max_flooring)
            corrs = np.zeros(y_flip.shape[0])
            for i, row in enumerate(y_flip):
                corrs[i] = np.corrcoef(y_flip[i], data_flip[i])[0][1]
            cc_norm = corrs / cc_max
    return SPE, cc_norm, cc_max, corrs


def process_arguments(args):
    """
    Positional‑argument order **after this patch**

    0  script name                    (handled by sys.argv)
    1  input_type     'audio' | 'text'
    2  feature_type   e.g. 'context_separation'
    3  feature_tag    '_TAG...' | 'None'
    4  layer          int | 'None'
    5  roi_arg        'ALL' | mask.nii.gz[,mask2.nii.gz]
    6‑ …  fold flags  (‑1 / 0 / 1)  — exactly one '1'
    """
    # --------------------------------------------------------------
    input_type   = str(args[1])
    feature_type = str(args[2])
    feature_tag  = '' if args[3] == 'None' else str(args[3])
    layer_arg    = str(args[4])
    roi_arg      = str(args[5])          # keep raw string for later
    fold_flags   = [int(x) for x in args[6:]]

    # ---- basic validation ----------------------------------------
    assert input_type in ('audio', 'text'), "input_type must be 'audio' or 'text'"

    text_feats = {'hidden_states', 'log_prob_actual', 'max_log_prob',
                  'entropy_logits', 'surprisal', 'perplexity',
                  'anticipation_gap', 'hidden_state_diff',
                  'kl_div_next', 'context_novelty', 'diff_energy',
                  'hidden_state_diff_norm', 'similarity_entropy',
                  'velocity', 'context_separation',
                  'reinstatement', 'context_drift',
                  'newline_log_prob','stack_sep-drift-reinst','eventboundary_log_prob'}
    #Removed the constraint
    if input_type == 'audio':
        assert feature_type == 'hidden_states', "audio mode only supports 'hidden_states'"


      #  if input_type == 'text':
        #    assert feature_type in text_feats, f"invalid feature_type {feature_type}"
       # else:
           # assert feature_type == 'hidden_states', "audio mode only supports 'hidden_states'"

    # --- layer checks ---------------------------------------------
    layer = None if layer_arg == 'None' else layer_arg
    layer_needed = feature_type in {
        'hidden_states', 'hidden_state_diff', 'context_novelty', 'diff_energy',
        'hidden_state_diff_norm', 'similarity_entropy', 'velocity',
        'context_separation', 'reinstatement', 'context_drift'
    }
    if layer_needed:
        assert layer is not None, "layer cannot be None for this feature"
        assert layer.isdigit(), "layer must be an integer"
        layer = int(layer)
    else:
        assert layer is None, "layer must be None for this feature"

    # --- fold flags -----------------------------------------------
    valid = {-1, 0, 1}
    assert all(f in valid for f in fold_flags), "fold flags must be -1, 0 or 1"
    assert fold_flags.count(1) == 1, "exactly one test story per run"

    return input_type, feature_type, feature_tag, layer, roi_arg, fold_flags

(input_type, feature_type, feature_tag,
 layer, roi_arg, arg_list) = process_arguments(sys.argv)

if len(arg_list) > 1:
    k_fold = True
    folds = arg_list
else:
    k_fold = False

cpus = os.getenv('SLURM_CPUS_PER_TASK')
cpus = int(cpus)
print(f"using {cpus} cpus")

config = configparser.ConfigParser()
config.read('encoding-model-scaling-laws/robert_sessions_config.ini')

stories_config = configparser.ConfigParser()
stories_config.read('encoding-model-scaling-laws/stories.ini')

# TODO move to config
chunk_sz, context_sz = 100, 16100
model = 'whisper-large'

# Determine the current working directory
current_dir = Path.cwd()

# If the script is run locally (from within the encoding-model-scaling-laws directory)
if current_dir.name == "encoding-model-scaling-laws":
    base_dir = current_dir
# If the script is run from one level above (as in the cluster setup)
else:
    base_dir = current_dir / "encoding-model-scaling-laws"

# TODO: move to config
huth = False
train = True

# Huth
# test_trim_start = 50 # Trim 50 TRs off the start of the story
# test_trim_end = 5 # Trim 5 off the back


if huth:
    TR = 2  # MRI TR in seconds
    ndelays = 4  # We use 4 FIR delays (2 seconds, 4 seconds, 6 seconds, 8 seconds)
    data_pth = 'data/huth/'
    train_stories = [
        'adollshouse',
        'adventuresinsayingyes',
        'afatherscover',
        'afearstrippedbare',
        'againstthewind',
        'alternateithicatom',
        'avatar',
        'backsideofthestorm',
        'becomingindian',
        'beneaththemushroomcloud',
        'birthofanation',
        'bluehope',
        'breakingupintheageofgoogle',
        'buck',
        'canadageeseandddp',
        'catfishingstrangerstofindmyself',
        'cautioneating',
        'christmas1940',
        'cocoonoflove',
        'comingofageondeathrow',
        'escapingfromadirediagnosis',
        'exorcism',
        'eyespy',
        'findingmyownrescuer',
        'firetestforlove',
        'food',
        'forgettingfear',
        'fromboyhoodtofatherhood',
        'gangstersandcookies',
        'goingthelibertyway',
        'goldiethegoldfish',
        'golfclubbing',
        'googlingstrangersandkentuckybluegrass',
        'gpsformylostidentity',
        'hangtime',
    ]
    test_stories = ['gpsformylostidentity']
    trim_dict_start = {}
    trim_dict_end = {}
    train_trim_start = 10
    train_trim_end = 5
    for story in train_stories + test_stories:
        trim_dict_start[story] = train_trim_start
        trim_dict_end[story] = train_trim_end

    # ADAPTED TO NEW TR
    chunklen = 40
    alphas = np.logspace(0, 4, 40)  # Equally log-spaced ridge parameters between 10 and 10000.
    # print(response_dict)
    nboots = 20  # Number of cross-validation ridge regression runs. You can lower this number to increase speed.
    tag = '_huth'
else:
    TR = 1.18  # MRI TR in seconds
    hemo_max = 16  # how far out you want to model (seconds)
    ndelays = int(np.ceil(hemo_max / TR))  # ≈12 delays
    #ndelays = 12  # We use 8 FIR delays (...)
    test_trim_start = 30  # Trim 50 TRs off the start of the story
    test_trim_end = 20  # Trim 5 off the back
    train_trim_start = 20
    train_trim_end = 20

    trim_dict_start = {}
    trim_dict_end = {}

    data_pth = 'data/ours/'
    train_stories = [
        'maupassant_hand',
        'die_pflanzen_des_dr',
        'die_maske_des_roten_todes',
        'der_fall_stretelli',
        'koenig_pest',
        'der_katechismus_der_familie_musgrave',
        'lebendig_begraben',
        'fuenf_apfelsinenkerne',
        'der_blaue_karfunkel',
        'ligeia',
        # 'das_wachsfigurenkabinett',
        'das_manuskript_in_der_flasche',
        'die_schwarze_katze',
        'mord_in_sunningdale'
    ]

    test_stories = ['mord_in_sunningdale']

    for story in train_stories + test_stories:
        if input_type == 'audio':
            offset_trs = int(float(stories_config[story]['audio_start_time']) / TR)
            trim_dict_start[story] = train_trim_start + offset_trs
            trim_dict_end[story] = train_trim_end
        else:  # text mode
            trim_dict_start[story] = train_trim_start
            trim_dict_end[story] = train_trim_end

    # Bootstrap chunking parameters

    # ADAPTED TO NEW TR
    chunklen = 40
    # Define dynamic alpha parameters:
    alpha_min = -6
    alpha_max = 0
    #alphas = np.logspace(alpha_min, alpha_max, 80)  # Equally log-spaced ridge parameters between 10 and 10000.
    alphas = np.concatenate(([0.0], np.logspace(-6, 0, 49)))
   # alphas = np.logspace(-5, 1.4, 120)

    nboots = 20  # Number of cross-validation ridge regression runs. You can lower this number to increase speed.
    use_corr = True  # This is the value that will be passed to bootstrap_ridge.
    # Automatically generate tag based on the current values.
    if input_type == 'audio':
        assert feature_type is not None, "You must provide --feature_type when using audio mode."
        base_features_path = f"{data_pth}extracted_features/features_cnk{chunk_sz:0.1f}_ctx{context_sz:0.1f}/{model}"
    else:
        assert feature_type is not None, "You must provide --feature_type when using text mode."
        base_features_path = f"{data_pth}extracted_text_features"
    if layer:
        tag = (f"{input_type}_{feature_type}{feature_tag}_{layer}"
               f"_chunklen{chunklen}_alphas_{alpha_min}to{alpha_max}_{len(alphas)}"
               f"_nboots_{nboots}_use_corr_{use_corr}")
       # tag = f"{input_type}_{feature_type}_{layer}_BATCH_B_chunklen{chunklen}_alphas_{alpha_min}to{alpha_max}_{len(alphas)}_nboots_{nboots}_use_corr_{use_corr}"
    else:
        tag = (f"{input_type}_{feature_type}{feature_tag}"
               f"_chunklen{chunklen}_alphas_{alpha_min}to{alpha_max}_{len(alphas)}"
               f"_nboots_{nboots}_use_corr_{use_corr}_MEAN_HIP_R")
        #tag = f"{input_type}_{feature_type}_BATCH_B_chunklen{chunklen}_alphas_{alpha_min}to{alpha_max}_{len(alphas)}_nboots_{nboots}_use_corr_{use_corr}"

    # Create an output folder (named the same as tag) in which all files will be saved.
    output_dir = 'scratch/'+tag
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if k_fold:
    assert (len(train_stories) == len(
        folds)), f"Missmatch in the number of arguments: {len(arg_list)} and the number of stories: {len(train_stories)}"

    # Collect indices where folds[idx] == -1
    to_delete = [idx for idx, value in enumerate(folds) if value == -1]

    # Sort the indices in reverse order
    to_delete.sort(reverse=True)

    # Delete items from the end towards the beginning
    for i in to_delete:
        del train_stories[i]
        del folds[i]

    # Prepare test_stories
    test_stories = []
    if 1 in folds:
        test_index = folds.index(1)
        test_stories.append(train_stories[test_index])
        del train_stories[test_index]
    else:
        print("No test stories found with fold value 1.")

# Define base features path and tag based on mode


features_dict = {}
response_dict = {}

for story_ in train_stories + test_stories:
    if huth:
        base_response_path = f"{data_pth}/response_data/{story_}"

        with open(base_response_path + "/response.pkl", 'rb') as response_file:
            response = pickle.load(response_file)

        print(story_, response.shape)
        response_dict[story_] = response
        base_story_path = f"{data_pth}/story_data/{story_}"
        # AUdio file has 335/336 seconds
        # Our data will likely use a different sampling rate
        with open(base_story_path + "/tr_times.pkl", 'rb') as tr_file:
            tr_times = pickle.load(tr_file)
        times = np.load(base_features_path + f"/{story_}_times.npz")['times'][:, 1]  # shape: (time,)
        current_features = np.load(base_features_path + f"/{story_}.npz")['features']
        downsampled_features = lanczosinterp2D(current_features, times, tr_times)
        features_dict[story_] = downsampled_features
    else:
        base_response_path = f"{data_pth}response_data/{story_}"

        with open(base_response_path + "/response_batch_b.pkl", 'rb') as response_file:
            response = pickle.load(response_file)
        response_dict[story_] = response
        print(story_, response.shape)

        if input_type == 'audio':
            #TODO: for completion also add layer and layer check to audio features
            base_features_path = f"{data_pth}extracted_features/features_cnk{chunk_sz:0.1f}_ctx{context_sz:0.1f}/{model}"
            #TODO: check this
            num_TRs = int(stories_config[story_]['num_trs'])
            tr_times = np.arange(0, num_TRs * TR, TR)
            times = np.load(base_features_path + f"/{story_}_times.npz")['times'][:, 1]
            shifted_times = shift_extracted_features(float(stories_config[story_]['audio_start_time']), times)
            current_features = np.load(base_features_path + f"/{story_}.npz")['features']
            downsampled_features = lanczosinterp2D(current_features, shifted_times, tr_times)
            features_dict[story_] = downsampled_features


        elif input_type == 'text':

            assert feature_type is not None, "In text mode, --feature_type must be specified"
            lookback1, lookback2 = 256, 512
            base_features_path = f"{data_pth}extracted_text_features/{story_}"

            # ---------- 2.1  ordinary single-metric case -------------------
            if feature_type != 'stack_sep-drift-reinst':
                pattern = (f"final_outputs_{feature_type}"
                           f"{feature_tag}_layer{layer}_context_{lookback1}_{lookback2}.npz"
                           if layer is not None else
                           f"final_outputs_{feature_type}{feature_tag}_context_{lookback1}_{lookback2}.npz")

                path = os.path.join(base_features_path, pattern)
                current_features = np.load(path)['features']




            # ---------- 2.2  SPECIAL: stacked separation + drift + reinst ---

            else:

                # hard-coded file names (your best variants)
                fn_sep = ("final_outputs_context_separation_CLIP_NEG_COS0_RHO06_"
                          "ROLLING_WINDOW100_USE_PCA0_WIN_SKIP20_layer32_context_256_512.npz")
                fn_drift = ("final_outputs_context_drift_DETREND_STABILIZE0_PCA_DIM50_"
                            "RHO06_USE_PCA1_layer32_context_256_512.npz")
                fn_rein = ("final_outputs_reinstatement_RHO03_USE_PCA0_layer32_context_256_512.npz")
                f_sep = np.load(os.path.join(base_features_path, fn_sep))['features']
                f_drift = np.load(os.path.join(base_features_path, fn_drift))['features']
                f_rein = np.load(os.path.join(base_features_path, fn_rein))['features']

                # safety: make all (N,1)

                if f_sep.ndim == 1: f_sep = f_sep[:, None]
                if f_drift.ndim == 1: f_drift = f_drift[:, None]
                if f_rein.ndim == 1: f_rein = f_rein[:, None]

                # concat along feature axis
                current_features = np.hstack([f_sep, f_drift, f_rein])

            # ---------- sanity + storage -----------------------------------

            if current_features.ndim == 1:  # (N,) → (N,1)
                current_features = current_features[:, None]
            features_dict[story_] = current_features


# --- PARAMETER --- #

delays = range(1, ndelays + 1)
#delays = [1,2,3,4]

if not huth:
    # Initialize a list to store zero-columns masks for each story

    zero_columns_per_story = []

    # Loop over all stories
    for story in train_stories + test_stories:
        response = response_dict[story]
        # Identify zero-response voxels for the current story
        columns_with_all_zeros = np.all(response == 0, axis=0)
        zero_columns_per_story.append(columns_with_all_zeros)

    # Stack the zero-columns masks into a 2D array
    zero_columns_per_story = np.array(zero_columns_per_story)  # Shape: (n_stories, n_voxels)

    # Find voxels that are zero in any story (logical OR)
    columns_with_all_zeros_across_stories = np.any(zero_columns_per_story, axis=0)

    # Get indices of voxels that are zero in any story
    indices_of_zero_columns_across_stories = np.where(columns_with_all_zeros_across_stories)[0]
    print("Columns with all zeros in any story:", indices_of_zero_columns_across_stories,
          indices_of_zero_columns_across_stories.shape)

    with open('indices_of_zero_voxels_GM.pkl', 'wb') as f:
        pickle.dump(indices_of_zero_columns_across_stories, f)

    non_zero_voxel_mask = ~columns_with_all_zeros_across_stories  # Boolean mask

    # ------------------------------------------------------------------
    # ROI masking (union of requested masks)  --> vector in masker space
    # ------------------------------------------------------------------
    if roi_arg != 'ALL':
        from nilearn import image, masking
        import nibabel as nib

        # --- reference mask used by the original preprocessing ----------
        ref_mask_img = nib.load("data/brain_masks/binarized_mask.nii.gz")  # same as MASK_IMG
        masker_ref = ref_mask_img  # shorthand

        roi_union_3d = np.zeros(ref_mask_img.shape, dtype=bool)

        for p in roi_arg.split(','):
            p = p.strip()
            print("p=",p)
            if not os.path.isabs(p):
                p = os.path.join("data/brain_masks", p)  # default folder
            if not os.path.exists(p):
                raise FileNotFoundError(f"ROI mask not found: {p}")

            roi_img = nib.load(p)
            # resample to the functional grid
            roi_resamp = image.resample_to_img(roi_img, ref_mask_img,
                                               interpolation='nearest')
            roi_union_3d |= roi_resamp.get_fdata().astype(bool)

        # turn 3‑D union → 1‑D vector *only where ref_mask is 1*
        roi_union_img = nib.Nifti1Image(roi_union_3d.astype(np.uint8),
                                        ref_mask_img.affine)
        roi_mask_flat = masking.apply_mask(roi_union_img, ref_mask_img).astype(bool)
        # roi_mask_flat   length == number of 1‑voxels in ref_mask_img
        print("ROI voxels inside mask:", roi_mask_flat.sum())

        # combine with zero‑response filter
        non_zero_voxel_mask &= roi_mask_flat
        print("Voxels kept after ROI ∩ non‑zero filter:",
              non_zero_voxel_mask.sum())

    # Adjust responses for all stories
    for story in response_dict.keys():
        response = response_dict[story]
        # Apply the mask to exclude zero-response voxels
        response = response[:, non_zero_voxel_mask]
        # Update the response_dict with the adjusted response
        response_dict[story] = response

       # response_dict[story] = response.mean(axis=1, keepdims=True)  # → (T, 1)


# For sanity only: check every story individually
for story in train_stories + test_stories:
    f = features_dict[story][trim_dict_start[story] : -trim_dict_end[story]]
    r = response_dict[story][trim_dict_start[story] : -trim_dict_end[story]]
    assert f.shape[0] == r.shape[0], (
        f"Time‐mismatch in {story}: "
        f"{f.shape[0]} feature samples vs. {r.shape[0]} BOLD samples"
    )

# Training data
Rstim = np.nan_to_num(np.vstack(
    [ridge_utils.npp.zs(features_dict[story][int(trim_dict_start[story]):-int(trim_dict_end[story])]) for story in
     train_stories]))

# Test data
Pstim = np.nan_to_num(np.vstack(
    [ridge_utils.npp.zs(features_dict[story][int(trim_dict_start[story]):-int(trim_dict_end[story])]) for story in
     test_stories]))

# Add FIR delays
# Delays by 1,2,3,4 sampled which if sampling rate is 1/2 seconds corresponds 2,4,6,8 seconds
delRstim = make_delayed(Rstim, delays)
delPstim = make_delayed(Pstim, delays)

if not huth:
    Rresp = np.vstack(
        [response_dict[story][int(trim_dict_start[story]):-int(trim_dict_end[story])] for story in train_stories])
    Presp = np.vstack(
        [response_dict[story][int(trim_dict_start[story]):-int(trim_dict_end[story])] for story in test_stories])

    # Collapse to one “average voxel”
    # Rresp: shape (T_train, n_voxels) → (T_train, 1)
   # Rresp = np.mean(Rresp, axis=1, keepdims=True)
    # Presp: shape (T_test,  n_voxels) → (T_test,  1)
   # Presp = np.mean(Presp, axis=1, keepdims=True)
else:
    Rresp = np.vstack([response_dict[story] for story in train_stories])
    Presp = np.vstack([response_dict[story] for story in test_stories])
nchunks = int(len(Rresp) * 0.25 / chunklen)

t0 = time.time()

if train:
    # TODO: check if we can parallelize bootstrap ridge
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, Rresp, delPstim, Presp,
                                                         alphas, nboots, chunklen, nchunks,
                                                         use_corr=use_corr, single_alpha=False)

    # Save alphas in the output folder
    with open(f"{output_dir}/val_alphas.pkl", "wb") as f:
        pickle.dump(alphas, f)
    t1 = time.time()
    print("finihsed ridge in ", t1 - t0, "seconds = ", (t1 - t0) / 60, "minutes = ", (t1 - t0) / 3600, "hours",
          flush=True)
    save_weights = True

    if save_weights:
        with open(f"{output_dir}/weights_story_{test_stories[0]}.pkl", "wb") as f:
            pickle.dump(wt, f)
else:
    with open(f"{output_dir}/weights_story_{test_stories[0]}.pkl", 'rb') as f:
        wt = pickle.load(f)

print("weights shape = ", wt.shape)

nvox = Rresp.shape[1]
figc, axc = plt.subplots(figsize=(6, 4))
for current_story in train_stories + test_stories:
    if huth:
        response = response_dict[current_story]
    else:
        response = response_dict[current_story][int(trim_dict_start[current_story]):-int(trim_dict_end[current_story])]
    response = np.expand_dims(response, axis=0)
    Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(
        features_dict[current_story][int(trim_dict_start[current_story]):-int(trim_dict_end[current_story])])]))
    delPstim = make_delayed(Pstim, delays)
    pred = np.dot(delPstim, wt)
    print(response.shape,response.shape[1],response.shape[0])
    if response.shape[2] == 1:
        # single-voxel branch: squeeze out the 2nd dim
        true_ts = response.reshape(-1)
        pred_ts = pred.reshape(-1)

        print(true_ts, true_ts.shape)
        print(pred_ts, pred_ts.shape)

        # 1) Pearson r
        actual_corrs = np.array([np.corrcoef(true_ts, pred_ts)[0, 1]])

        # 2) null distribution
        chunk_size = 10
        num_chunks = len(true_ts) // chunk_size
        chunks = np.array_split(true_ts, num_chunks)
        null_corrs = Parallel(n_jobs=cpus)(
            delayed(compute_null_corrs_single)(chunks, pred_ts)
            for _ in range(500)
        )

        # 3) z‑score
        null_corrs = np.array(null_corrs)
        null_mean = null_corrs.mean()
        null_std = null_corrs.std()
        z_score = (actual_corrs - null_mean) / null_std

        # Wrap them back into the variables your plotting code expects:
        corrs_unnorm = actual_corrs  # shape (1,)
        SPE, cc_norm, cc_max = None, None, None

    else:
        # multi-voxel branch: do what you did before
        SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(response, pred, max_flooring=0.25)
        # and you already have your nulls via compute_null_corrs
    acc = corrs_unnorm
    sorted_corrs_indices = np.argsort(corrs_unnorm)  # order for worst to best
    sorted_corrs = corrs_unnorm[sorted_corrs_indices]
    if current_story in train_stories:
        plt.plot(sorted_corrs, alpha=0.12, color='gray')
    else:
        if current_story != test_stories[0]:
            print(f"AHHHHH this is terrible news: {test_stories[0]} is not the same as {current_story}")
        test_pred = pred
        plt.plot(sorted_corrs, alpha=0.3, color='red')

    # Set labels and title
plt.ylim([-0.5, 1.0])
plt.xlabel('Voxel Indices (Sorted)')
plt.ylabel('r')
axc.set_title('Unnormalized correlation (r) from Worst to Best')

# Add a legend to the plot
axc.legend()
plt.savefig(
    f"{output_dir}/training_accuracies_sorted_correlation_{model}_{chunk_sz}_{context_sz}_story_{test_stories[0]}.png")
if huth:
    response = response_dict[test_stories[0]]
else:
    response = response_dict[test_stories[0]][
               int(trim_dict_start[test_stories[0]]):-int(trim_dict_end[test_stories[0]])]

Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(
    features_dict[test_stories[0]][int(trim_dict_start[test_stories[0]]):-int(trim_dict_end[test_stories[0]])])]))
delPstim = make_delayed(Pstim, delays)
pred = np.dot(delPstim, wt)
response = np.expand_dims(response, axis=0)
if response.shape[2] == 1:
    # single-voxel branch: squeeze out the 2nd dim
    true_ts = response.reshape(-1)
    pred_ts = pred.reshape(-1)

    print(true_ts, true_ts.shape)
    print(pred_ts, pred_ts.shape)

    # 1) Pearson r
    actual_corrs = np.array([np.corrcoef(true_ts, pred_ts)[0, 1]])

    # 2) null distribution
    chunk_size = 10
    num_chunks = len(true_ts) // chunk_size
    chunks = np.array_split(true_ts, num_chunks)
    null_corrs = Parallel(n_jobs=cpus)(
        delayed(compute_null_corrs_single)(chunks, pred_ts)
        for _ in range(500)
    )

    # 3) z‑score
    null_corrs = np.array(null_corrs)
    null_mean = null_corrs.mean()
    null_std = null_corrs.std()
    z_score = (actual_corrs - null_mean) / null_std

    # Wrap them back into the variables your plotting code expects:
    corrs_unnorm = actual_corrs  # shape (1,)
    SPE, cc_norm, cc_max = None, None, None

else:
    # multi-voxel branch: do what you did before
    SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(response, pred, max_flooring=0.25)
    # and you already have your nulls via compute_null_corrs

if huth:
    true_response = response_dict[test_stories[0]]
else:
    true_response = response_dict[test_stories[0]][
                    int(trim_dict_start[test_stories[0]]):-int(trim_dict_end[test_stories[0]])]
chunk_size = 10
num_chunks = len(true_response) // chunk_size
chunks = np.array_split(true_response, num_chunks)
null_corrs = Parallel(n_jobs=cpus)(delayed(compute_null_corrs)(i, chunks, pred) for i in range(500))
null_corrs = np.array(null_corrs).flatten()

with open(f"{output_dir}/null_corrs_story_{test_stories[0]}.pkl", 'wb') as f:
    pickle.dump(null_corrs, f)

null_mean = np.mean(null_corrs)
null_std = np.std(null_corrs)

actual_corrs = corrs_unnorm
z_score = (actual_corrs - null_mean) / null_std
with open(f"{output_dir}/zscored_correlations_story_{test_stories[0]}.pkl", 'wb') as f:
    pickle.dump(z_score, f)

figh, axh = plt.subplots(figsize=(10, 6))

plt.hist(null_corrs, bins=50, alpha=0.6, color='blue', label='Null Distribution', density=True)
plt.hist(actual_corrs, bins=50, alpha=0.6, color='orange', label='Actual Correlations', density=True)

# Plot labels and legend
plt.title('Histogram of Null Distribution and Actual Correlations')
plt.xlabel('Correlation Values')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f"{output_dir}/corrs_vs_null_story_{test_stories[0]}.png")

if huth:
    response = response_dict[test_stories[0]]
else:
    response = response_dict[test_stories[0]][
               int(trim_dict_start[test_stories[0]]):-int(trim_dict_end[test_stories[0]])]

nvox = response.shape[1]

print("hey ho there are nvox ",nvox," voxels")

print("Are there nan values in the response?", np.isnan(response).any())
print(np.min(response), np.max(response))

# Calculate the absolute difference
diff = np.abs(pred.T - response.T)

print("diff has shape =",diff.shape)

# Calculate the mean difference over the 157 time points for each element
mean_diff = np.mean(diff, axis=1)

# Find the index with the minimum average difference
best_vox_idx = np.argmin(mean_diff)

print("best idx = ", best_vox_idx)

fig, ax1 = plt.subplots(figsize=(6, 4))
plt.plot(response.T[best_vox_idx], alpha=0.12, color='black')
plt.plot(pred.T[best_vox_idx], color='red', alpha=0.8)
plt.xlabel("TR", size=16)
plt.ylabel("Z-scored Response", size=16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

plt.savefig(
    f"{output_dir}/BEST_single_voxel_single_adio_single_session_test_{model}_{chunk_sz}_{context_sz}_story_{test_stories[0]}.png")

# Calculate the absolute difference (L1 norm)
diff_l1 = np.abs(pred.T - response.T)
mean_diff_l1 = np.mean(diff_l1, axis=1)

# Calculate the squared difference (L2 norm)
diff_l2 = np.square(pred.T - response.T)
mean_diff_l2 = np.sqrt(np.mean(diff_l2, axis=1))

# Calculate the relative root mean squared error (voxel-wise)
rrmse = np.sqrt(np.mean((pred.T - response.T) ** 2, axis=1)) / np.sqrt(np.mean(response.T ** 2, axis=1))

# Sort the indices from worst to best based on L1 norm
sorted_indices_l1 = np.argsort(mean_diff_l1)[::-1]  # reverse order for worst to best
sorted_diff_l1 = mean_diff_l1[sorted_indices_l1]

# Sort the indices from worst to best based on L2 norm
sorted_indices_l2 = np.argsort(mean_diff_l2)[::-1]  # reverse order for worst to best
sorted_diff_l2 = mean_diff_l2[sorted_indices_l2]

# Sort indices for RRMSE
sorted_indices_rrmse = np.argsort(rrmse)[::-1]  # reverse order for worst to best
sorted_rrmse = rrmse[sorted_indices_rrmse] * 100  # Convert to percentage

# Plotting the L1, L2 norm differences and RRMSE
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12))

# Plot for L1 norm (Absolute Difference)
ax2[0].plot(np.arange(nvox), sorted_diff_l1, label='L1 Norm (Absolute Difference)')
ax2[0].set_xlabel('Voxel Indices (Sorted)')
ax2[0].set_ylabel('L1 Norm')
ax2[0].set_title('L1 Norm (Absolute Difference) from Worst to Best')
ax2[0].legend()

# Plot for L2 norm (Euclidean Distance)
ax2[1].plot(np.arange(nvox), sorted_diff_l2, label='L2 Norm (Euclidean Distance)', color='orange')
ax2[1].set_xlabel('Voxel Indices (Sorted)')
ax2[1].set_ylabel('L2 Norm')
ax2[1].set_title('L2 Norm (Euclidean Distance) from Worst to Best')
ax2[1].legend()

# Plot for RRMSE (in percentage)
ax2[2].plot(np.arange(nvox), sorted_rrmse, label='RRMSE (%)', color='green')
ax2[2].set_xlabel('Voxel Indices (Sorted)')
ax2[2].set_ylabel('RRMSE (%)')
ax2[2].set_title('Relative Root Mean Squared Error (RRMSE) from Worst to Best')
ax2[2].legend()

plt.tight_layout()

# Save the plot to a file
plt.savefig(f"{output_dir}/sorted_l1_l2_rrmse_norms_{model}_{chunk_sz}_{context_sz}_story_{test_stories[0]}.png")

if k_fold:
    with open(f"{output_dir}/corr_story_{test_stories[0]}.pkl", "wb") as f:
        pickle.dump(corrs_unnorm, f)

print("Are there nan valuesin the correlation ?", np.isnan(corrs_unnorm).any())
print("min = ", np.min(corrs_unnorm), "max = ", np.max(corrs_unnorm))

best_vox_idx_by_corr = np.argmax(corrs_unnorm)
best_corr_value = corrs_unnorm[best_vox_idx_by_corr]

# Plot best voxel by sorting according to correlation
figc2, axc2 = plt.subplots(figsize=(6, 4))
plt.plot(response.T[best_vox_idx_by_corr], alpha=0.12, color='black')
plt.plot(pred.T[best_vox_idx_by_corr], color='red', alpha=0.8)
plt.xlabel("TR", size=16)
plt.ylabel("Z-scored Response", size=16)
axc2.spines['right'].set_visible(False)
axc2.spines['top'].set_visible(False)
axc2.spines['left'].set_visible(False)
axc2.spines['bottom'].set_visible(False)

# Add the correlation value inside the plot
corr_text = f"Correlation: {best_corr_value:.3f}"
axc2.text(0.05, 0.95, corr_text, transform=axc2.transAxes, fontsize=14,
          verticalalignment='top')
axc2.text(0.55, 0.95, 'idx = ' + str(best_vox_idx_by_corr), transform=axc2.transAxes, fontsize=14,
          verticalalignment='top')

plt.savefig(f"{output_dir}/BEST_voxel_by_correlation_{model}_{chunk_sz}_{context_sz}_story_{test_stories[0]}.png")

with open(f"{output_dir}/full_prediction_{test_stories[0]}.pkl", 'wb') as f:
    pickle.dump(pred.T, f)

with open(f"{output_dir}/best_voxel_by_correlation_prediction_{test_stories[0]}.pkl", 'wb') as f:
    pickle.dump(pred.T[best_vox_idx_by_corr], f)

with open(f"{output_dir}/best_voxel_by_correlation_response_{test_stories[0]}.pkl", 'wb') as f:
    pickle.dump(response.T[best_vox_idx_by_corr], f)

with open(f"{output_dir}/full_response_{test_stories[0]}.pkl", 'wb') as f:
    pickle.dump(response.T, f)

