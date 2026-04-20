# BrainEncode

Code accompanying the paper ["Coherence in the brain unfolds across separable temporal regimes"](https://arxiv.org/abs/2512.20481).

The project studies two complementary aspects of coherence during naturalistic language comprehension: slow contextual drift and rapid event-driven shifts. The repository contains the feature extraction code, voxelwise ridge-encoding code, event-boundary analyses, and post-processing scripts used to generate the paper results.

## What Is In This Repository

- `Code/feature_creation/`: extract text and speech features, derive event-boundary regressors, and build additional TEM-style features.
- `Code/ridge_universal.py`: run voxelwise ridge regression for text or audio features.
- `Code/post_processing/`: statistical comparisons, ROI summaries, and paper plots.
- `Code/LLM_eventboundaries_tests/`: standalone analyses for LLM-based event-boundary prompting and paper figures.
- `Code/ridge_utils/`: helper utilities reused across the pipelines.
- `Code/uroman/`: vendored text-normalization utilities used by the text-feature pipeline.

The large fMRI datasets, local stimulus folders, and most generated result artifacts are intentionally not bundled in GitHub. Several scripts expect you to provide those local data directories yourself.

## Download

Clone with HTTPS:

```bash
git clone https://github.com/davidestaub/BrainEncode.git
cd BrainEncode
```

Or with SSH:

```bash
git clone git@github.com:davidestaub/BrainEncode.git
cd BrainEncode
```

## Environment Setup

Python 3.10+ is recommended.

1. Create and activate a virtual environment.
2. Install PyTorch and Torchaudio for your platform first.
3. Install the remaining Python packages.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio
pip install numpy scipy pandas matplotlib scikit-learn statsmodels nibabel nilearn joblib tqdm regex tables transformers
```

Notes:

- GPU access is effectively required for the larger Hugging Face models used in the paper.
- `Code/uroman` is included in the repository, so there is no separate install step for that package.
- Most scripts are research scripts rather than packaged CLI tools, so some local path adaptation is still expected.

## Expected Local Data Layout

The code assumes a local workspace with directories such as:

```text
data/ours/
  processed_stimuli/
  extracted_text_features/
  extracted_features/
  response_data/

../../../../DATA/BrainEncode/
  MASKS/
  anatomical_files/
  text_metrics_results/
```

In practice this means:

- feature extraction writes into `data/ours/...`
- encoding models read extracted features plus fMRI responses from `data/ours/...`
- post-processing scripts read second-level outputs and anatomical masks from `DATA/BrainEncode/...`

Those data folders are not included here.

## Config Files

Two bundled config files are relevant for first-pass setup:

- `Code/feature_creation/text_features_arguments.ini`
- `Code/feature_creation/stories.ini`

You will likely need to edit `text_features_arguments.ini` to match your local stimulus and output folders.

## Typical Workflow

1. Extract text features in `Code/feature_creation/extract_text_features.py`.
2. Optionally derive additional drift, reinstatement, separation, or event-boundary features from those saved outputs.
3. Run `Code/ridge_universal.py` to fit voxelwise encoding models.
4. Use the scripts in `Code/post_processing/` to compute ROI summaries, significance tests, and paper figures.
5. Use `Code/LLM_eventboundaries_tests/` for the standalone event-boundary validation analyses.

## Example Commands

Text feature extraction:

```bash
python Code/feature_creation/extract_text_features.py \
  --stories maupassant_hand \
  --config Code/feature_creation/text_features_arguments.ini \
  --stories_config Code/feature_creation/stories.ini \
  --feature_types hidden_states,surprisal \
  --layers 32,79
```

Speech feature extraction:

```bash
python Code/feature_creation/extract_speech_feature_simplified.py \
  --input path/to/story.wav
```

Voxelwise encoding:

```bash
python Code/ridge_universal.py text context_drift _RHO03_USE_PCA0 32 ALL 0 0 0 0 0 0 0 0 0 0 0 1
```

The ridge script is still close to the research code used in the project, so you may need to adapt fold flags, feature tags, and local paths for your exact setup.

## Practical Caveats

- Some scripts still reflect the original research environment and contain hard-coded path assumptions.
- `Code/post_processing/` is primarily for the final paper analyses, not for a clean end-user API.
- Generated result folders are excluded from the lightweight GitHub export to keep the repository size manageable.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{staub2025coherence,
  title={Coherence in the brain unfolds across separable temporal regimes},
  author={Staub, Davide and Rabe, Finn and Misra, Akhil and Pauli, Yves and H\"uppi, Roya and Yang, Ni and Lang, Nils and Michels, Lars and Edkins, Victoria and Fr\"uhholz, Sascha and Sommer, Iris and Hinzen, Wolfram and Homan, Philipp},
  journal={arXiv preprint arXiv:2512.20481},
  year={2025}
}
```
