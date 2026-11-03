# Hiding a Voice From Speaker Encoders

Code, results and analysis for the paper

> **Hiding a Voice From Speaker Encoders: Adversarial Protection Trained Over Measured Room Channels, Evaluated Through Channel Models, Human Listeners and a Real Air Gap**
> Bhavishay and Mayank Goswami, Department of Physics, IIT Roorkee (PHC-351 project, 2026)

A small perturbation is added to speech so that a speaker encoder no longer recognises the talker. The perturbation is optimised with projected gradient descent and Expectation over Transformation over impulse responses measured in four real rooms (laptop loudspeaker to phone microphone). It is tested on a second encoder that was trained independently and never used in the optimisation, through rooms left out of training, by human listeners, and in one physical air-gap test.

- All three encoders were trained from scratch on LibriSpeech. No pretrained models were used.
- No code was taken from other repositories.
- Everything ran in Google Colab on a T4 GPU.

## Quick check: recompute the paper's numbers (CPU, about 1–2 minutes)

`analysis/paper_numbers.py` computes every number in the paper that comes from the optimisation trials:

- Tables III, IV and VI;
- Sections VII-B to VII-E and VII-G;
- corrections C9 and C10, including the flat-channel check of C10.

It reads the saved trial results in `results/`. From the top folder of this repository, run:

```bash
pip install numpy pandas scipy torch
python analysis/paper_numbers.py results/overnight_v2/results_rows.csv results/device_control/results_rows.csv out.json
```

The printed output is identical to `analysis/paper_numbers_output.txt`. In Colab, first clone this repository with `!git clone <repository URL>` and `%cd hiding-a-voice`, then run the same line with `!python`.

## What is where

| Folder | Contents | Paper |
|---|---|---|
| `code/` | The main pipeline in 16 files, numbered in the order they were run | Sections IV–VII |
| `notebooks/` | The same code as Colab notebooks, one per Colab session | |
| `experiments_v2/` | The two later runs, as Colab notebooks plus the same code as `.py`. **Overnight run v2:** matched comparison of channel models, processing ablations, five repeated runs, and the audio for listening test 2. **Device-control run:** the control for the colouring shared by all measured responses | VII-B to VII-E, Tables III and IV |
| `analysis/` | `paper_numbers.py` and its output | Tables III, IV, VI, C9, C10 |
| `results/` | Trial-level results of both later runs (one row per trial), plus their run logs and setup files | |
| `listening_test_2/` | The analysis script (plan fixed before data collection), the clip manifest, the test page, and the aggregate results | VII-K, Table X |
| `models/` | Trained encoders A and B | |
| `data/` | Measured room impulse responses and decision thresholds | |

## The main pipeline (`code/`)

| File | Stage | Result |
|---|---|---|
| `01_features.py` | First log-mel test | (80, frames) picture |
| `02_dataset.py` | Front end, speaker split, data loader | 251 speakers → 200 / 51 |
| `03_model_A.py` | Encoder A | 0.53 M weights, 192-number embedding |
| `04_train_A.py` | Training encoder A | Model saved to Drive |
| `05_eer_A.py` | Clean equal error rate of A | EER 0.92 %, τ 0.463 |
| `06_model_B.py` | Encoder B, the independent judge | 1.65 M weights |
| `07_rooms.py` | Sine sweep, deconvolution, RT60 | Four rooms, RT60 0.42–1.81 s |
| `08_thresholds.py` | Thresholds under the test conditions (correction C2) | τ_A 0.734 (EER 18.33 %), τ_B 0.793 (EER 16.25 %) |
| `09_main_run.py` | Protection through measured rooms, 20 speakers × 4 held-out rooms | Attacker 98 % → 28 %, SNR 23.9 dB, PESQ 2.07 |
| `10_level_norm_and_ttest.py` | Paired t-test; level-normalised clip (correction C7) | t(79) = 18.92 |
| `11_eps_curve.py` | ε against similarity and quality | Trade-off (Table V) |
| `12_airgap.py` | Clips played aloud and re-recorded in 4 rooms | Attacker succeeds in 4 of 4 rooms → 3 of 4 |
| `13_full_chain.py` | Training through the whole recording chain | +0.020 on average (Table VIII) |
| `14_model_C_and_ensemble.py` | Encoder C and the A + C ensemble | No consistent gain |
| `15_transfer_tricks.py` | Masking, targeted objective, momentum, input diversity | None helped at equal SNR |
| `16_sim_vs_measured.py` | First, unmatched comparison of simulated and measured rooms | Replaced by the matched comparison in `experiments_v2/` (correction C8) |

Files 14 and 15 define encoder C, the ensemble protection and the transfer tricks. The cells that evaluated them were run in Colab and are not included.

## Running the code

1. **Colab:** open a notebook in Google Colab, choose Runtime → Change runtime type → T4 GPU, and mount Google Drive when asked.
2. **Drive folder.** The code reads and writes `MyDrive/PHC351_24324008_VoiceProtection/`. To run it with the files in this repository, put them in that folder:

   | File in this repository | Place in the Drive folder |
   |---|---|
   | `models/modelA_speaker_encoder.pt`, `models/modelB_independent_judge.pt` | `code/` |
   | `data/measured_rirs.pt` | `testing_data/recordings/` |
   | `data/thresholds_measured.pt` | `testing_data/results/` |
   | Held-out clips: the two LibriSpeech utterances per speaker listed in `results/overnight_v2/setup_info.json` | `training_data/heldout_subset/`, named `<speaker>_<utterance>.flac` |

3. **Session order.**
   - `06`, `09`, `13` and `16` start by mounting Drive and loading what they need, so each can begin a fresh session.
   - The other files use names defined by the files before them. Run the files of one notebook in order, in one session.
4. **Data.** LibriSpeech `train-clean-100` (6.3 GB) and the OpenSLR-28 room impulse responses are downloaded from openslr.org by `06`, `16` and the two later runs. Files 01–05 expect LibriSpeech already unpacked at `/content/LibriSpeech/train-clean-100`.
5. **Long runs.** The two runs in `experiments_v2/` take several hours. They save every finished unit to Drive, so a disconnected run resumes where it stopped. Both runs were resumed this way; see `results/*/run_log.txt`.

## Randomness

- **Fixed seeds:**
  - the speaker split (seed 0);
  - the trial selection;
  - the training of encoders B (seed 1234) and C (seed 777);
  - each pass of the two later runs.
- **Not seeded:**
  - the training of encoder A (`04`);
  - the random choice of rooms inside the optimisation of `09`.

So a re-run of those two steps reproduces the reported figures only up to that randomness. The reported numbers come from the saved models in `models/`.

The later runs recorded Python 3.13 and torch 2.11 (see `results/*/setup_info.json`). Versions for `code/` were not recorded.

## Listening tests

- **Test 2 (controlled, 28 listeners).**
  - The analysis plan and the exclusion rules are written in the header of `listening_test_2/analyse_listening_v2.py` and were fixed before data collection.
  - `listening_test_2/results/` holds the aggregate report.
- **Test 1 (uncontrolled, 22 listeners).** The results are in the paper (Table IX).

Individual answers from either test are not published, because listeners were told that only the researchers would read them.

## Not included

- LibriSpeech and the OpenSLR-28 room impulse responses. The code downloads them from openslr.org.
- The recorded audio (the sweeps and the re-recorded clips).
- Individual listening-test answers.

## Citation

Bhavishay and M. Goswami, "Hiding a Voice From Speaker Encoders: Adversarial Protection Trained Over Measured Room Channels, Evaluated Through Channel Models, Human Listeners and a Real Air Gap," Department of Physics, IIT Roorkee, 2026.

## License

MIT, see `LICENSE`.
