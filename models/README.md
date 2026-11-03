# Trained encoders

| File | What | Trained in |
|---|---|---|
| `modelA_speaker_encoder.pt` | Encoder A: the encoder the protection is optimised against (0.53 M weights, 192-number embedding) | `code/04_train_A.py` |
| `modelB_independent_judge.pt` | Encoder B: the independent judge, used only for testing (1.65 M weights, seed 1234) | `code/06_model_B.py` |

- Both files are PyTorch state dicts.
- Load them with the class definitions in `code/09_main_run.py` (`SpeakerNet` for A, `SpeakerNetB` for B).
- To run the code, put both files in `MyDrive/PHC351_24324008_VoiceProtection/code/` in Google Drive.
