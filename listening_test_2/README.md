# Listening test 2 (controlled)

- 28 listeners took part; none was excluded.
- Each listener heard 4 of the 12 speakers:
  - 14 same-or-different pairs;
  - 4 sentences to type;
  - 9 quality ratings;
  - a headphone check.
- The analysis plan and the exclusion rules are written in the header of `analyse_listening_v2.py`. They were fixed before any answers came in.

| File | What |
|---|---|
| `analyse_listening_v2.py` | The analysis: primary outcomes, Holm correction, equivalence tests, sensitivity analyses |
| `manifest.json` | Which clip is which: speaker, condition, sentence |
| `listening_test_2_page.html` | The test page as given to listeners. The researcher key that unlocks the answers is removed from this copy, and the audio is not included. |
| `results/` | Aggregate results: `REPORT.md`, `summary.json`, per-clip means (`per_item.csv`) and the figure |

- **Audio.** The clips were made by the overnight run in `../experiments_v2/` (its section E).
- **Individual answers** are not published, because listeners were told that only the researchers would read them.
- **Running the analysis** on a file of answers:

```bash
python analyse_listening_v2.py --manifest manifest.json --responses responses.json --out results
```
