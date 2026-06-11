# DAVA AND MLPR

## Included

- model definition
- DAVA alignment module
- MIL head
- MLPR reweighting
- EMA teacher-student training
- training and evaluation entry scripts

## Excluded

- dataset files
- data split generation scripts
- benchmark runners
- plotting and reviewer-only utilities

## Main Files

- `scripts/strong_trainer.py`
- `scripts/eval_enhanced.py`
- `configs/config.yaml`

## Training

```bash
python scripts/strong_trainer.py  --output outputs/run_ours
```

## Evaluation

```bash
python scripts/eval_enhanced.py  --checkpoint outputs/checkpoints/best_f1.pth --output outputs/eval_ours
```

## Notes

- Prepare the CSV files referenced by the config before running.
- “The Intel Robotic Welding Multimodal Dataset used in this study is publicly available from the original source for research purposes.As the authors are not
authorized to redistribute or mirror the raw data,interested readers should obtain
the dataset directly from the original provider.
- This package is intended for the proposed method path only.
- The gradient update in the original network relies on GPU computing power during training. If you encounter an “Out Of Memory Error: CUDA out of memory” or similar error during runtime, please stop the process and adjust the settings based on your computer's computing power.
