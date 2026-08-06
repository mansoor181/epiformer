# Checkpoints

EpiFormer as reported in the paper uses geometric features only (`model.epiformer.use_plm=false`).
PLM embeddings were removed by ablation and performance improved, so the released model has no PLM input.

## Released checkpoints (geometric-only EpiFormer)

| Directory | Training run | Seed | Split | Decoder layers |
|---|---|---|---|---|
| `epiformer-epitope-ratio/` | `m1_noplm_epi_ratio_seed456_20260329-095214` | 456 | epitope_ratio | 2 |
| `epiformer-epitope-group/` | `m1_noplm_epi_group_seed123_20260329-025003` | 123 | epitope_group | 3 |

Reproduce with:

```bash
cd code
python evaluate.py \
    --checkpoint ../checkpoints/epiformer-epitope-ratio/epiformer_best.pt \
    --data_dir /path/to/data --gpu_id 0
```

Held-out test metrics at threshold 0.3, evaluated on the split the checkpoint was trained for:

| Checkpoint | AUROC | AUPRC | F1 | MCC |
|---|---|---|---|---|
| `epiformer-epitope-ratio` (epitope_ratio test) | 0.9265 | 0.5094 | 0.4963 | 0.4748 |
| `epiformer-epitope-group` (epitope_group test) | 0.8685 | 0.3416 | 0.3473 | 0.3436 |

Each checkpoint is valid only for its own split. Evaluating a checkpoint on the other split leaks
training complexes into the test set and the numbers are not meaningful.

## Relation to the paper tables

Table 1 reports the mean over three seeds (42, 123, 456), produced by
`code/scripts/run_m1_noplm_multiseed.sh`. Only one checkpoint per split was retained.

| Split | Seed 42 | Seed 123 | Seed 456 | Mean (paper) |
|---|---|---|---|---|
| epitope_ratio AUROC | 0.9200 | 0.9240 | 0.9265 | 0.924 |
| epitope_ratio F1 | 0.4714 | 0.4787 | 0.4963 | 0.482 |
| epitope_ratio MCC | 0.4539 | 0.4625 | 0.4747 | 0.464 |
| epitope_group AUROC | 0.8822 | 0.8685 | 0.7273 | 0.826 |
| epitope_group F1 | 0.3558 | 0.3472 | 0.2119 | 0.305 |

Per-seed logs are in `rebuttals/m1_plm/logs/m1_noplm_epi_*_seed*_output.log`.

## Legacy checkpoints (PLM-era, superseded)

`best-playful-sweep-72/` (epitope_ratio) and `best-glamorous-sweep-37/` (epitope_group) are the
earlier EpiFormer variants that consumed ESM-2 and AntiBERTy embeddings. They are kept for the PLM
sensitivity ablation and are not the models reported in the main results.
