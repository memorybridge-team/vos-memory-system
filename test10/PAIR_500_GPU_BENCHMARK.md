# 500 downloaded state-pair GPU timing

Measured on 2026-10-01 with an RTX 4080 SUPER (32 GiB), PyTorch 2.8.0+cu128. Seed 7 selected one `fit` pair per video: 250 MOSEv2 and 250 LVOSv2, 500 distinct videos total. The same 500 pairs were timed once for each method. No checkpoint was saved, and this was not a quality evaluation.

Each timed pair performed one Adam training step with `batch_records=4`. CUDA was synchronized before and after. Wall time includes local `.pt` SHA256 verification, `torch.load`, CPU→GPU transfer, forward/backward and optimizer step. One warm-up pair per method was excluded. The pairs totaled 7.85 GB on disk. The raw 1,500 pair-method measurements are saved locally at `/home/home/test/test10_benchmark_500.jsonl`; that file contains machine-specific absolute paths and is not committed.

| Method | MOSEv2 mean (250) | LVOSv2 mean (250) | Combined mean / median / p90 (500) | 500-pair total |
| --- | ---: | ---: | ---: | ---: |
| Affine | 0.227 s | 0.214 s | 0.221 / 0.219 / 0.267 s | 110.3 s |
| Residual MLP | 0.199 s | 0.196 s | 0.198 / 0.194 / 0.242 s | 98.8 s |
| Transformer | 0.206 s | 0.254 s | 0.230 / 0.234 / 0.277 s | 115.1 s |

The three timed sets took 324.3 seconds in total, excluding model setup, warm-up and reporting. Mean time to first batch (checksum/load/first transfer included) was 0.192/0.160/0.166 s for Affine/MLP/Transformer; the rest of the step averaged 0.029/0.038/0.064 s. This is an end-to-end pair throughput measurement, not isolated GPU kernel time. Sequential method order and OS file cache can affect cross-method differences, so the mean times should not be read as a precise model-only speed ranking. The sample averaged 14.934 valid memory records per pair; extrapolation to the full corpus requires accounting for its different record-count distribution and repeated epochs.

The benchmark code is [`benchmark_pairs.py`](benchmark_pairs.py). The local selection JSON was built from the downloaded pair root using `pair_catalog.py`; its SHA256 was `3bbf4ebbb6fb6d83975b812ccb6700cc7aa664cca1a9035cfa854a0cda53e69c`.
