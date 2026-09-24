# Local benchmark data

Benchmark images and annotations are intentionally not version-controlled. The VOST experiment fetches the two selected validation sequences from the official archive into `data/vost_subset/val/`:

```bash
python scripts/vost_subset.py --videos 555_tear_aluminium_foil 6922_split_paper
python scripts/evaluate_vost_base_roundtrip.py --check-data-only
```

The downloader writes a per-file SHA-256 manifest beside the files. Follow the [official VOST data terms](https://www.vostdataset.org/data.html); do not redistribute the downloaded media through this repository. Other dataset folders in a local working copy are also excluded by `.gitignore`.
