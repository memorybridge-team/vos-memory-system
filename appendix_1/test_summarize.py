import csv
import json
import tempfile
import unittest
from pathlib import Path

from summarize import bootstrap_interval, import_server_results, summarize_run, paired_official_interval


def metric(value):
    return {"j": value, "f": value, "j_and_f": value,
            "objects": [{"object_id": 1, "prompt_frame": 0, "j": value,
                         "f": value, "j_and_f": value, "frames": [{"frame_id": 1}]}]}


class SummaryTests(unittest.TestCase):
    def test_bootstrap_uses_object_weighting(self):
        results = []
        for n, delta in ((1, .1), (9, -.1)):
            cold, full_memory = metric(.5), metric(.5 + delta)
            for data in (cold, full_memory):
                data["objects"] = [{**data["objects"][0], "object_id": i} for i in range(n)]
            results.append({"metrics": {"cold": cold, "full_memory": full_memory}})
        # Seed 0's first draw is [1,1]; inspect a mixed draw seed explicitly.
        import numpy as np
        seed = next(s for s in range(100) if len(set(np.random.default_rng(s).integers(0, 2, size=2))) == 2)
        low, high = paired_official_interval(results, "davis2017", samples=1, seed=seed)
        self.assertAlmostEqual(low, -.08)
        self.assertAlmostEqual(high, -.08)

    def test_paired_aggregation_and_server_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps({
                "output": "results", "cache": "cache", "sam2_repo": "sam2",
                "models": ["tiny"], "datasets": [{"name": "davis2017", "split": "val"}],
                "device": "cpu", "precision": "float32", "seed": 5,
                "bootstrap_samples": 200,
            }))
            run = root / "results/run1"
            (run / "davis2017_val/tiny/videos/a").mkdir(parents=True)
            (run / "davis2017_val/tiny/videos/b").mkdir(parents=True)
            (run / "run.json").write_text("{}")
            (run / "davis2017_val/tiny/selection.json").write_text(
                json.dumps({"videos": ["a", "b"]}))
            for name, cold, full_memory in (("a", .4, .5), ("b", .7, .6)):
                result = {
                    "identity": {"dataset": "davis2017", "split": "val", "model": "tiny"},
                    "video": name, "partial_run": False,
                    "metrics": {"cold": metric(cold), "full_memory": metric(full_memory)},
                    "full_memory_minus_cold": {"j": full_memory-cold, "f": full_memory-cold,
                                                "j_and_f": full_memory-cold},
                    "timing_seconds": {"cold": 1, "full_memory": 1,
                                       "full_memory_including_first_pass": 2},
                    "full_memory_audit": [{"unique_memory_hits": {"first_pass": 2,
                        "second_pass": 0}, "events": [{"is_future": True}],
                        "second_pass_writes": 3,
                        "overwritten_injected_records": 2,
                        "removed_injected_records": 0,
                        "remaining_injected_records": 1}],
                    "condition_deltas": {"full_memory": {"j": full_memory-cold,
                        "f": full_memory-cold, "j_and_f": full_memory-cold}},
                }
                (run / f"davis2017_val/tiny/videos/{name}/result.json").write_text(json.dumps(result))
            summary = summarize_run(config, "run1")
            group = summary["groups"][0]
            self.assertAlmostEqual(group["delta_j_and_f"], 0.0)
            self.assertEqual((group["improved_videos"], group["decreased_videos"]), (1, 1))
            with (run / "videos.csv").open(newline="") as stream:
                video_rows = list(csv.DictReader(stream))
            self.assertTrue(all(int(row["full_memory_second_pass_writes"]) == 3
                                for row in video_rows))
            self.assertTrue(all(int(row["overwritten_injected_records"]) == 2
                                for row in video_rows))
            self.assertTrue(all(int(row["remaining_injected_records"]) == 1
                                for row in video_rows))
            server = root / "server.json"
            server.write_text(json.dumps({"j_and_f_dot": 0.55}))
            record = import_server_results(config, server, "submission-1", "tiny", "cold", "run1")
            self.assertEqual(record["metrics"]["j_and_f_dot"], 0.55)

    def test_bootstrap_single_value(self):
        self.assertEqual(bootstrap_interval([0.2]), [0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
