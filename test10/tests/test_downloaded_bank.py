"""CPU contracts for full-bank evaluation, without training or GPU calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "tools/evaluate_downloaded_bank.py"
spec = importlib.util.spec_from_file_location("evaluate_downloaded_bank",PATH)
bank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bank)


class BankTests(unittest.TestCase):
    def test_actual_checkpoint_membership_not_catalog_split(self):
        report={"collection":{"pairs":[dict(dataset="MOSEv2",video_id="a",split="train"),
                                          dict(dataset="MOSEv2",video_id="b",split="validation")]}}
        train,val=bank.membership(report)
        self.assertEqual(bank.role("MOSEv2","a",train,val),"training_video")
        self.assertEqual(bank.role("MOSEv2","b",train,val),"checkpoint_selection_video")
        self.assertEqual(bank.role("MOSEv2","c",train,val),"excluded_from_training_and_selection")

    def test_membership_overlap_is_an_error(self):
        with self.assertRaises(ValueError):
            bank.membership({"collection":{"pairs":[dict(dataset="d",video_id="v",split=s)
                                                       for s in ("train","validation")]}})

    def test_interleaving_keeps_all_cases_and_ignores_scores(self):
        rows=[dict(dataset=d,video_id=str(i),pair_id=f"{d}:{i}",score=-i)
              for d,n in (("MOSEv2",3),("LVOSv2",2)) for i in range(n)]
        ordered=bank.interleave(list(reversed(rows)))
        self.assertEqual([r["dataset"] for r in ordered],["MOSEv2","LVOSv2"]*2+["MOSEv2"])
        self.assertEqual({r["pair_id"] for r in ordered},{r["pair_id"] for r in rows})

    def test_shared_features_do_not_share_tracking_memory(self):
        class Predictor:
            def __init__(self):self.encodes=[]
            def _get_image_feature(self,state,frame_idx,batch_size):
                if frame_idx not in state["cached_features"]:
                    self.encodes.append(frame_idx)
                    state["cached_features"]={frame_idx:(frame_idx,frame_idx*10)}
                return state["cached_features"][frame_idx]
        predictor=Predictor()
        original=predictor._get_image_feature
        a={"cached_features":{},"memory":"method_a"}
        b={"cached_features":{},"memory":"method_b"}
        with bank.FutureFeatureCache(predictor,[11,12]) as cache:
            for state in (a,b):
                for pos in (11,12):
                    self.assertEqual(predictor._get_image_feature(state,pos,1),(pos,pos*10))
            self.assertEqual(predictor.encodes,[11,12])
            self.assertEqual(cache.calls,[11,12,11,12])
            self.assertEqual(a["memory"],"method_a")
            self.assertEqual(b["memory"],"method_b")
            with self.assertRaises(RuntimeError):predictor._get_image_feature(a,10,1)
        self.assertEqual(predictor._get_image_feature,original)

    def test_resume_is_method_specific_and_checks_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"score.json"
            c=dict(pair_id="p",pair_sha256="sha",input_hashes={"rgb":"hash"})
            self.assertFalse(bank.valid_saved(p,c,"affine","run"))
            row=dict(fingerprint="run",pair_sha256="sha",method="affine",pair_id="p",horizon=5,
                     input_fingerprint=bank.digest(c["input_hashes"]),scores={"average":{"frames":5}})
            bank.write(p,row)
            self.assertTrue(bank.valid_saved(p,c,"affine","run"))
            with self.assertRaises(ValueError):bank.valid_saved(p,c,"transformer","run")
            with self.assertRaises(ValueError):bank.valid_saved(p,c,"affine","different_run")

    def test_summary_equal_video_weight_not_equal_pair_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            prov={"checkpoints":{},"files":{}}
            cases=[]
            # Video a has two perfect cases, video b one failed case.
            for vid,index,val in (("a",0,1.0),("a",1,1.0),("b",0,0.0)):
                c=dict(pair_id=f"{vid}:{index}",dataset="MOSEv2",pair_sha256="sha",input_hashes={})
                cases.append(c)
                for m in bank.METHODS:
                    bank.write(bank.row_path(root,c,m),dict(fingerprint=bank.digest(prov),pair_id=c["pair_id"],
                        dataset="MOSEv2",video_id=vid,pair_sha256="sha",method=m,horizon=5,
                        source_pool="development",evaluation_role="excluded_from_training_and_selection",
                        input_fingerprint=bank.digest({}),scores={"average":{"frames":5,"J":val,"F":val,"J_and_F":val}}))
            bank.write(root/"provenance.json",prov)
            bank.write(root/"selection.json",dict(cases=cases,warning="synthetic"))
            result=bank.summarize(root)
            self.assertEqual(result["complete_pairs"],3)
            tab=result["groups"]["all"]["MOSEv2"]
            self.assertEqual(tab["videos"],2)
            self.assertEqual(tab["methods"]["affine"]["J_and_F"]["mean"],50)
            self.assertEqual(tab["paired"]["transformer-minus-affine"]["mean"],0)


if __name__=="__main__":unittest.main()
