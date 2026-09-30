import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (Artifacts, METHODS, GATE, digest, key, positions, replay_frames,
                  select_smoke, freeze_schedule, snapshot, verify_snapshot, write)
from manifest import imports, make_case, normalize
from report import per_video, paired, build_report


def case(cid="c", dataset="MOSEv2", video="v", **kw):
    return dict(case_id=cid, dataset=dataset, video_id=video, object_id=1,
                first=0, switch=16, end=20, frame_stems=[str(i) for i in range(21)],
                input_sha256="rgb", annotation_sha256="annotation", sampling={"raw_stride":1},
                cohort="additional", checkpoint_video=False, length_bin=0, **kw)


class Contracts(unittest.TestCase):
    def test_switch_uses_positions_not_stem_arithmetic(self):
        stems = [str(1+5*i) for i in range(30)]
        self.assertEqual(positions(stems,1,16,146), (0,16,29))

    def test_no_suffix_rejected(self):
        with self.assertRaises(ValueError): positions(list(map(str,range(17))),0,5,16)

    def test_replay_anchor_disjoint_and_exact(self):
        for k in (4,8,16):
            fs=replay_frames(0,16,k)
            self.assertEqual(len(fs), k+1); self.assertEqual(fs[-1],16)
            self.assertEqual(len(set(fs)),k+1)
        with self.assertRaises(ValueError): replay_frames(3,16,16)

    def test_cache_ignores_paths_cohort_manifest_order(self):
        c=case(); other=dict(c,video_dir="/different",cohort="legacy_dev",case_id="newlabel")
        self.assertEqual(key(c,"affine",{}),key(other,"affine",{}))
        for field in ("switch","end","input_sha256","annotation_sha256","sampling"):
            other=copy.deepcopy(c); other[field]="changed"
            self.assertNotEqual(key(c,"affine",{}),key(other,"affine",{}))
        self.assertNotEqual(key(c,"affine",{}),key(c,"affine_spatial",{}))

    def test_resume_method_granularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Artifacts(tmp,{})
            store.save(case(),"affine",origin="executed",scores={})
            self.assertIsNotNone(store.load(case(),"affine"))
            self.assertIsNone(store.load(case(),"direct"))

    def test_cross_run_score_reuse_requires_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Artifacts(Path(tmp)/"a",{"code":"a"})
            source.save(case(),"affine",origin="executed",scores={})
            target=Artifacts(Path(tmp)/"b",{"code":"a"},[source.root])
            self.assertEqual(target.load(case(),"affine")["origin"],"reused")
            wrong=Artifacts(Path(tmp)/"c",{"code":"changed"},[source.root])
            self.assertIsNone(wrong.load(case(),"affine"))

    def test_snapshot_detects_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"x"; p.write_text("a"); s=snapshot([p]); verify_snapshot(s)
            p.write_text("b")
            with self.assertRaises(ValueError): verify_snapshot(s)

    def test_smoke_balanced_and_includes_changes(self):
        cases=[]
        for d in ("MOSEv2","LVOSv2","DAVIS2017"):
            for i in range(25):
                c=case(f"{d}{i}",d,str(i)); c.update(changed=d=="MOSEv2" and i<8, length_bin=i%3)
                if c["changed"]: c["cohort"]="legacy_dev"
                cases.append(c)
        picked=select_smoke(cases)
        self.assertEqual(len(picked),32)
        self.assertEqual(sum(c["changed"] for c in picked),8)
        self.assertEqual([sum(c["dataset"]==d for c in picked) for d in ("MOSEv2","LVOSv2","DAVIS2017")],[12,12,8])
        self.assertEqual(picked,select_smoke(list(reversed(cases))))

    def test_scheduler_budget_and_accuracy_independence(self):
        cases=[case(str(i),d,str(i)) for d in ("MOSEv2","LVOSv2","DAVIS2017") for i in range(10)]
        times=[dict(dataset=d,length_bin=0,seconds=10) for d in ("MOSEv2","LVOSv2","DAVIS2017")]
        schedule=freeze_schedule(cases,times,100)
        self.assertLessEqual(sum(s["estimated_seconds"] for s in schedule),80)
        for c in cases: c["score"]=123
        self.assertEqual(schedule,freeze_schedule(cases,times,100))

    def test_davis_manifest_schema(self):
        raw=normalize(dict(case_id="x",sequence="bike-packing",object_id=1,switch_frame=14,future_end_frame=67),"DAVIS2017")
        self.assertEqual(raw["video_id"],"bike-packing"); self.assertEqual(raw["first_prompt_stem"],0)

    def test_manifest_no_future_visibility_needed(self):
        inv=dict(frame_stems=[f"{i:05}" for i in range(30)],image_hashes=[str(i) for i in range(30)],
                 annotations={f"{i:05}":str(i) for i in range(30)},video_dir="x",annotation_dir="y")
        raw=dict(case_id="old",dataset="MOSEv2",video_id="v",object_id=1,first_prompt_stem=0,switch_stem=5,end_stem=29)
        with patch("manifest.inventory",return_value=inv):
            c=make_case(raw,set()); self.assertEqual(c["switch"],16); self.assertTrue(c["changed"])
            newer=make_case(raw,set(),additional=True); self.assertEqual(newer["switch"],16)

    def test_video_weight_not_case_weight(self):
        rows=[dict(case_id=str(i),dataset="d",video_id=v,method="affine",scores={"post_switch":{"J_and_F":s}})
              for i,(v,s) in enumerate([("a",0.),("a",0.),("b",1.)])]
        self.assertEqual(per_video(rows,("post_switch","J_and_F")),{("d","a"):0.,("d","b"):1.})
        rows += [dict(r,method="direct",scores={"post_switch":{"J_and_F":0.}}) for r in rows]
        self.assertEqual(paired(rows,"affine","direct")["mean"],.5)

    def test_report_excludes_partial_and_ungated(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Artifacts(tmp,{}); c=case()
            store.save(c,"affine",scores={"post_switch":{"J_and_F":1.}})
            r=build_report({"cases":[c]},store)
            self.assertEqual(r["complete_cases"],0); self.assertEqual(len(r["incomplete"]),1)

    def test_complete_report_has_common_pair_and_event_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Artifacts(tmp,{}); c=case()
            for method in METHODS:
                store.save(c,method,origin="executed",scores={"post_switch":{"J_and_F":.5,"J":.4,"F":.6},
                    "false_positives":{"gt_absent_frames":2,"rate":.25},
                    "reappearance":{"count":1,"no_recovery_rate":0.,"mean_recovery_length":1.}})
            store.save(c,GATE,gate_passed=True)
            r=build_report({"cases":[c]},store)
            self.assertEqual(r["complete_cases"],1)
            d=r["groups"]["all"]["datasets"]["MOSEv2"]
            self.assertEqual(d["paired"]["affine-minus-direct"]["mean"],0.)
            self.assertEqual(d["methods"]["affine"]["absent_videos"],1)
            self.assertIsNone(r["groups"]["all"]["equal_dataset_mean_JF"]["affine"])

    def test_cli_requires_audit_before_gpu_import(self):
        import run
        with tempfile.TemporaryDirectory() as tmp:
            args=run.parser().parse_args(["--stage","full","--run-dir",tmp])
            with self.assertRaisesRegex(ValueError,"audit first"):
                run.run(args)
            self.assertFalse((Path(tmp)/"budget.json").exists())

    def test_audit_orchestration_does_not_infer(self):
        import run
        selection={"schema":"test10.v1","seed":7,"cases":[case()]}
        fit={"fingerprint":{"sha256":"fit"},"videos":{"MOSEv2":[]},"shards":600,"records":11677}
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            args=run.parser().parse_args(["--stage","audit","--run-dir",tmp])
            stack.enter_context(patch("run.provenance",return_value={"files":{}}))
            stack.enter_context(patch("run.build",return_value=selection))
            stack.enter_context(patch("run.verify_fit_store",return_value=fit))
            stack.enter_context(patch("run.audit_legacy",return_value=[]))
            stack.enter_context(patch("run.select_smoke",return_value=selection["cases"]))
            real_read=run.read
            stack.enter_context(patch("run.read",side_effect=lambda p: {"store_fingerprint":fit["fingerprint"]}
                                     if str(p).endswith("train_report.json") else real_read(p)))
            run.run(args)
            self.assertTrue((Path(tmp)/"audit.json").exists())
            self.assertFalse((Path(tmp)/"budget.json").exists())


class AffineCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        imports()
        import torch
        cls.torch=torch

    def test_existing_affine_checkpoint_is_wx_plus_b(self):
        from core import LEGACY
        from vos_memory_inspector.transformer_translator import build_translator,SAM21_MEMORY_SPEC
        torch=self.torch
        model=build_translator("linear",SAM21_MEMORY_SPEC,SAM21_MEMORY_SPEC)
        model.load_state_dict(torch.load(LEGACY/"train/linear/epoch_030.pt",map_location="cpu",weights_only=True)["state_dict"])
        self.assertEqual(sum(p.numel() for p in model.parameters()),69952)
        x=torch.randn(3,64); p=torch.randn(3,256)
        torch.testing.assert_close(model._feature_head(x), x@model.feature.weight.T+model.feature.bias)
        torch.testing.assert_close(model._pointer_head(p), p@model.pointer.weight.T+model.pointer.bias)

    def test_component_wrapper_preserves_other_component(self):
        from vos_memory_inspector.state_schema import CanonicalState,StateSpec
        from vos_memory_inspector.translators import LinearStateTranslator,LearnedComponentPolicyTranslator
        t=self.torch; spec=StateSpec(4,2,2,5); model=LinearStateTranslator(spec,spec)
        spatial=t.randn(1,1,2,4,2,2); pointer=t.randn(1,1,2,5)
        state=CanonicalState(spatial_memory=spatial,object_pointer=pointer,presence_logits=t.zeros(1,1,2,1),
            frame_indices=t.tensor([[[0,1]]]),slot_order=t.tensor([[[0,1]]]),is_conditioning=t.tensor([[[True,False]]]),
            validity=t.ones(1,1,2,dtype=t.bool),object_ids=(1,),switch_frame=1)
        for component,unchanged in (("spatial_memory","object_pointer"),("object_pointer","spatial_memory")):
            result=LearnedComponentPolicyTranslator(model,learned_components=(component,)).translate(state)
            t.testing.assert_close(getattr(result,unchanged),getattr(state,unchanged))
            self.assertFalse(t.equal(getattr(result,component),getattr(state,component)))

    def test_logit_hash_detects_same_mask_different_logits(self):
        from runtime import logit_hash
        t=self.torch
        self.assertNotEqual(logit_hash(t.tensor([1.,2.])),logit_hash(t.tensor([1.,3.])))
        with self.assertRaises(ValueError): logit_hash(t.tensor([float("nan")]))

    def test_backbone_hooks_restore_on_exception(self):
        from runtime import Counter
        class Fake:
            def forward_image(self,x): return x
            def _get_image_feature(self,state,frame_idx,batch_size): return self.forward_image(frame_idx)
        p=Fake(); before=p.forward_image
        with self.assertRaises(RuntimeError):
            with Counter(p) as counter:
                p._get_image_feature({},16,1); self.assertEqual(counter.frames,[16]); raise RuntimeError()
        self.assertEqual(p.forward_image,before)

    def test_runtime_replay_restart_and_no_replay_with_cpu_predictor(self):
        """Exercise orchestration, hooks and interval checks without loading SAM2/GPU."""
        import runtime
        t=self.torch
        class FakePredictor:
            device=t.device("cpu")
            def forward_image(self,image): return image
            def _get_image_feature(self,inference_state,frame_idx,batch_size):
                if inference_state.get("last") != frame_idx:
                    self.forward_image(t.zeros(1,3,2,2)); inference_state["last"]=frame_idx
            def add_new_mask(self,state,frame_idx,obj_id,mask):
                self._get_image_feature(state,frame_idx,1)
            def propagate_in_video(self,state,start_frame_idx,max_frame_num_to_track,reverse=False):
                for p in range(start_frame_idx,start_frame_idx+max_frame_num_to_track+1):
                    self._get_image_feature(state,p,1)
                    yield p,[1],t.ones(1,1,2,2)
        class Frames:
            video_height=2; video_width=2
            def __len__(self): return 21
        @dataclass
        class State:
            spatial_memory: object
            object_pointer: object
            presence_logits: object
            def handoff_bytes(self): return 10
        class Identity:
            def translate(self,state): return state
        evaluator=runtime.Evaluator.__new__(runtime.Evaluator)
        evaluator.base=FakePredictor(); evaluator.methods={"direct":Identity()}; evaluator.model_load_s=0.
        payload=dict(state=State(t.zeros(1),t.zeros(1),t.zeros(1)), export_s=.1,last_mask=t.zeros(2,2).numpy())
        c=case(); c["annotation_dir"]="unused"
        with ExitStack() as stack:
            stack.enter_context(patch("runtime.now",side_effect=time.perf_counter))
            stack.enter_context(patch("runtime.inference_context",side_effect=lambda *_:nullcontext()))
            stack.enter_context(patch("runtime.torch.autocast",side_effect=lambda *a,**k:nullcontext()))
            stack.enter_context(patch("runtime.torch.cuda.reset_peak_memory_stats"))
            stack.enter_context(patch("runtime.torch.cuda.max_memory_allocated",return_value=0))
            stack.enter_context(patch.object(t.Tensor,"cuda",lambda self:self))
            stack.enter_context(patch("runtime.load_label_mask",return_value=t.ones(2,2).numpy()))
            inject=stack.enter_context(patch("runtime.inject_sam2_canonical_state"))
            for m in (GATE,"direct","last_mask","anchor_replay_4","anchor_replay_8","anchor_replay_16"):
                result=evaluator.method(c,Frames(),payload,payload,m)
                self.assertEqual(set(result["masks"]),{17,18,19,20})
                calls=result["runtime"]["past_backbone_frames"]
                if m in (GATE,"direct"): self.assertEqual(calls,[])
                elif m=="last_mask":
                    self.assertEqual(calls,[16]); self.assertTrue(result["runtime"]["empty_prompt"])
                    self.assertEqual(result["runtime"]["export_s"],0)
                else: self.assertEqual(calls,replay_frames(0,16,int(m.rsplit("_",1)[1])))
            def bad_inject(selected,predictor,inference_state):
                predictor._get_image_feature(inference_state,0,1)
            inject.side_effect=bad_inject
            with self.assertRaisesRegex(RuntimeError,"no-replay"):
                evaluator.method(c,Frames(),payload,payload,"direct")

    def test_prediction_blob_cross_run_and_checksum(self):
        from runtime import save_blob,load_blob
        with tempfile.TemporaryDirectory() as tmp:
            source=Artifacts(Path(tmp)/"a",{}); target=Artifacts(Path(tmp)/"b",{},[source.root])
            save_blob(source,case(),"predictions",{"masks":{17:((1,),b"x")}})
            self.assertEqual(load_blob(target,case(),"predictions")["masks"][17],((1,),b"x"))
            source.path(case(),"predictions",".pt").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError,"corrupt"):
                load_blob(target,case(),"predictions")


if __name__ == "__main__": unittest.main()
