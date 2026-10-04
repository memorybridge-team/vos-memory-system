"""Record weighting, frozen comparison conditions, and trained-model bundle boundaries."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comparable_training as cmp
from core import read, write, sha
from state_pairs import save_pair, MODELS
from training import load_models, fresh_model, training_report
from vos_memory_inspector.state_schema import CanonicalState
from vos_memory_inspector.upstream import SUPPORTED_SAM2_COMMIT
import torch


def pair(path, video, count=1, target_value=.125):
    frames = torch.arange(count).reshape(1,1,count)
    source = CanonicalState(spatial_memory=torch.zeros(1,1,count,64,64,64,dtype=torch.bfloat16),
        object_pointer=torch.zeros(1,1,count,256), presence_logits=torch.zeros(1,1,count,1),
        frame_indices=frames, slot_order=frames.clone(), is_conditioning=(frames==0),
        validity=torch.ones_like(frames,dtype=torch.bool), object_ids=(1,), switch_frame=count-1)
    target = replace(source, spatial_memory=source.spatial_memory+target_value,
                     object_pointer=source.object_pointer+target_value)
    meta = dict(MODELS, upstream_commit=SUPPORTED_SAM2_COMMIT, video_id=video, object_id=1,
                switch_frame=count-1, num_frames=count+10, cache_mode='state_only', active_memory_only=True,
                num_maskmem=7, max_obj_ptrs_in_encoder=16)
    save_pair(path, source, target, meta)
    return dict(dataset='LVOSv2', video_id=video, path=str(path), sha256=sha(path)), source


def inputs(root, fit_records=1, dev_records=1, dev_value=.125):
    rows, entries = [], []
    for video, split, count, value in [('fit','train',fit_records,.125),('dev','validation',dev_records,dev_value)]:
        row, source = pair(root/f'{video}.pt', video, count, value)
        row['split'] = split; rows.append(row)
        entries.append(dict(path=row['path'],sha256=row['sha256'],records=count,
            valid_indices=source.validity.nonzero().tolist(), case=dict(case_id=video,video_id=video,
                paired_split='fit' if split=='train' else 'development')))
    selection=root/'pairs.json'; index=root/'raw_index.json'
    write(selection,dict(schema='test10.pair_selection.v1',pairs=rows))
    write(index,dict(schema_version=cmp.INDEX_SCHEMA,state='tensor_validated',
        identity=dict(scope='full_frozen_fit_development'),entries=entries,
        normalization=dict(scope='fit_only',scales=dict(spatial=.125,pointer=.125))))
    return selection,index


class ComparableTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_fit_normalization_excludes_development_and_preserves_every_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); selection,index=inputs(root,3,2,8.)
            data=cmp.PairData(index,cache_gib=0)
            self.assertEqual(cmp.fit_scales(data),dict(spatial=.125,pointer=.125))
            self.assertEqual([len(b[0]) for b in data.batches('train',2)], [2,1])
            self.assertEqual([len(b[0]) for b in data.batches('validation',2)], [2])
            self.assertEqual(data.collection['fingerprint'],cmp.PairData(selection).collection['fingerprint'])
            result=cmp.validate(fresh_model('affine'),data,cmp.fit_scales(data),cmp.recipe(),'cpu')
            self.assertEqual(result['records'],2)
            self.assertAlmostEqual(result['loss'],8192.)

    def test_accumulated_tail_update_matches_full_batch(self):
        torch.manual_seed(17)
        first=fresh_model('affine'); second=copy.deepcopy(first)
        batch=(torch.randn(5,64,64,64),torch.randn(5,256),torch.randn(5,64,64,64),torch.randn(5,256))
        config=cmp.recipe(); config['microbatch_records']=2
        reference=dict(config,microbatch_records=5)
        for model,cfg in ((first,config),(second,reference)):
            optimizer=torch.optim.SGD(model.parameters(),lr=.03)
            cmp.optimize_window(model,batch,optimizer,dict(spatial=2.,pointer=.5),cfg,'cpu')
        for actual,expected in zip(first.parameters(),second.parameters()):
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7)

    def test_corruption_and_split_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); selection,index=inputs(root)
            data=cmp.PairData(selection)
            (root/'fit.pt').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'changed'): list(data.batches('train',16))
            selection,index=inputs(root)
            doc=read(selection); doc['pairs'][1]['video_id']='fit'; write(selection,doc)
            with self.assertRaisesRegex(ValueError,'video'): cmp.PairData(selection)

    def test_fixed_thirty_epoch_affine_only_training_and_reference_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,index=inputs(root)
            args=cmp.parser().parse_args(['train','--index',str(index),'--output-dir',str(root/'affine'),'--device','cpu'])
            report=cmp.train(args)
            self.assertEqual(report['completed_epochs'],30)
            self.assertEqual(report['optimizer_steps'],30)
            self.assertEqual(set(load_models(args.output_dir,'cpu')),{'affine'})
            with self.assertRaisesRegex(ValueError,'missing'): training_report(args.output_dir,['transformer'])
            ref=root/'reference'; (ref/'translator').mkdir(parents=True)
            cfg=cmp.recipe()
            configuration=dict(lr=.0003,weight_decay=.0001,lambda_spatial=1.,lambda_pointer=1.,lambda_cos=0.,
                max_epochs=30,warmup_fraction=.05,min_lr=.00001,clip_norm=1.,seed=7,precision='fp32',augmentation='none')
            identity=dict(configuration=configuration,collection_digest=sha(index),training_contract=dict(global_batch=64))
            write(ref/'run.json',dict(identity=identity,normalization=report['normalization']))
            history=[dict(epoch=e,records=1,optimizer_step=e,dev=dict(loss=abs(e-3)+.5,records=1)) for e in range(1,31)]
            write(ref/'metrics/history.json',history)
            export=ref/'translator/epoch-00003.pth'; torch.save(fresh_model('transformer').to_payload(),export)
            write(export.with_suffix('.json'),dict(identity=identity,epoch=3,weights=dict(sha256=sha(export))))
            attach=SimpleNamespace(affine_dir=args.output_dir,transformer_run=ref,output_dir=root/'bundle')
            bundled=cmp.attach_transformer(attach)
            self.assertEqual(bundled['models']['transformer']['epoch'],3)
            self.assertEqual(set(load_models(attach.output_dir,'cpu')),{'affine','transformer'})
            self.assertEqual(set(load_models(attach.output_dir,'cpu',['affine'])),{'affine'})
            # Index storage metadata may differ; identical split/pair/record content is comparable.
            other_index=root/'relocated_index.json'
            write(other_index,dict(read(index),storage_note='relocated copy'))
            relocated=read(ref/'run.json'); relocated['identity']['collection_digest']=sha(other_index)
            write(ref/'run.json',relocated)
            export_meta=read(export.with_suffix('.json')); export_meta['identity']=relocated['identity']
            write(export.with_suffix('.json'),export_meta)
            attach.reference_index=other_index; attach.output_dir=root/'relocated_bundle'
            self.assertEqual(cmp.attach_transformer(attach)['models']['transformer']['epoch'],3)
            foreign=read(ref/'run.json'); foreign['identity']['configuration']['lr']=.001
            write(ref/'run.json',foreign)
            attach.output_dir=root/'bad_bundle'
            with self.assertRaisesRegex(ValueError,'recipe differs'): cmp.attach_transformer(attach)

    def test_budget_stop_never_claims_completed_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,index=inputs(root)
            args=cmp.parser().parse_args(['train','--index',str(index),'--output-dir',str(root/'out'),
                '--device','cpu','--budget-hours','0.0000000001'])
            report=cmp.train(args)
            self.assertEqual(report['status'],'incomplete')
            self.assertEqual(report['completed_epochs'],0)
            with self.assertRaisesRegex(ValueError,'completed'): training_report(args.output_dir)

    def test_single_dataset_smoke_and_report_do_not_require_other_datasets(self):
        from run import comparison_smoke
        from report import build_report
        from core import Artifacts, GATE
        cases=[dict(case_id=str(i),dataset='LVOSv2',video_id=str(i),length_bin=i%3,
                    object_id=1,first=0,switch=0,end=10,cohort='additional',checkpoint_video=False,
                    frame_stems=[str(i) for i in range(11)],
                    input_sha256='rgb',annotation_sha256='gt',sampling=dict(raw_stride=5)) for i in range(6)]
        selected=comparison_smoke(cases,7)
        self.assertEqual(len(selected),3)
        self.assertEqual({c['length_bin'] for c in selected},{0,1,2})
        self.assertEqual(selected,comparison_smoke(list(reversed(cases)),7))
        with tempfile.TemporaryDirectory() as tmp:
            store=Artifacts(tmp,dict(training=dict(comparison_protocol=cmp.recipe()['schema'],config=cmp.recipe())))
            for case in cases:
                store.save(case,'affine',origin='executed',scores=dict(post_switch=dict(J=.5,F=.5,J_and_F=.5)))
                store.save(case,GATE,gate_passed=True)
            report=build_report(dict(cases=cases,methods=['affine']),store)
            self.assertEqual(report['groups']['all']['equal_dataset_mean_JF']['affine'],.5)
            self.assertEqual(report['training_conditions']['global_records'],64)


if __name__=='__main__': unittest.main()
