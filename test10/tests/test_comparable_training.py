"""Record weighting, frozen comparison conditions, and trained-model bundle boundaries."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
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


def inputs(root, fit_records=1, dev_records=1, dev_value=.125, recorded_scale=.125):
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
        normalization=dict(scope='fit_only',scales=dict(spatial=recorded_scale,pointer=recorded_scale))))
    return selection,index


def transformer_run(root, index, scales, *, records=1, dev_records=1, losses=None, **overrides):
    """Completed reference DDP run files (run.json, metrics/history.json) without weights."""
    configuration=dict(lr=.0003,weight_decay=.0001,lambda_spatial=1.,lambda_pointer=1.,lambda_cos=0.,
        max_epochs=30,warmup_fraction=.05,min_lr=.00001,clip_norm=1.,seed=7,precision='fp32',augmentation='none')
    configuration.update(overrides)
    identity=dict(configuration=configuration,collection_digest=sha(index),training_contract=dict(global_batch=64),
                  scope='state_supervised_ddp')
    write(root/'run.json',dict(identity=identity,normalization=dict(scope='fit_only',scales=scales),
                               JF_early_stopping_applied=False))
    losses=losses or [abs(e-3)+.5 for e in range(1,31)]
    per_epoch=-(-records//64)
    write(root/'metrics/history.json',[dict(epoch=e,records=records,optimizer_step=e*per_epoch,
        dev=dict(loss=losses[e-1],records=dev_records)) for e in range(1,31)])
    return identity


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
            identity=transformer_run(ref,index,report['normalization']['scales'])
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

    def test_every_epoch_is_kept_and_selection_follows_reference_best_state_loss(self):
        self.assertEqual(cmp.select_epoch([dict(epoch=1,loss=2.),dict(epoch=2,loss=1.),dict(epoch=3,loss=1.)]),3)
        self.assertEqual(cmp.select_epoch([dict(epoch=1,loss=1.),dict(epoch=2,loss=2.)]),1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,index=inputs(root)
            args=cmp.parser().parse_args(['train','--index',str(index),'--output-dir',str(root/'affine'),'--device','cpu'])
            report=cmp.train(args)
            self.assertEqual([e['epoch'] for e in report['epoch_checkpoints']],list(range(1,31)))
            for entry in report['epoch_checkpoints']:
                self.assertEqual(sha(args.output_dir/entry['checkpoint']),entry['sha256'])
            chosen=report['models']['affine']
            expected=cmp.select_epoch([dict(epoch=h['epoch'],loss=h['validation']['loss']) for h in report['history']])
            self.assertEqual(chosen['epoch'],expected)
            self.assertEqual(sha(args.output_dir/'affine.pt'),
                             sha(args.output_dir/f'epochs/affine-epoch-{expected:05d}.pt'))
            self.assertEqual(report['model_contract']['parameters'],69952)
            # A declared epoch (e.g. the rule that chose the compared Transformer) yields a loadable bundle.
            selected=cmp.select_checkpoint(SimpleNamespace(affine_dir=args.output_dir,epoch=30,
                reason='Transformer result uses the final epoch',output_dir=root/'final'))
            self.assertEqual(selected['models']['affine']['epoch'],30)
            self.assertEqual(selected['models']['affine']['selection']['rule'],'declared_epoch')
            self.assertEqual(set(load_models(root/'final','cpu')),{'affine'})
            with self.assertRaisesRegex(ValueError,'epoch 31'):
                cmp.select_checkpoint(SimpleNamespace(affine_dir=args.output_dir,epoch=31,reason='x',output_dir=root/'bad'))

    def test_raw_index_scales_are_the_ones_trained_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,index=inputs(root,recorded_scale=.125*(1+5e-8))
            data=cmp.PairData(index,cache_gib=0)
            self.assertEqual(cmp.fit_scales(data)['spatial'],.125*(1+5e-8))
            root=Path(tmp)/'far'; root.mkdir(); _,index=inputs(root,recorded_scale=.126)
            with self.assertRaisesRegex(ValueError,'fit RMS'): cmp.fit_scales(cmp.PairData(index,cache_gib=0))

    def test_affine_contract_rejects_other_translators(self):
        self.assertEqual(cmp.affine_contract(fresh_model('affine'))['parameters'],69952)
        with self.assertRaisesRegex(ValueError,'Wx\\+b'): cmp.affine_contract(fresh_model('residual_mlp'))

    def test_transformer_run_is_checked_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,index=inputs(root)
            transformer_run(root/'ref',index,dict(spatial=.125,pointer=.125))
            args=cmp.parser().parse_args(['train','--index',str(index),'--output-dir',str(root/'ok'),'--device','cpu',
                                          '--transformer-run',str(root/'ref')])
            report=cmp.train(args)
            self.assertFalse(report['transformer_reference']['weights_read'])
            self.assertEqual(report['transformer_reference']['best_state_loss_epoch'],3)
            self.assertNotIn(str((root/'ref/run.json').resolve()),report['files'])
            for name,change,message in [('lr',dict(lr=.001),'recipe differs'),
                                        ('records',dict(),'coverage'),
                                        ('scales',dict(),'normalization')]:
                ref=root/f'ref_{name}'
                transformer_run(ref,index,dict(spatial=.25,pointer=.125) if name=='scales' else dict(spatial=.125,pointer=.125),
                                records=2 if name=='records' else 1,**change)
                bad=cmp.parser().parse_args(['train','--index',str(index),'--output-dir',str(root/f'bad_{name}'),
                                             '--device','cpu','--transformer-run',str(ref)])
                with self.assertRaisesRegex(ValueError,message): cmp.train(bad)
                self.assertEqual(read(root/f'bad_{name}'/'train_report.json')['completed_epochs'],0)
            foreign=root/'ref_foreign'; transformer_run(foreign,index,dict(spatial=.125,pointer=.125))
            moved=read(foreign/'run.json'); moved['identity']['collection_digest']='0'*64; write(foreign/'run.json',moved)
            bad=cmp.parser().parse_args(['train','--index',str(index),'--output-dir',str(root/'bad_data'),
                                         '--device','cpu','--transformer-run',str(foreign)])
            with self.assertRaisesRegex(ValueError,'reference-index'): cmp.train(bad)

    @unittest.skipUnless(os.environ.get('TEST10_REFERENCE_REPO'), 'set TEST10_REFERENCE_REPO to the translator git checkout')
    def test_bitwise_equal_to_reference_ddp_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool=Path(__file__).resolve().parents[1]/'tools/verify_reference_recipe.py'
            result=subprocess.run([sys.executable,str(tool),'--reference-repo',os.environ['TEST10_REFERENCE_REPO'],
                                   '--epochs','3','--work-dir',tmp],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout[-3000:]+result.stderr[-3000:])
            comparison=json.loads((Path(tmp)/'comparison.json').read_text())
            self.assertTrue(comparison['exact_counts'])
            self.assertEqual(comparison['max_weight_abs_difference'],0.)
            self.assertEqual(comparison['max_dev_loss_abs_difference'],0.)

    def test_affine_only_evaluation_audit_needs_no_transformer(self):
        import run
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,index=inputs(root)
            args=cmp.parser().parse_args(['train','--index',str(index),'--output-dir',str(root/'affine'),'--device','cpu'])
            cmp.train(args)
            def case(i,video):
                return dict(case_id=f'LVOSv2:{video}:obj1',dataset='LVOSv2',video_id=video,object_id=1,first=0,switch=4,end=20,
                            cohort='heldout_core',checkpoint_video=False,length_bin=i%3,frame_stems=[str(f) for f in range(21)],
                            input_sha256='rgb',annotation_sha256='gt',sampling=dict(raw_stride=5))
            def audit(run_dir,videos):
                selection=root/f'{run_dir}.json'
                write(selection,dict(schema='test10.v1',seed=7,cases=[case(i,v) for i,v in enumerate(videos)]))
                prov=dict(files={},training=training_report(args.output_dir),training_dir=str(args.output_dir.resolve()))
                with patch('run.provenance',return_value=prov),patch('run.audit_legacy',return_value=[]):
                    run.run(run.parser().parse_args(['--stage','audit','--run-dir',str(root/run_dir),'--selection',str(selection),
                        '--training-dir',str(args.output_dir),'--methods','affine']))
            audit('eval',['v1','v2','v3','v4'])
            self.assertEqual(read(root/'eval/selection.json')['methods'],['affine'])
            self.assertEqual(read(root/'eval/audit.json')['fit']['videos'],{'LVOSv2':['dev','fit']})
            self.assertEqual(len(read(root/'eval/smoke_selection.json')['case_ids']),3)
            with self.assertRaisesRegex(ValueError,'overlap'): audit('leak',['v1','dev'])

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
            store=Artifacts(tmp,dict(training=dict(comparison_protocol=cmp.recipe()['schema'],config=cmp.recipe(),
                models=dict(affine=dict(epoch=12,origin='fresh',selection=dict(rule='declared_epoch',epoch=12))))))
            for case in cases:
                store.save(case,'affine',origin='executed',scores=dict(post_switch=dict(J=.5,F=.5,J_and_F=.5)))
                store.save(case,GATE,gate_passed=True)
            report=build_report(dict(cases=cases,methods=['affine']),store)
            self.assertEqual(report['groups']['all']['equal_dataset_mean_JF']['affine'],.5)
            self.assertEqual(report['training_conditions']['global_records'],64)
            self.assertEqual(report['comparison_checkpoints']['affine']['selection']['epoch'],12)
            self.assertIsNone(report['transformer_reference'])


if __name__=='__main__': unittest.main()
