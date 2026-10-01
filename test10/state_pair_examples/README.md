# Small→Base+ state-pair examples

These are two **state-only** examples from the RunPod `CMMT-task07-artifacts/production` cache, one per dataset. They are examples of the cached source/target states, not the full evaluation dataset or test10 results.

| Dataset | Split | Case | `.pt` SHA256 |
| --- | --- | --- | --- |
| MOSEv2 | development | `mose_0thyrhvg_obj1_switch4` | `a025c89ea65df4eff4bff543279150b101a0d7e5a8b1487a1b4bdb08dd0d6175` |
| LVOSv2 | development | `lvos_1umdNtrE_obj1_switch491` | `7331a5b495bf60f669a8cda35367a021e8e6cedf4793c9719bdc51839da36b0d` |

Each example includes its `.pt`, `.prepare.json` contract/provenance, and `.pt.sha256`. The metadata identifies `sam2.1-small` as source, `sam2.1-base-plus` as target, and `cache_mode: state_only`. Original RGB videos and annotations are not included.

Verify after checkout from the dataset subdirectories with `sha256sum -c *.pt.sha256`. The remaining cached pairs stay in external storage; these two samples do not establish dataset completeness or model performance.
