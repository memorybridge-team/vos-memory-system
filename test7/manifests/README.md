# Frozen evaluation manifests

이 디렉터리는 데이터셋의 원본 이미지·마스크를 저장하지 않는다. 같은 dataset/split/video/object/switch 조합을 반복해서 평가하기 위한 작은 JSON manifest만 version control 한다.

## DAVIS 2017 validation v1

- File: `davis2017_val_v1.json`
- Dataset: DAVIS 2017 trainval, 480p validation split
- Seed: `7`
- Sequences: `30`
- Cases: `249`
- File SHA-256: `5b036f173d74e6939c3096fa03e0af64f3dfba8bdaff21a43e1db8f5c1ada2f5`

Ground truth는 switch case를 고정하고 난이도 tag를 붙이는 데만 쓴다. 모델 입력, handoff payload, translator 학습 target 선택에는 GT pixel을 사용하지 않는다. 원본 dataset root는 runtime argument로만 제공하며 manifest에 기록하지 않는다.

생성 명령은 다음과 같다.

```bash
cmmt-davis-build-manifest \
  --root /path/to/DAVIS \
  --split val --resolution 480p --seed 7 \
  --output manifests/davis2017_val_v1.json
```

## MOSEv2 validation v1

- File: `mosev2_valid_v1.json`
- Dataset: MOSEv2 v2 validation split
- Videos: `433`
- Cases: `1,720` (three fixed temporal switch candidates per first-frame object)
- File SHA-256: `1adccb98aa5a73d48c34eb72ed8990f1cc00925e6cc2242c40dc0ce43263fd00`
- Annotation policy: validation publishes only the first-frame mask. Future visibility,
  disappearance and reappearance tags are unavailable locally and are not inferred.

생성 명령:

```bash
PYTHONPATH=src python -c "from vos_memory_inspector.mose import build_mosev2_evaluation_manifest, write_mosev2_evaluation_manifest; m=build_mosev2_evaluation_manifest('/path/to/MOSEv2/extracted', split='valid'); write_mosev2_evaluation_manifest(m, 'manifests/mosev2_valid_v1.json')"
```

## MOSEv2 train v1

- File: `mosev2_train_v1.json`
- Videos: `3,666`
- Cases: `20,841`
- Annotation policy: dense train annotations from `meta_train.json`
- Manifest content SHA-256: `3051f1cb2025ae6a8565544e4311a0cb56efc33d9be77623c9bfc6f98814bd07`

Train은 validation처럼 first-frame-only가 아니므로 `meta_train.json`의 객체 목록·영상 길이와
실제 dense mask를 함께 확인해 객체별 최초 등장 frame 이후에만 3개 temporal switch 후보를 만든다.
전체 311,843개 train mask 스캔 결과, 최종 평가 대상 객체의 최초 prompt frame은 모두 0이었지만
이 값은 더 이상 가정이 아니라 실제 annotation으로 검증된 값이다.

```bash
cmmt-mose-build-manifest \
  --root /path/to/MOSEv2/extracted \
  --split train --seed 7 \
  --output manifests/mosev2_train_v1.json
```

## Video-level fit/development 후보

train manifest의 case를 video ID 기준으로 seed 7, 80/20 비율로 나누며 한 영상의
객체·switch case가 fit과 development에 동시에 들어가지 않게 한다.

```bash
python scripts/build_video_splits.py \
  --input manifests/mosev2_train_v1.json \
  --output-prefix manifests/mosev2_train_v1
```

최종 split은 DAVIS fit 43 videos/466 cases와 development 16/133,
MOSEv2 fit 2,680/16,931과 development 615/3,910,
LVOS v2 fit 347/1,488과 development 73/315다. split manifest는 case를 복제하지 않고
video membership과 source manifest checksum만 보존한다.

## LVOS v2 validation

LVOS v2 Eval archive는 공식 `valid.zip`으로 확보했다. RunPod에서 압축 해제한 결과는
140개 video directory, JPEG 66,056개, annotation 66,056개이며 archive SHA-256은
`beb488046f74e0cb4154a0cb2bcc2c79cae858da0693e5adabcf966a5712d4e2`이다.
manifest 파일은 140 sequences·714 cases를 담고, content SHA-256은
`a8afee3094d3326c04f1898108a19348c5c3ad17aa6a1ded653e4051c5d848a5`이다.
LVOS v2 loader는 공식 metadata의 object frame range 안에 실제 존재하는 sparse frame ID만
사용해 switch 후보를 만든다.
현재 archive의 `meta.json`은 loader 입력용 `val_meta.json`으로 파생해 사용했으며,
attribute·prompt/correction loader 검증은 후속 작업이다.

```bash
cmmt-lvos-build-manifest \
  --root /path/to/LVOS-v2 \
  --split val --seed 7 \
  --output manifests/lvosv2_val_v1.json
```

## LVOS v2 train v1

- File: `lvosv2_train_v1.json`
- Dataset: LVOS v2 official train split
- Videos: `420`
- Cases: `1,803` (object frame range 안의 세 temporal switch 후보)
- Archive size: `22,414,546,249` bytes
- Archive SHA-256: `e4c0cfcb400dbd103ddea6eac3b5ffb5cdd3d50cb87013bf457d963a0b248c0b`
- Manifest content SHA-256: `8292b1b4674a2ef34955c908e980826f3d041c518a035b5f5b42ad8fb49df74a`
- `meta.json`의 object별 frame range를 사용하며, train mask는 로컬에서 future 평가에 사용할 수 있다.

```bash
cmmt-lvos-build-manifest \
  --root /path/to/LVOS-v2 \
  --split train --seed 7 \
  --output manifests/lvosv2_train_v1.json
```
