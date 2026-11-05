# 06 재현 방법

## 표와 그림 다시 만들기

모델을 돌리지 않고 원자료에서 표를 다시 만든다. `numpy`와 `matplotlib`만 있으면 된다.

```bash
cd test14
bash code/analysis/unpack_runs.sh /tmp/test14_runs
python code/analysis/build_tables.py --runs /tmp/test14_runs --out /tmp/test14_tables
python code/analysis/make_figures.py results/tables /tmp/test14_figures
```

`paper_*.csv`는 논문 평가 DB가 있어야 다시 만들어진다. `vos-memory-benchmark/outputs/runpod_import_20261010`을 `--paper`로 넘긴다.

```bash
python code/analysis/build_tables.py --runs /tmp/test14_runs --paper <runpod_import_20261010 경로> --out /tmp/test14_tables
```

파일 무결성은 다음 명령으로 확인한다.

```bash
cd test14 && shasum -a 256 -c MANIFEST.sha256
```

## 실험 환경

| 항목 | 값 |
|---|---|
| GPU | RTX 4080 SUPER 16 GB, 32 GB |
| 평가와 파인튜닝 | torch 2.8 (`code/review/env28.sh`) |
| MSE 학습과 보고서 | torch 2.11 (`code/review/env.sh`) |
| 저장소 코드 스냅샷 | `prepared-source.tar.gz`, SHA256 `aa419e37…` |
| 공식 인덱스 | `raw_index.json`, SHA256 `3ae8b00b…` |
| SAM2 체크포인트 | SAM 2.1 Small, Base+ |

torch 2.11에서는 prebuilt SAM2 `_C.so`가 프레임당 약 0.57 MiB의 GPU 메모리를 흘린다. 평가는 torch 2.8로 해야 한다. 코드의 서버 경로는 실행 당시 값 그대로 두었다.

## 데이터 준비

```bash
python code/review/build_memmap.py   # state pair를 메모리맵으로 변환
python code/review/build_pool.py     # 포인터 실험용 풀링 특징
```

결과는 334 GB이고 저장소에는 없다. 만들어진 파일의 해시는 [data_sha256.txt](../data_manifest/data_sha256.txt)에 있다.

## 학습

MSE 학습 예시이다. 설정 전체는 `results/tables/training_models.csv`의 `args` 열과 `weights/spatial/*/result.json`에 있다.

```bash
python code/review/train_spatial.py --name mlp_context --updates 30000 --batch 64 --lr 3e-4
```

2라운드 파인튜닝이다. 권장 레시피이고 RTX 4080 SUPER에서 약 2.5시간 걸린다.

```bash
python code/review/train_e2e.py --tag e2e_ctx --init mlp_context --updates 12000 --batch 4 --lr 1e-4 \
  --w-feature 10 --w-mask 5 --w-score .5 --lam .1
```

3라운드 파인튜닝이다.

```bash
python code/review/train_e2e.py --tag e2e3_ctx --init e2e_ctx_last --updates 12000 --batch 4 --lr 1e-4 \
  --alone-share .5 --cond-share .8 --ridge-pointer --w-feature 10 --w-mask 5 --w-score 1 --w-logit .1 \
  --w-dice 1 --lam .1 --lvos-share .25 --held-out 512
```

평가는 `model.pt`가 아니라 마지막 가중치로 해야 한다. `use_last.py`로 `last.pt`에서 뽑는다. `e2e_ctx`, `e2e_mlp`의 `model.pt`는 조기 스텝 값이고 `e2e_ctx_last`, `e2e_mlp_last`가 마지막 가중치다.

## 평가

```bash
python code/review/handoff_eval.py --autocast --split val --tag t1_val --method ...
python code/review/more_eval.py --autocast --split dev --dataset MOSEv2 --limit 300 --tag t10_mose --method ...
python code/review/unseen_more.py --dataset pumavos --tag pumavos3 --fractions 0.25,0.5,0.75 --method ...
```

실행에 쓴 전체 명령은 `code/review/pipeline*.sh`에 있다. 모든 평가는 케이스별 JSON을 쓰고, 다시 실행하면 끝난 케이스를 건너뛴다.

## 실행 관리

| 파일 | 역할 |
|---|---|
| `pipeline*.sh` | SSH와 무관하게 도는 감독 스크립트. 실패 시 재시도 |
| `governor.py` | GPU별 프로세스 수를 제한. SIGSTOP/SIGCONT로 조절 |
| `status.sh` | 진행 상황 확인 |
| `stop_all.py` | 정확한 PID로 전체 중단 |

제한 설정의 마지막 상태는 `logs/control`에 있다.
