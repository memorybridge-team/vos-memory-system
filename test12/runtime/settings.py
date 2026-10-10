"""모든 숫자와 경로는 여기에만 둔다.

다른 파일은 `import settings` 후 `settings.이름` 으로 읽는다 (복사해 두지 않는다).
"""

# ── 경로 ─────────────────────────────────────────────────────────────
DATA_ROOT = "/workspace"                 # RunPod 서버
DATA_FOLDERS = {                         # DATA_ROOT 아래 데이터셋별 폴더
    "lvos_v2": "CMMT/data/LVOSv2/extracted",
    "vost": "datasets/VOST/extracted/VOST",
    "m3vos": "CMMT/data/M3VOS-manual",
    "pumavos": "CMMT/data/PUMaVOS/extracted/PUBLIC_PUMaVOS",
}
OUTPUT_ROOT = "outputs"                  # 목록·결과·표가 모두 여기로
SAM2_CHECKPOINT_DIR = "/workspace/CMMT/checkpoints"
LVOS_SPLIT_FOLDER = None                 # val/valid가 둘 다 있으면 사용할 폴더를 명시
LVOS_ATTRIBUTE_FILE = None               # 속성 JSON이 여러 개면 파일 이름을 명시

# ── 모델 ─────────────────────────────────────────────────────────────
MODELS = {
    "small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "sam2.1_hiera_small.pt",
    },
    "base_plus": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
    },
}
SOURCE_MODEL = "small"                   # 전환 전에 도는 모델
TARGET_MODEL = "base_plus"               # 전환 뒤에 이어받는 모델
DEVICE = "cuda"
USE_BF16 = True                          # SAM2 공식 예제와 같이 bfloat16으로 돌림
OFFLOAD_STATE_TO_CPU = False             # GPU 메모리가 모자라면 True
MEMORY_WINDOW = 16                       # s-16~s의 non-cond 최대 17칸 + 모든 cond 칸 보존
                                         # (기본 SAM2가 읽는 범위: 최근 6장 spatial + 15장 pointer)

# ── 본 모델 (translator) ─────────────────────────────────────────────
# 팀 전달본 official_state_loss_final_delivery 를 푼 폴더 (안에 selected_state_loss_best/, source/)
TRANSLATOR_DIR = "/workspace/CMMT-official-isolated/official-20261005T012256KST/final-state-loss-20261005T1627KST/delivery"
TRANSLATOR_WEIGHTS = "selected_state_loss_best/translator_weights.pth"   # 선정 epoch 27
TRANSLATOR_SHA256 = "92802842b0f9c2247f95627784a4919625aaced43633c5203f619472ae0f17da"   # 전달 보고서 값
TRANSLATOR_SOURCE = "source/src"                                          # 팀 코드 (vos_memory_inspector)

# ── 영상 목록 ─────────────────────────────────────────────────────────
MIN_PRE_SWITCH_FRAMES = 8                # s-start: 최초 프롬프트 뒤 s까지 최소 8프레임

# ── 전환 시점 ─────────────────────────────────────────────────────────
SWITCH_FRACTIONS = (0.25, 0.50, 0.75)    # 객체 최초 등장부터 영상 끝까지의 전환 지점

# ── 비교군 ───────────────────────────────────────────────────────────
REPLAY_FRAMES = 8                        # Original-Prompt(s)+Replay-8: 전환 직전 다시 볼 프레임 수

# ── 채점 (주) ─────────────────────────────────────────────────────────
BOUNDARY_THRESHOLD = 0.008               # F: 경계 허용 거리 = 이미지 대각선 × 이 값 (DAVIS 기본값)
J_MAIN_DATASETS = ("vost_val", "m3vos")  # 주 지표가 J 인 데이터셋 (VOST 는 F 를 쓰지 않고, M3VOS 논문도 F 로 평가하지 않음). 나머지는 J&F

# ── 평가 지표 ────────────────────────────────────────────────────────
RECALL_J = 0.5                          # 실패 비율: 1 − mean(J > 0.5), J는 내부적으로 0~1
VIDEO_LIST_REVISION = 3                 # 25/50/75%, 전환별 최소 관찰 구간
EVALUATION_REVISION = 6                 # 전환 자격과 전체 프레임 SQLite 원점수
EVALUATION_RUNS = 3                     # 전체 평가 반복 횟수
EVALUATION_SEED = 0                     # 조건별 seed의 기준값

# 계산/저장 정의는 같지만 메모리 복사·측정 오버헤드를 줄인 실행 버전. 비용 비교 시 구분한다.
EVALUATION_RUNTIME_REVISION = 5
PREFETCH_FRAMES = True                  # 시간 측정 없는 추적 구간만 CPU 다음 프레임 준비
SCORING_WORKERS = 1                     # CPU 채점 worker 수. 0이면 동기 채점
SCORING_QUEUE_FRAMES = 4                # 실행/대기 중인 예측 마스크 수의 상한
GT_CACHE_MB = 128                       # 객체별 정답 마스크·경계·거리 변환 LRU 상한 (MiB)
RGB_CACHE_MB = 256                      # 객체 실행 안에서 공유하는 CPU 입력 LRU (측정 구간 우회)
RESTORATION_CACHE_MB = 128              # 같은 회차 Native float64/SST LRU 상한 (MiB)
BENCHMARK_SKIP_VISIBLE = True           # 미사용 visible 조회 생략. 일반 Session에는 적용하지 않음
SQLITE_JOURNAL_MODE = "DELETE"         # 공유 볼륨에서 WAL을 자동 사용하지 않음; 로컬 디스크만 WAL
SQLITE_BUSY_TIMEOUT_SECONDS = 30
