# Small → Base+ memory preconditioning protocol

상태: implementation protocol v1, 2026-09-27.

학습 대상은 `maskmem_features`와 `obj_ptr`의 독립 head뿐이다. Frame/object/slot
metadata는 복사하고, spatial/temporal PE와 RoPE는 Base+가 생성한다. Presence는
진단 sidecar이며 기본 handoff 입력이 아니다.

데이터는 LVOS v2 Train과 VOST Train을 각각 video/source-group 단위 80/10/10으로
고정한 후 state를 추출한다. Aligned pair 학습 뒤 같은 fixed transform을 유지한 채
native pair로 미세조정한다. 후보별 record, spatial position, optimizer budget과 seed는
같다. Fit 통계는 aligned fit에서만 계산한다.

Spatial 후보는 Raw, standardization, consumer projection aware, ZCA, output projection
SVD 순으로 screening한다. Pointer 후보는 Raw, standardization, no-object anchor,
anchor+consumer, final-linear canonicalization 순이다. Regularized 또는 truncated inverse는
lossy transform으로 표시한다.

실제 성공 기준은 held-out native Small prefix의 두 tensor를 변환·주입한 뒤 Base+가
prefix backbone call 0회로 `switch+1`부터 이어간 suffix J&F다. State MSE, K/V 및
attention similarity는 원인 진단 지표다.
