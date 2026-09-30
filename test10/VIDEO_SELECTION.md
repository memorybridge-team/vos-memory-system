# test10 추가 평가 영상 목록

`heldout01`의 고정 목록이다. 데이터셋마다 서로 다른 영상 35개(기본 30개 + 추가 5개), 총 140개 영상에서 영상당 객체 1개·전환 1회를 평가했다. 아래 순서는 [frozen_selection.json](runs/heldout01/frozen_selection.json)의 순서와 같다. 객체 ID, switch 처리 index, 실제 frame stem, RGB·annotation hash는 해당 manifest에 기록했다.

| 데이터셋 | 사용 split | 기본 | 추가 | 합계 | 평가 성격 |
|---|---|---:|---:|---:|---|
| MOSEv2 | development/train의 기존 미평가 영상 | 30 | 5 | 35 | fit·checkpoint 선택·기존 test10 평가 영상 제외 |
| LVOS v2 | validation | 30 | 5 | 35 | fit·checkpoint 선택·기존 test10 평가 영상 제외 |
| DAVIS 2017 | train | 30 | 5 | 35 | 기존 validation 영상과 분리한 탐색적 평가; 공식 test 아님 |
| VOST | validation | 30 | 5 | 35 | 새 도메인의 객체별 독립 평가; 공식 leaderboard 점수 아님 |

모든 집합은 추론 전 seed 7의 결정적 순서와 영상 길이 구간을 이용해 정했다. 추가 5개는 성능이 아닌 남은 실행 시간을 기준으로, 미리 정한 순서대로 포함했다. 네 데이터셋의 사례 수는 동일하지만 영상 길이와 평가 frame 수는 동일하지 않다.

## MOSEv2

- 기본 30개: `e33cddc9`, `31rry8ca`, `k0vcz0st`, `aa185af5`, `51vddywa`, `aatd0ihe`, `5d107jy0`, `pz9rf3kp`, `or8eklj5`, `tug3h8uh`, `r2iallw7`, `bd9untye`, `ca619b7f`, `iijj01lq`, `zj3p2iaa`, `nu73npba`, `730bc486`, `vd2kzgc9`, `rj6on8oa`, `0644075b`, `1e95ef5e`, `edab084s`, `0518cuig`, `cv63itgy`, `54eda06d`, `qcyzntli`, `my6cd5xb`, `2080824a`, `ouyf2et0`, `hxaid41o`.
- 추가 5개: `5949ee84`, `hy97dn1d`, `zxh1jkdr`, `c9080ed1`, `l0ek62ei`.

## LVOS v2

- 기본 30개: `fqbAXAoT`, `UZzAPST3`, `vjG0jbkQ`, `rSFfONgp`, `3nsHQkEK`, `afoR2rH6`, `bxjTooAn`, `LEo03dYV`, `HYSm91eM`, `cmprgw5z`, `3bvEjhOT`, `UgZ759gD`, `yExgitit`, `Erx5Q6Vf`, `x3nD3QQ9`, `pyq6I5Hh`, `aT6JIUVU`, `SRRojucs`, `xWs99Q4N`, `8lxxCA5h`, `KhDr3o6f`, `xpI7xRWN`, `YfoFnMgm`, `4H7FEAHv`, `iWjeKM6a`, `vJ8W2TO5`, `NFbsxmYE`, `gYaLtvfc`, `Q7jVRHO5`, `N6CONZUW`.
- 추가 5개: `oWARMtpa`, `gAHn8WFw`, `Q3kk9fuH`, `usxBn0LJ`, `ae7pGvGx`.

## DAVIS 2017

- 기본 30개: `tuk-tuk`, `scooter-gray`, `stroller`, `motocross-bumps`, `kid-football`, `drone`, `crossing`, `stunt`, `bmx-bumps`, `dog-agility`, `tennis`, `mallard-water`, `surf`, `breakdance-flare`, `dog-gooses`, `horsejump-low`, `car-turn`, `color-run`, `dance-jump`, `tractor-sand`, `scooter-board`, `rallye`, `walking`, `train`, `classic-car`, `sheep`, `skate-park`, `motorbike`, `varanus-cage`, `dogs-scale`.
- 추가 5개: `swing`, `boat`, `koala`, `longboard`, `lindy-hop`.

## VOST

- 기본 30개: `6922_split_paper`, `10625_knead_dough`, `5118_squeeze_bag`, `559_cut_cucumber`, `5304_unpack_broccoli`, `9383_spread_cement`, `8267_pour_flour`, `4340_roll_dough`, `4031_cut_broccoli`, `4357_trim_dough`, `1206_cut_mango`, `5110_paint_nail`, `1205_cut_mango`, `4222_peel_wire`, `7202_clean_car`, `1210_cut_garlic`, `9691_divide_dough`, `4030_cut_broccoli`, `9692_divide_dough`, `8032_divide_dough`, `7049_paint_nail`, `9673_cut_tomato`, `3993_cut_tomato`, `7869_squeeze_bag`, `3562_break_egg`, `1179_cut_onion`, `4626_cut_paper`, `3161_peel_banana`, `7866_squeeze_bag`, `6503_apply_paint`.
- 추가 5개: `8028_divide_dough`, `1184_cut_chilli`, `4021_cut_broccoli`, `8024_divide_dough`, `1211_cut_garlic`.
