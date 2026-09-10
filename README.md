# vos-memory-system

## Appendix_1
#### 실험 과정
(1) Cold 상태에서 SAM2가 영상 전체를 추론한다
(2) 추론이 끝난 시점에서 memory bank를 그대로 복사한다
(3) 동일한 새 SAM2 추론 상태를 만든다
(4) 동일한 최초 mask를 입력하고 복사한 memory bank를 주입한다.
(5) 영상을 다시 0 프레임부터 추론하여 메모리, 프레임별 전체 정확도를 비교합니다

첫 번째 추론 시작: memory bank = {}
첫 번째 추론 완료: memory bank = {M1, M2, M3, … , MN}
두 번째 추론 시작 직전: {}
새 SAM2의 memory bank = {M1, M2, M3, … , MN}
두 번째 추론은 SAM2 원래 로직 그대로 실행

#### 실험 환경
SAM2 tiny, small, base-plus, large 4개 종류의 모델에서 DAVIS 2017, LVOS, MOSEv2 3가지 데이터셋 모두에서 테스트를 수행 → MOSEv2 는 너무 오래 걸려서 패스,  DAVIS와 LVOS도 전체 데이터셋이 아닌 일부 데이터셋만 사용
