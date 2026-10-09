# Paired-state 제작 소스 — 2026-10-09

[ZIP 다운로드](paired_state_collection_source_20261009.zip) · [설치 및 실행 안내](OPERATOR_GUIDE.md)

SAM 2.1 Small/Base+ paired-state 제작에 사용되는 현재 로컬 collector와 수정된 runtime snapshot이다. MOSEv2/LVOS v2 train·validation과 VOST/PUMaVOS/M³-VOS manifest를 포함한다. 원본 데이터셋·checkpoint·cache·가상환경은 별도로 준비한다.

ZIP에는 runtime 모듈, collector/manifest adapter, prompt 정렬 복구 도구, frozen manifests/splits, pyproject.toml, 실행 안내 및 파일별 원본 위치·SHA-256 inventory가 들어 있다. Prompt 정렬 수정본은 전체 영상 runtime 좌표에서 실제 prompt frame에 mask를 넣고 Small/Base+가 같은 조건으로 switch까지 처리한다.

- 파일: `paired_state_collection_source_20261009.zip`
- 크기: 475,626 bytes
- SHA-256: `6a196770d1342cd9541f1299484137dbc64e44e25f7d01563c1e370792a17aff`
- ZIP entry: 51개, 원본 파일 49개를 압축 내 스트림 hash로 대조해 일치 확인
- 원본 소스: `vos-memory-translator-nonlinear` Task 07 reference와 로컬 CMMT workspace의 미커밋 변경을 포함한 snapshot. 원격 main의 특정 commit과 동일하다고 가정하지 않는다.

이 게시 작업에서는 ZIP 내용의 SHA-256과 파일 목록을 확인했다. 새 GPU 수집 실험은 수행하지 않았다. 기존 prompt/cache 관련 CPU tests 22개와 collector tests 4개, 복구 cache 38개 production 검증 결과가 묶음 보고서에 기록돼 있다.
