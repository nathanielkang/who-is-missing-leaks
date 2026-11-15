# 재현 코드

Python 3.13, numpy 2.5, scipy 1.18 (HiGHS). 이 폴더에서 실행.

| 스크립트 | 산출 | 실행 |
|---|---|---|
| `nh_table1.py` | 표 I 점추정, `data/nhanes_counts.npy` 생성 | `python3 nh_table1.py <nhanes_2015_2016.csv 경로>` |
| `nh_boot.py` | 표 I 부트스트랩 구간 | `python3 nh_boot.py 10000 10000` |
| `sim.py` → `sim_summary.py` | 표 II, 정리 5 평가 | `python3 sim.py 0 1000 2 chunk0.json` (시드 0–999, 프로세스 2개) … `python3 sim_summary.py chunk*.json` |
| `acs.py` | 표 III·IV | `python3 acs.py reported 10000` (V2 설계: `all`) |
| `extra.py` | 수송 세부 수치(1인당 보류 개수 등) | `python3 extra.py` |
| `synth.py` | 표 V, 탈락, self-masking | `python3 synth.py 100000` |
| `verify_theory.py` | 닫힌 형태의 수치 검증 | `python3 verify_theory.py` |
| `running_example.py` | 본문 running example의 수치 검증 | `python3 running_example.py` |

데이터
- `nhanes_2015_2016.csv`: https://github.com/kshedden/statswpy (NHANES/merged/). 집계본 `data/nhanes_counts.npy`는 [PSU, 연령대, 시민권, 패턴] 빈도.
- `data/acs2021_ca_adult_cells.json`: 2021 ACS 1-year PUMS 캘리포니아 성인의 셀 빈도. 키는 `연령대|CIT|FCITP|FWAGP|FPINCP|FSCHLP|FHINS1P` (연령대 0: 18–39, 1: 40–59, 2: 60+).
- `data/acs_extract_reported.npy`: 본문에 쓴 n = 40,000 추출본 [연령대, 시민권, 패턴].

패턴 번호는 비트 0–3이 필드 순서(NHANES: 소득, 음주, 혈압, 허리둘레 / ACS: 임금, 총소득, 학력, 고용주보험)의 응답 여부.
