# RLinf 기반 VLA-RLVR 연구 저장소

이 저장소는 [RLinf](https://github.com/RLinf/RLinf)의 공식 실행 경로를 사용해 공개 OpenVLA-OFT 모델의 LIBERO 평가와 후속 분석을 재현하기 위한 연구용 작업 공간이다. 기존 `vla-rlvr` 프로젝트와 분리되어 있으며 모델 가중치나 RLinf 소스 전체를 Git에 넣지 않는다.

## 구성 한눈에 보기

| 경로 | 역할 | 현재 상태 |
| --- | --- | --- |
| `upstream/RLinf/` | 설치, 모델 로딩, Ray 환경·롤아웃, 평가를 담당하는 공식 코드의 독립 clone | 서버에 설치; commit `e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d` |
| `checkpoints/` | Hugging Face에서 받은 모델·LoRA 가중치 | 서버에 LIBERO-130 SFT·GRPO 다운로드; Git 제외 |
| `manifests/` | 모델 revision·파일 메타데이터, 실제 패키지 목록, 스모크 결과 | Git 추적 |
| `docs/` | 구조, 설치 기록, 모델 관계, 평가 프로토콜, 연구 계획 | 이 저장소에서 관리 |

**실행 흐름:** LIBERO 관측(영상·언어 지시) → RLinf의 OpenVLA-OFT 모델 → 7차원 동작, 8개씩의 action chunk → LIBERO 환경 → 성공 여부 집계. 평가 프로세스와 환경·롤아웃 worker는 RLinf가 관리한다. 자세한 코드 지도는 [RLinf 읽는 법](docs/RLINF_GUIDE.md)에 있다.

## 지금까지 확인한 결과

- 서버 연구 경로: `/nfs4/jhkim/repos/rlinf`. 이 로컬 저장소는 문서·설정·스크립트 편집용이며, GPU 설치와 평가 결과는 서버에서 얻었다.
- Original LIBERO용 격리 venv 설치 완료. 이름은 `.venv-rlvr-libero-cu121`이지만 **실제 PyTorch는 2.5.1+cu124**다. A5000에서 CUDA 텐서 계산이 통과했다.
- LIBERO assets `articulated_objects`, `scenes`, `stable_hope_objects` 존재 확인.
- LIBERO-130 SFT 베이스+LoRA와 GRPO 모델을 revision 고정으로 다운로드하고 원격 파일 목록·크기를 확인했다.
- RLinf 공식 평가 진입점으로 모델별 **1개 에피소드**를 끝까지 실행했다. SFT: `success_once=0`, 520 step. GRPO: `success_once=1`, 88 step, exit 0. 고정 RLinf 코드에서 `temperature_eval=1.6`은 `do_sample=false` 설정에도 불구하고 실제 stochastic 추론을 선택하므로 **두 스모크 모두 effective stochastic**이었다. 이는 실행 스모크 결과이지 성능 비교가 아니다.
- 추가 LoRA를 붙인 SFT deterministic nominal 수집은 Goal/Spatial/Long 각 0/50이었다. 공식 Goal task 0/trial 25 대조에서는 adapter 적용 실패(520 step), base 가중치만 사용 시 성공(128 step)이어서 SFT 로딩 방식을 재검증 중이다. [진단 기록](docs/MODELS.md)을 먼저 확인한다.
- LIBERO-PLUS 설치·OOD 평가, 4조건 통제 비교, 전체 suite 평가, SFT/RL 학습은 아직 수행하지 않았다.

## 문서 순서

1. [RLinf 읽는 법](docs/RLINF_GUIDE.md): upstream 코드, 설정, 실행 과정의 역할
2. [설치와 검증](docs/INSTALL.md): 서버 환경, 설치 명령, 확인된 문제
3. [모델과 LoRA](docs/MODELS.md): SFT/GRPO 대응 관계와 다운로드 provenance
4. [평가](docs/EVALUATION.md): 실제 통과한 명령과 비교 시 주의점
5. [실험 계획](docs/EXPERIMENTS.md): nominal → OOD → action/representation 분석
6. [Nominal 수집](docs/NOMINAL.md): Goal/Spatial/Long 매핑 감사, 평가 전용 수집기, 서버 실행·검증·재개 명령

`manifests/models_130.json`은 Hugging Face 파일별 크기와 제공된 LFS SHA-256을 기록한다. `manifests/installed-libero-packages.txt`는 실제 설치 환경의 inventory이며 lockfile은 아니다. `manifests/libero_130_smoke.json`은 두 번의 원격 실행에서 관찰한 값이다.

Upstream은 Git submodule이 아니라 지정 SHA의 독립 clone으로 재생성한다. `upstream/`, `checkpoints/`, 데이터와 실행 산출물은 `.gitignore`로 제외한다. 공개 수치의 출처는 [RLinf LIBERO 문서](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)이며 이 저장소의 스모크 결과와 혼동하지 않는다.
