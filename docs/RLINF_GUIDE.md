# RLinf를 이 프로젝트에서 읽는 법

## RLinf는 무엇인가

[RLinf](https://github.com/RLinf/RLinf)는 LLM·VLM·VLA의 강화학습과 평가를 위한 오픈소스 인프라다. 이 프로젝트에서는 RLinf의 분산 실행·환경 worker·모델 worker·설정 체계를 그대로 활용한다. 우리의 연구 저장소는 평가 조건과 결과 기록을 그 바깥에서 관리한다. [공식 LIBERO 예제](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)는 OpenVLA-OFT와 LIBERO의 학습·평가 흐름을 보여 준다.

## 어디에서 무엇을 찾나

| upstream 경로 | 이 프로젝트에서의 역할 |
| --- | --- |
| `requirements/install.sh` | uv venv와 `embodied --model openvla-oft --env maniskill_libero` 패키지 구성 |
| `evaluations/run_eval.sh` | 평가 CLI. benchmark·config 이름을 받아 렌더링 환경, 로그 경로와 Python entrypoint 설정 |
| `evaluations/eval_embodied_agent.py` | Hydra config를 읽어 평가 runner 시작 |
| `evaluations/libero/libero_10_openvlaoft_eval.yaml` | LIBERO/OpenVLA-OFT 평가의 기본값. LIBERO-130은 CLI override로 선택 |
| `examples/embodiment/config/env/libero_130.yaml` | LIBERO-130 환경 설정 예시 |
| `examples/embodiment/config/model/openvla_oft.yaml` | 모델 preset와 action/sampling 관련 기본 설정 |
| `rlinf/models/embodiment/openvla_oft/` | OpenVLA-OFT용 RLinf 구현과 공식 구현 경로 |
| `rlinf/models/__init__.py` | 선택한 모델을 만들고 필요 시 LoRA adapter를 붙이는 공통 로더 |
| `rlinf/envs/`, `rlinf/workers/`, `rlinf/runners/` | 환경 실행, 롤아웃 worker, 평가·학습 orchestration |

이는 서버에서 고정한 commit을 조사한 경로 요약이다. 최신 GitHub `main`과는 파일 구성이 달라질 수 있으므로, 실제 재현 때는 [설치 문서](INSTALL.md)의 SHA를 먼저 확인한다.

## 평가 한 번의 흐름

`libero_130`은 개별 난이도 범주가 아니라 Spatial 10개 + Object 10개 + Goal 10개 + Long(`libero_10`) 10개 + LIBERO-90 90개를 합친 130개 task suite다. 따라서 `libero_130` 모델은 이 범주 전체를 대상으로 한 통합 모델이다. 특정 범주만 평가하려면 `env.eval.task_suite_name`을 `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` 또는 `libero_90`으로 지정한다. [공식 suite 표](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)를 참고한다.

1. `run_eval.sh libero libero_10_openvlaoft_eval`을 실행한다.
2. Hydra가 eval YAML과 model/env preset을 합친 후 CLI override를 적용한다. `--cfg job`으로 최종 설정만 확인할 수 있다.
3. RLinf가 로컬 Ray instance를 시작하고 환경 worker와 rollout worker를 배치한다. 현재 스모크에는 GPU 1개를 사용했다.
4. 환경 worker가 LIBERO를 초기화한다. 모델 worker는 OpenVLA-OFT checkpoint를 읽고, SFT일 때만 LoRA adapter를 추가한다.
5. 모델이 관측과 지시를 받아 action chunk를 생성한다. LIBERO가 이를 실행하고 reward·success·episode length를 집계한다.
6. `eval/success_once` 등의 집계값과 최종 설정이 로그에 남는다.

중요한 점은 `rollout.model.model_path`만 바꾼다고 두 모델이 같은 방식으로 로딩되지 않는다는 것이다. SFT와 GRPO의 LoRA 및 sampling 설정 차이는 [모델 문서](MODELS.md)와 [평가 문서](EVALUATION.md)에 명시한다.

## 이 저장소와 upstream의 경계

원본 RLinf는 `upstream/RLinf/`의 별도 Git clone이다. 연구용 문서·manifest·향후 config/script는 이 저장소에 둔다. 이 방식은 upstream 수정 여부와 연구 결과를 분리하고, pinned commit을 재설치하기 쉽게 한다. 로컬 Mac clone에는 GPU venv·checkpoints·서버 로그가 따라오지 않는다. 서버에서 연구 저장소를 갱신한 후에만 이 문서의 상대경로를 원격에서 사용할 수 있다.

공식 참고: [설치](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html), [LIBERO 평가](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/libero.html), [OpenVLA-OFT](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/openvla_oft.html).
