# Original LIBERO 평가 기록과 재실행

## 통과한 스모크

2026-11-08 서버의 pinned RLinf로 `libero_130`에서 모델당 1개 환경·1개 에피소드를 평가했다. 공통으로 seed 0, 고정/순서 reset state, horizon 520, action chunk 8, video 저장 해제를 사용했다. [기록 JSON](../manifests/libero_130_smoke.json)에 checkpoint revision과 로그 상대경로가 있다.

| 조건 | `do_sample` | 완료 | `success_once` | 길이 |
| --- | --- | --- | ---: | ---: |
| SFT + LoRA | 설정 false, 실제 true | 예 | 0 | 520 |
| GRPO, 별도 adapter 없음 | 실제 true | exit 0 | 1 | 88 |

SFT 결과는 한 episode의 실패, GRPO 결과는 한 episode의 성공이다. **성능 우열이나 논문 수치 재현을 뜻하지 않는다.** 고정 RLinf commit의 `HuggingfaceWorker.setup_sample_params()`는 eval의 실제 `do_sample`을 `temperature_eval > 0`으로 결정한다. 따라서 SFT CLI에 `do_sample=false`를 주었어도 기본 `temperature_eval=1.6` 때문에 **실제 SFT도 stochastic**이었다. task ID와 trial ID가 터미널 출력에 기록되지 않아 paired trial 비교도 아니다. import 때 TensorFlow/protobuf 경고가 있었으나 평가가 끝났다.

## 서버에서 같은 범위로 재실행

GPU 점유를 먼저 확인하고 공유 Ray 작업을 종료하지 않는다. 아래 명령은 서버의 venv와 checkpoint가 그대로 있을 때 사용한다. `run_eval.sh`가 내부 로그 디렉터리를 만든다.

~~~bash
nvidia-smi
cd /nfs4/jhkim/repos/rlinf/upstream/RLinf
source .venv-rlvr-libero-cu121/bin/activate

LIBERO_TYPE=standard env -u PYTHONPATH -u PYTHONHOME \
bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
  env.eval.task_suite_name=libero_130 \
  env.eval.total_num_envs=1 \
  env.eval.rollout_epoch=1 \
  env.eval.auto_reset=false \
  env.eval.ignore_terminations=false \
  env.eval.reset_gripper_open=true \
  env.eval.video_cfg.save_video=false \
  rollout.model.model_path=/nfs4/jhkim/repos/rlinf/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora \
  rollout.model.unnorm_key=libero_130_no_noops_trajall \
  rollout.model.is_lora=true \
  rollout.model.lora_path=/nfs4/jhkim/repos/rlinf/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora/lora_adapter \
  rollout.sampling_params.do_sample=false \
  runner.logger.experiment_name=libero_130_sft_smoke
~~~

GRPO는 다음과 같이 모델 및 sampling 부분을 바꾼다.

~~~bash
LIBERO_TYPE=standard env -u PYTHONPATH -u PYTHONHOME \
bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
  env.eval.task_suite_name=libero_130 \
  env.eval.total_num_envs=1 \
  env.eval.rollout_epoch=1 \
  env.eval.auto_reset=false \
  env.eval.ignore_terminations=false \
  env.eval.reset_gripper_open=true \
  env.eval.video_cfg.save_video=false \
  rollout.model.model_path=/nfs4/jhkim/repos/rlinf/checkpoints/RLinf-OpenVLAOFT-LIBERO-130 \
  rollout.model.unnorm_key=libero_130_no_noops_trajall \
  rollout.model.is_lora=false \
  rollout.model.lora_path=null \
  rollout.sampling_params.do_sample=true \
  runner.logger.experiment_name=libero_130_grpo_smoke
~~~

`evaluations/libero/libero_10_openvlaoft_eval.yaml`은 여기서 **출발점으로만** 사용했다. CLI에서 suite를 `libero_130`으로 바꾼 최종 Hydra 설정을 `--cfg job`으로 확인했고, 평가 완료 출력에도 `task_suite_name=libero_130`이 남았다. [공식 평가 CLI](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/libero.html)는 `evaluations/run_eval.sh <benchmark> <config>`와 Hydra override를 사용한다.

## 공정한 비교를 위한 다음 단계

공개 모델 카드의 권장 조건은 SFT deterministic, GRPO stochastic이다. 이는 **각 모델의 권장 설정 비교**이지 sampling을 고정한 인과적 SFT→GRPO 효과 비교가 아니다. 후자는 같은 task/trial/init state에서 아래 4조건을 수집해야 한다. 이 commit의 공식 eval 경로에서 deterministic을 원하면 `rollout.sampling_params.temperature_eval=-1`로 지정하고 실제 inference kwargs를 검증해야 한다.

| 모델 | deterministic | stochastic |
| --- | --- | --- |
| SFT + LoRA | 미실행 | 실행됨: 1 episode (CLI `do_sample=false`였음) |
| GRPO | 미실행 | 실행됨: 1 episode |

정식 프로토콜에는 suite, task ID, trial ID, initial-state ID, seed, horizon, action chunk, normalization key, sampling(temperature/top-p/top-k), checkpoint revision, success, episode length를 episode별로 기록한다. `total_num_envs=1`만으로 전체 suite를 평가할 수 없다. 전체 LIBERO-130은 130 task × 50 trial = 6,500 조합으로, 병렬 환경 수와 epoch·step budget을 함께 설계해야 한다. GPU 공유 서버에서 전체 평가는 아직 시작하지 않는다. [공식 coverage 설명](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/libero.html)을 확인한다.

## OOD

Original LIBERO가 준비되었지만 LIBERO-PLUS는 아직 설치되지 않았다. RLinf의 [LIBERO-PLUS 평가 모드](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/libero.html)는 `LIBERO_TYPE=plus` 및 `LIBERO_SUFFIX`를 사용한다. 별도 venv·fork/assets·렌더링 스모크를 완료하고 동일 checkpoint를 학습 없이 평가할 때만 이 모드를 사용한다.
