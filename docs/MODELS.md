# LIBERO-130 모델·LoRA 관계

## 다운로드와 출처

두 checkpoint는 서버의 `/nfs4/jhkim/repos/rlinf/checkpoints/`에 저장됐다. Git에는 넣지 않는다. Hugging Face revision, 원격 파일별 크기·blob ID·제공된 LFS SHA-256은 [`models_130.json`](../manifests/models_130.json)에 있다.

| 역할 | Hugging Face repo | 고정 revision | 원격 크기 | 파일 |
| --- | --- | --- | ---: | ---: |
| SFT 베이스 + LoRA | [RLinf-OpenVLAOFT-LIBERO-130-Base-Lora](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora) | `a4ea91b3897648f3c1395e5221edb40b8f53d1a4` | 14.50 GiB | 26 |
| GRPO 후 모델 | [RLinf-OpenVLAOFT-LIBERO-130](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130) | `473954241c8148bc9a1bf9ed547a74523e4a7d41` | 14.05 GiB | 22 |

`snapshot_download`으로 받고 원격 manifest에 기재된 모든 파일의 존재·크기를 확인했다. LFS SHA-256은 Hugging Face가 제공한 메타데이터를 기록한 값이며 **로컬 가중치 전체를 다시 해시한 검증 결과는 아니다**. SFT에는 4개 base weight shard와 `lora_adapter/adapter_model.safetensors`(약 462 MiB)가 있다. GRPO에는 4개 weight shard가 있고 별도 LoRA adapter 폴더가 없다.

## 어떻게 서로 대응하나

[RLinf 모델 카드](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130)는 GRPO 모델이 LIBERO-130 LoRA-base에서 출발했다고 명시한다. 두 checkpoint의 `dataset_statistics.json`에는 `libero_130_no_noops_trajall` 정규화 key가 있다. 확인한 구성은 7차원 action, 8개 action chunk, 카메라 image 1개, proprio 미사용, bf16, `flash_attention_2`이다. SFT adapter는 LoRA r=32/alpha=16이다.

| RLinf 평가 옵션 | 130 SFT checkpoint: base only 진단 | 130 SFT checkpoint: adapter 추가 실험 | GRPO |
| --- | --- | --- | --- |
| `rollout.model.model_path` | `checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora` | 동일 | `checkpoints/RLinf-OpenVLAOFT-LIBERO-130` |
| `rollout.model.is_lora` | `false` | `true` | `false` |
| `rollout.model.lora_path` | `null` | SFT 폴더 아래 `lora_adapter` | `null` |
| `rollout.model.unnorm_key` | `libero_130_no_noops_trajall` | 동일 | 동일 |
| sampling | deterministic | deterministic | stochastic, temperature 1.6 |

주의: 위 sampling 행은 **모델 카드의 권장 조건**이다. 현재 고정된 RLinf eval worker는 실제 sampling 여부를 `temperature_eval > 0`으로 결정하므로, 앞선 SFT 스모크의 CLI `do_sample=false`는 deterministic 추론을 만들지 못했다. [평가 기록](EVALUATION.md)에 설정값과 실제값을 나눠 기록했다.

RLinf 공통 로더는 `is_lora=true`와 `lora_path`가 설정되면 adapter를 붙인다. 그러나 **adapter가 존재한다는 사실만으로 평가 시 추가 적용이 맞다고 결론낼 수 없다.** 고정 upstream의 `libero_130_grpo_openvlaoft.yaml`은 `Base-Lora` 경로를 사용하고, 상속한 `model/openvla_oft.yaml`의 `is_lora=false`를 바꾸지 않는다. 따라서 base shard에 SFT가 이미 반영됐을 가능성을 검증 중이다. SFT adapter metadata에 남은 작성자 머신의 절대 `base_model_name_or_path`를 이 서버 경로로 사용하지 않는다. 기존 프로젝트의 continuous-action policy loader를 섞지 않고 RLinf native `openvla_oft` 경로를 쓴다.

## 확인 범위와 남은 것

2026-11-09 동일한 저장 관측에서 adapter 활성화 재추론의 56개 action token은 기존 수집값과 전부 일치했고, adapter를 끄면 9개가 달랐다. 즉 adapter는 실제로 작동한다. 하지만 adapter 적용 SFT deterministic 수집은 Goal/Spatial/Long 각 0/50이었다. 같은 Goal task 0, trial 25의 공식 RLinf 평가에서 adapter 적용 시 실패·520 step, base only 시 성공·128 step이었다. trial 0은 두 방식 모두 실패·520 step이었다. **base only가 공식 SFT baseline인지, 첫 5개 trial의 선택 편향이 있는지는 아직 확정하지 않았다.** adapter 적용 0/150을 공개 SFT 성능으로 인용하지 않는다.

LIBERO-90/Goal checkpoint는 현재 다운로드하지 않았다. 90 RL 모델의 공식 repo ID는 [모델 카드](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130)에서 확인할 수 있지만, 다운로드·revision 감사 전에는 이 manifest의 대상에 넣지 않는다.
