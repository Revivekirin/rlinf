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

| RLinf 평가 옵션 | SFT | GRPO |
| --- | --- | --- |
| `rollout.model.model_path` | `checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora` | `checkpoints/RLinf-OpenVLAOFT-LIBERO-130` |
| `rollout.model.is_lora` | `true` | `false` |
| `rollout.model.lora_path` | SFT 폴더 아래 `lora_adapter` | `null` |
| `rollout.model.unnorm_key` | `libero_130_no_noops_trajall` | 동일 |
| 공식 공개 평가 sampling | `do_sample=false` | `do_sample=true`, temperature 1.6, top-p 1.0 |
| 이 서버 실제 로딩 | base shard + adapter가 평가에서 작동 | GRPO shard가 adapter 없이 평가에서 작동 |

주의: 위 `do_sample` 행은 **모델 카드의 권장 조건**이다. 현재 고정된 RLinf eval worker는 실제 sampling 여부를 `temperature_eval > 0`으로 결정하므로, 앞선 SFT 스모크의 CLI `do_sample=false`는 deterministic 추론을 만들지 못했다. [평가 기록](EVALUATION.md)에 설정값과 실제값을 나눠 기록했다.

RLinf 공통 로더는 `is_lora=true`와 `lora_path`가 설정되면 adapter를 붙인다. 따라서 SFT는 **base 디렉터리와 그 안의 adapter를 모두 지정**한다. GRPO에는 adapter를 다시 붙이지 않는다. SFT의 adapter metadata에 남아 있는 작성자 머신의 절대 `base_model_name_or_path`는 이 서버의 실제 경로가 아니므로 그 경로로 로딩하지 않는다. 기존 프로젝트의 continuous-action policy loader를 섞지 않고 RLinf native `openvla_oft` 경로를 쓴다.

## 확인 범위와 남은 것

config·weight 목록·정규화 key·LoRA 적용 경로는 조사했고, 두 모델의 실제 환경 롤아웃도 완료했다. **동일한 고정 observation에서 생성된 두 action tensor를 별도로 덤프해 finite 여부를 비교하는 분석은 아직 하지 않았다.** 스모크 성공만으로 action 분포나 representation이 어떻게 달라졌다고 결론내리지 않는다.

LIBERO-90/Goal checkpoint는 현재 다운로드하지 않았다. 90 RL 모델의 공식 repo ID는 [모델 카드](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130)에서 확인할 수 있지만, 다운로드·revision 감사 전에는 이 manifest의 대상에 넣지 않는다.
