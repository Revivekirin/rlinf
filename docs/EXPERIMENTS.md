# 연구 단계와 기록 규칙

이 문서는 계획과 완료 상태를 구분한다. 대규모 학습·전체 벤치마크는 아직 실행하지 않았다.

| 단계 | 질문 | 현재 상태 | 다음 산출물 |
| --- | --- | --- | --- |
| 0. 설치·checkpoint 감사 | 환경과 모델이 실제 로딩되는가? | Original LIBERO와 LIBERO-130 2모델 스모크 완료 | dependency 충돌 정리, 설치 재현성 강화 |
| 1. nominal 평가 | 공개 성능과 권장 sampling이 재현되는가? | 단일 episode만 완료 | episode별 metadata와 task/trial coverage |
| 2. LIBERO-PLUS OOD | 학습 없이 perturbation에 얼마나 취약한가? | 미설치·미평가 | 별도 venv, fork revision, 변형별 zero-shot 결과 |
| 3. action 분석 | 동일 observation에서 SFT→GRPO action이 어떻게 바뀌는가? | 미구현 | paired observation/action, finite 검사, 분포·성공 연결 |
| 4. representation 분석 | 어느 layer 변화가 action·성공과 연관되는가? | 미구현 | layer feature와 episode ID 연결; CKA 외 지표 포함 |
| 5. optimizer ablation | PPO/GRPO/RLOO류 설정 차이는? | 미실행 | 자원 예산·baseline·동일 평가 프로토콜 확정 후 학습 |

## 공통 episode 식별자

향후 episode별 데이터는 적어도 아래 필드를 가진다. 스모크 [JSON](../manifests/libero_130_smoke.json)의 `task_id`/`trial_id`는 아직 null이다. 없는 값을 임의로 채우지 않는다.

~~~json
{
  "run_id": "libero130-grpo-seed0-example",
  "checkpoint_repo": "RLinf/RLinf-OpenVLAOFT-LIBERO-130",
  "checkpoint_revision": "473954241c8148bc9a1bf9ed547a74523e4a7d41",
  "rlinf_commit": "e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d",
  "suite": "libero_130",
  "task_id": null,
  "trial_id": null,
  "initial_state_id": null,
  "seed": 0,
  "sampling": {"do_sample": true, "temperature": 1.6, "top_p": 1.0, "top_k": -1},
  "horizon": 520,
  "action_chunks": 8,
  "unnorm_key": "libero_130_no_noops_trajall",
  "success": null,
  "episode_length": null,
  "observation_ref": null,
  "action_ref": null,
  "feature_ref": null
}
~~~

예시는 **스키마 형태**일 뿐 실제 실행 결과가 아니다. `run_id`와 task/trial/init-state의 조합을 관측·action·layer feature·성공 결과에 공통으로 써서 paired 분석이 가능하게 한다. representation은 CKA 하나로 결론내리지 않고, layer별 변화와 출력 action 및 task success를 함께 본다.

## 자원 경계

서버는 공유 GPU 1개(A5000 24 GiB)다. 공개 RLinf의 대규모 GRPO recipe를 이 머신에서 그대로 실행할 수 있다고 가정하지 않는다. 새 GPU 작업 전 `nvidia-smi`와 Ray 점유를 확인하고, 타인의 Ray를 종료하지 않는다. 전체 평가와 학습은 자원·episode budget·중단/재시작 절차를 정한 뒤 별도로 시작한다.
