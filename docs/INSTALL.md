# 서버 설치와 검증 기록

이 문서는 2026-11-08 서버에서 실제 실행·확인한 상태를 기록한다. 재구축 명령은 같은 upstream commit에서 사용하는 절차이지만, 패키지 저장소가 바뀌면 동일한 바이너리 집합을 보장하지 않는다. 실제 패키지 inventory는 [설치 목록](../manifests/installed-libero-packages.txt)에 있다.

## 머신과 경로

| 항목 | 확인값 |
| --- | --- |
| 연구 루트 | `/nfs4/jhkim/repos/rlinf` |
| RLinf | `upstream/RLinf`, commit `e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d` |
| Original LIBERO venv | `upstream/RLinf/.venv-rlvr-libero-cu121` |
| Python / Ray | 3.11.14 / 2.60.0 |
| GPU / driver | RTX A5000 24 GiB / 535.230.02 |
| Torch / CUDA build | **2.5.1+cu124 / 12.4** (실제 `torch` 출력) |

venv 이름의 `cu121`은 이전 설치 시도에서 정한 문자열일 뿐이다. `nvidia-smi`의 `CUDA Version: 12.2`는 드라이버가 표시한 값이다. cu124 Torch의 작은 CUDA 계산은 이 서버에서 성공했지만, 모든 CUDA 연산의 호환성을 보장하지 않는다.

## 재구축 절차

연구 저장소를 서버에 복제한 뒤 다음을 수행한다. 기존 `vla-rlvr`와 그 conda 환경은 건드리지 않는다.

~~~bash
export RESEARCH_ROOT=/nfs4/jhkim/repos/rlinf
cd "$RESEARCH_ROOT"
git clone https://github.com/RLinf/RLinf.git upstream/RLinf
git -C upstream/RLinf checkout e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d
git -C upstream/RLinf rev-parse HEAD
~~~

이미 `upstream/RLinf`가 있으면 다시 clone하지 않고 SHA와 작업 트리부터 확인한다. 비밀번호 없는 sudo 권한이 없어 공식 installer의 `--no-root`를 썼다. 시스템 라이브러리(`libOSMesa`, `libGLX`, `libGLEW`, `libEGL`, compiler, `cmake`, `ffmpeg` 등)는 별도로 존재를 확인했다.

~~~bash
cd "$RESEARCH_ROOT/upstream/RLinf"
mkdir -p "$HOME/rlinf-assets/tmp"
export TMPDIR="$HOME/rlinf-assets/tmp"
export UV_LINK_MODE=copy
unset UV_TORCH_BACKEND

bash requirements/install.sh \
  --no-root \
  --torch 2.5.1 \
  embodied \
  --model openvla-oft \
  --env maniskill_libero \
  --venv .venv-rlvr-libero-cu121
~~~

설치는 exit 0으로 끝났다. `UV_LINK_MODE=copy`는 캐시와 NFS 사이 hardlink 경고를 줄이는 설정이다. installer·bundle 명칭은 [RLinf 설치 문서](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)와 서버에서 checkout한 `requirements/install.sh`에서 확인했다.

## 실제 검증 결과

| 검사 | 결과 |
| --- | --- |
| `uv pip check` | **실패: 8개 dependency metadata 충돌** |
| `import rlinf`, `libero.libero`, `mani_skill` | 통과 |
| 평가 entrypoint 및 OpenVLA-OFT 모듈 import | 통과; TensorFlow 경고 동반 |
| LIBERO assets 3개 디렉터리 | 존재 확인 |
| `torch.cuda.is_available()` | `True` |
| CUDA 산술 | A5000에서 `sum(arange(16)^2) == 1240.0` |
| SFT/GRPO 실제 평가 | 각 1개 에피소드 완료; [평가 기록](EVALUATION.md) |

확인용 명령:

~~~bash
cd "$RESEARCH_ROOT/upstream/RLinf"
VENV_PY="$PWD/.venv-rlvr-libero-cu121/bin/python"
"$VENV_PY" - <<'PY'
import importlib, torch
for name in ("rlinf", "libero.libero", "mani_skill", "evaluations.eval_embodied_agent"):
    print(name, importlib.import_module(name).__file__)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print((torch.arange(16, device="cuda", dtype=torch.float32) ** 2).sum().item())
PY
"$PWD/.venv-rlvr-libero-cu121/bin/uv" pip check --python "$VENV_PY"
~~~

`pip check`의 8개 항목은 tyro/typeguard, swanlab/wrapt, torch/sympy, TensorFlow/protobuf, pandas/numpy, openvla-oft의 torch/torchvision/torchaudio 버전 pin이다. 설치된 `protobuf=6.33.6`은 TensorFlow 2.15의 `<5` 요구와 맞지 않는다. import 때 `MessageFactory.GetPrototype` 메시지가 나왔지만 두 평가는 끝났다. dependency 검사를 통과했다고 기록해서는 안 되며 전체 환경 업그레이드로 무작정 해결하지 않는다.

처음의 `.venv-rlvr-libero` 시도는 Torch 2.11.0+cu130 및 LIBERO 미설치로 끝난 **실패 환경**이다. 연구 실행에 사용한 환경은 `.venv-rlvr-libero-cu121`이다. 실패 환경을 삭제했는지는 확인하지 않았다.

## LIBERO와 다음 설치

Original LIBERO는 `rlinf-libero==0.1.3`과 assets가 venv에 있다. `LIBERO_TYPE=standard` 평가가 실제로 돌아갔다. LIBERO-PLUS는 아직 별도 설치·렌더링 검증을 하지 않았다. 공식 `--env liberoplus` 절차를 **다른 `--venv` 경로**에서 수행하고 fork revision·assets·카메라 렌더링을 기록한 뒤 OOD 평가에 들어간다. 이 문서는 LIBERO-PLUS가 준비되었다고 주장하지 않는다.

참고: [RLinf LIBERO 평가 문서](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/libero.html). 공유 서버에서 다른 사용자의 Ray 프로세스에 영향을 줄 수 있으므로 무심코 `ray stop`을 실행하지 않는다.
