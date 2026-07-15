# HiCardi Monitor (Android) — 실시간 ECG 모니터 + 벤치마크

## 모듈 구조 (수정할 파일 찾기)

| 파일 | 역할 |
|------|------|
| `ModelConfig.kt` | **가장 먼저 수정** — .pte 이름, WINDOW_SIZE, 클래스, FLOPs 파라미터 |
| `ModelRunner.kt` | ExecuTorch 로드/추론, 전처리 |
| `EcgPlotView.kt` | 실시간 그래프 (X/Y/grid) |
## Demo vs HiCardi

| | Demo (MIT-BIH) | HiCardi BLE |
|--|----------------|-------------|
| 실시간 그래프 | **숨김** | **표시** + 갱신 |
| 스트립 뷰 | 2초/비트, 2초 딜레이, 오버랩 없음 | 세그먼트 도착 즉시 갱신 |
| 세그먼트 | 720 samples @ 360Hz | 동일 |

스트립 뷰: `StripRecordView.kt` — eval_record_plots.py macro strip 대응

Demo 데이터: `assets/demo_record.json` (없으면 synthetic 40 beats)

| `MitBihDemoPlayer.kt` | Demo 순차 재생 |
| `HiCardiBleSource.kt` | BLE 스텁 (미구현, 주석 참고) |
| `FlopEstimator.kt` | FLOPs analytical (benchmark_4class.py) |
| `MemoryProfiler.kt` | Peak RSS / activation 추정 |
| `LatencyProfiler.kt` | 추론 레이턴시, RTF |
| `BenchmarkRunner.kt` | 위 셋 한번에 실행 |
| `BeatEvaluator.kt` | Demo accuracy / confusion |
| `SessionRecorder.kt` | Record → CSV export |
| `MonitorController.kt` | UI 연결 (얇게 유지) |
| `MainActivity.kt` | 진입점만 |

## 사용법

1. `app/src/main/assets/model.pte` 배치 (없으면 DemoClassifier fallback)
2. Run → **Demo Mode** 버튼 → synthetic N/S/V/F 스트림 + 추론
3. **Run Benchmark** → FLOPs / RAM / Latency 패널 확인
4. **Record** → **Save** → CSV 공유

## 4-class 모델 전환

`ModelConfig.kt`에서:
- `PTE_FILE = "model_4class.pte"`
- `WINDOW_SIZE = 720`
- `OUTPUT_MODE = SOFTMAX`
- `CLASS_NAMES = listOf("N","S","V","F")`

## HiCardi BLE

`HiCardiBleSource.kt` 주석의 GATT 골격 참고. 구현 후 `MonitorController.onSamples()` 로 샘플 전달.

## 벤치마크

Python `benchmark_4class.py` 와 FLOPs 공식 동기화: `FlopEstimator.mambaFlopsPerLayer()`

On-device RAM: forward 전후 PSS delta (`MemoryProfiler`)
