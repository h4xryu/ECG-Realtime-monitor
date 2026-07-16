package com.example.hicardimonitor

/**
 * HiCardi 실시간 모니터링 전용 설정.
 */
object ModelConfig {

  const val PTE_FILE = "model.pte"

  /** CPC 개인화 인코더 (cpc_dsconv, export_to_pte.py). 입력 [1,1,500] → 임베딩 [1,256] */
  const val CPC_PTE_FILE = "cpc_model.pte"

  /** ExecuTorch beat 입력 길이 (export 스펙과 동기화) */
  const val WINDOW_SIZE = 501

  /** CPC 2초 윈도우 @ 250 Hz (export seg_len=500) */
  const val CPC_WINDOW_SIZE = 500

  const val FS_HZ = 250f

  /** 1분 개인화 레코딩 샘플 수 = CALIB_SECONDS(60) × FS(250). DEPLOYMENT_ONDEVICE_SPEC §2. */
  const val CPC_RECORD_SAMPLES = 15_000

  /** 1분 버퍼를 타일링하는 patient-embedding 윈도우 hop (2s 창 50% overlap → 250 = 1s stride) */
  const val CPC_EMBED_HOP = 250

  /**
   * 캘리브레이션 슬라이딩 윈도우 개수 = floor((L−N)/S)+1
   *   = floor((15000−500)/250)+1 = 59.  DEPLOYMENT_ONDEVICE_SPEC §2 CALIB_NUM_WINDOWS.
   *
   * 주의: 이것은 "beat 수"가 아니라 60초를 2초 창·1초 stride 로 슬라이딩한 **윈도우 수**다.
   * 주 CPC 경로(CpcRunner.encodePatient)는 이 상수를 쓰지 않고 신호 전체를 hop 으로 타일링해
   * 자동으로 59개를 만든다. 이 상수는 레거시 model.pte([1,NUM_CALIB,1,500]) 경로 전용.
   */
  const val NUM_CALIB = 59

  /** patient embedding 차원 — cpc_ecg pooled_embedding 출력 (context_dim=256) */
  const val PATIENT_EMBEDDING_DIM = 256

  /** Sweep 실시간 플롯 기본 창 (초) */
  const val SWEEP_WINDOW_SEC = 2f

  /** Sweep 이전 주기와의 시각적 갭 — 지우개 바(erase bar). 공식앱처럼 큰 격자 ~1.5칸(0.3s @25mm/s). */
  const val SWEEP_GAP_SEC = 0.3f

  /** .mat dECG ADC 기준점 */
  const val ADC_ZERO = 8192

  val CLASS_NAMES = HiCardiLabels.BEAT_CLASSES

  enum class OutputMode {
    SOFTMAX,
    SIGMOID,
  }

  val OUTPUT_MODE = OutputMode.SIGMOID

  fun signalDurationMs(): Float = WINDOW_SIZE / FS_HZ * 1000f
}
