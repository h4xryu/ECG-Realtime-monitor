package com.example.hicardimonitor

/**
 * 그래프 축/표시 설정 — PyQt Graph Settings 대응
 */
data class GraphSettings(
  var xWindowSec: Float = ModelConfig.SWEEP_WINDOW_SEC,
  var yMin: Float = -50f,
  var yMax: Float = 50f,
  /** 표시 y축 자동 스케일 — 최근 파형 진폭에 맞춰 창을 조정(레퍼런스: ecgplot.py 는 ylim 고정 없이 auto). */
  var autoScaleY: Boolean = true,
  var sampleRateHz: Float = ModelConfig.FS_HZ,
  var showGrid: Boolean = true,
  var showChannelEcg: Boolean = true,
  var showRPeak: Boolean = false,
  /** 좌측 1 mV 캘리브레이션 마커 표시. */
  var showCalMarker: Boolean = false,
  /** 우하단 mm/mV · mm/s 스케일 표기. */
  var showScaleReadout: Boolean = false,
  /** 좌하단 배터리 잔량 표시. */
  var showBattery: Boolean = false,
  /**
   * 임상 캘리브레이션 표시 모드 — 화면 dpi 기준 고정 스케일(mmPerMv·mmPerSec).
   * 켜면 autoScaleY/yMin/yMax/xWindowSec 대신 mm 단위 그리드·게인·속도로 렌더링한다.
   */
  var calibrated: Boolean = false,
  /** 세로 게인 (mm/mV) — 임상 표준 20 또는 10. */
  var mmPerMv: Float = 20f,
  /** 가로 속도 (mm/s) — 임상 표준 25 또는 50. */
  var mmPerSec: Float = 25f,
  /** 표시 전용 스무딩(EcgDisplaySmoother) 적용 — 공식앱 수준 매끈한 파형. R-peak·기록·모델 무관. */
  var smoothDisplay: Boolean = true,
)
