package com.example.hicardimonitor

enum class MeasurementMode {
  /** Home > 모니터 — 연속 실시간 */
  CONTINUOUS_MONITOR,
  /** Home > 레코딩 — 중지할 때까지 연속 기록하는 전용 세션 */
  RECORDING,
  /** Home > 개인화 — 1분 CPC 캘리브레이션 */
  CPC_CALIBRATION,
  /** 센서 미연결 또는 일시정지 */
  IDLE,
}
