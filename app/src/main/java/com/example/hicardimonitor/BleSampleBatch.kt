package com.example.hicardimonitor

/**
 * BLE 프레임 파싱 — offset counts + 필터 후 mV + .mat용 raw ADC.
 * rPeakIndices: 이 배치의 ECG 샘플 중 tag 0xFF(센서 온칩 R-peak)로 표시된 위치.
 */
data class BleSampleBatch(
  val rawCounts: FloatArray,
  val filtered: FloatArray,
  val rawAdc: FloatArray,
  val temperatureC: Float?,
  val rPeakIndices: IntArray = IntArray(0),
  /** 직전 프레임 이후 패킷 유실이 감지됨 — 플롯은 이 배치 시작에서 선을 잇지 않는다. */
  val gapBefore: Boolean = false,
  /** 전극 미접촉(lead-off) 추정 상태 — true 면 추론·R-peak·BPM 을 돌리지 않는다. */
  val leadOff: Boolean = false,
) {
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (javaClass != other?.javaClass) return false
    other as BleSampleBatch
    return rawCounts.contentEquals(other.rawCounts) &&
      filtered.contentEquals(other.filtered) &&
      rawAdc.contentEquals(other.rawAdc) &&
      temperatureC == other.temperatureC &&
      rPeakIndices.contentEquals(other.rPeakIndices) &&
      gapBefore == other.gapBefore &&
      leadOff == other.leadOff
  }

  override fun hashCode(): Int {
    var result = rawCounts.contentHashCode()
    result = 31 * result + filtered.contentHashCode()
    result = 31 * result + rawAdc.contentHashCode()
    result = 31 * result + (temperatureC?.hashCode() ?: 0)
    result = 31 * result + rPeakIndices.contentHashCode()
    result = 31 * result + gapBefore.hashCode()
    result = 31 * result + leadOff.hashCode()
    return result
  }
}
