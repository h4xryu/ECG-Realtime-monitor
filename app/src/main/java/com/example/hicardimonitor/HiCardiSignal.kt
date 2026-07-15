package com.example.hicardimonitor

/**
 * HiCardi BLE ADC / 물리 단위 변환.
 *
 * - 필터: OnlineFilter(counts) → /32 mV
 * - .mat dECG: [ModelConfig.ADC_ZERO] 기준 raw ADC 저장
 */
object HiCardiSignal {

  /** 실시간 센서 기본 오프셋 (0x4000 = 16384) */
  const val BLE_ADC_OFFSET = 0x4000

  /**
   * (offset counts) / COUNTS_PER_MV → mV.
   * SSOT(DEPLOYMENT_ONDEVICE_SPEC §2): ADC_SCALE=1000, mV=(adc-offset)/1000 — 학습·모델·
   * clinical feature 와 동일 스케일. (구 표시용 /32 는 실제 mV 의 ~31배로 과대표시였음.)
   */
  const val COUNTS_PER_MV = 1000f

  const val ECG_TAG_F0 = 0xF0
  const val ECG_TAG_FF = 0xFF
  const val TEMP_TAG_F8 = 0xF8

  fun adcFromBytes(hi: Int, lo: Int): Int =
    ((hi and 0xFF) shl 8) or (lo and 0xFF)

  fun offsetCounts(adc: Int): Float = (adc - BLE_ADC_OFFSET).toFloat()

  /** ADC 레일(0/0xFFFF) 근처로 튀는 손상값 판정 여유. */
  private const val RAIL_MARGIN = 8

  /**
   * 손상/간섭 프레임에서 나오는 비생리학적 이상치의 상한(오프셋 counts).
   * ±8192 counts ≈ ±256 mV — 실제 ECG(±수 mV)와 베이스라인 드리프트를 크게 상회하므로,
   * 이 값을 넘거나 ADC 레일 근처면 명백한 손상으로 보고 폐기한다(레일-투-레일 세로 스파이크 제거).
   */
  const val MAX_ABS_OFFSET_COUNTS = 8192f

  /** tag 는 ECG(0xF0/0xFF)지만 hi/lo 가 손상돼 나온 이상치 triplet 인지. */
  fun isPlausibleEcg(adc: Int): Boolean {
    if (adc <= RAIL_MARGIN || adc >= 0xFFFF - RAIL_MARGIN) return false
    return kotlin.math.abs(offsetCounts(adc)) <= MAX_ABS_OFFSET_COUNTS
  }

  fun countsToMv(counts: Float): Float = counts / COUNTS_PER_MV

  /** 0xF8 triplet: 16bit 값을 0.01°C 단위로 해석 */
  fun temperatureFromF8(hi: Int, lo: Int): Float =
    adcFromBytes(hi, lo) / 100f

  fun isPlausibleBodyTemp(c: Float): Boolean = c in 20f..45f
}
