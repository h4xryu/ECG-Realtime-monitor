package com.example.hicardimonitor

import kotlin.math.PI
import kotlin.math.cos

/**
 * HiCardi BLE 실시간 필터. raw ADC counts (adc - 0x4000) 입력 → 필터 counts 출력.
 *
 * 파이프라인: 2단 EMA 베이스라인 제거(~0.1Hz HP) → 50Hz notch → LP(~28Hz).
 * PyQt 레퍼런스는 베이스라인에 위상선행 항(2*bl1 - bl2)을 썼으나, QRS 뒤에 깊은
 * 언더슈트(오버슈트)를 유발해 파형이 스파이크처럼 보였다. 여기서는 2단 EMA 결과(bl2)를
 * 그대로 빼는 정규 2차 HP로 바꿔 오버슈트를 제거한다(형태 보존).
 */
class OnlineFilter(fs: Float = ModelConfig.FS_HZ) {

  private val hpAlpha = 0.9975
  private var bl1 = 0.0
  private var bl2 = 0.0

  private val r = 0.98
  private val b0: Double
  private val b1: Double
  private val b2: Double
  private val a1: Double
  private val a2: Double
  private var nx1 = 0.0
  private var nx2 = 0.0
  private var ny1 = 0.0
  private var ny2 = 0.0

  private var lpY = 0.0
  private val lpAlpha = 0.5

  init {
    val w0 = 2.0 * PI * 50.0 / fs
    val c = cos(w0)
    b0 = 1.0
    b1 = -2.0 * c
    b2 = 1.0
    a1 = -2.0 * r * c
    a2 = r * r
  }

  fun reset() {
    bl1 = 0.0
    bl2 = 0.0
    nx1 = 0.0
    nx2 = 0.0
    ny1 = 0.0
    ny2 = 0.0
    lpY = 0.0
  }

  fun processOne(x: Double): Double {
    bl1 = hpAlpha * bl1 + (1.0 - hpAlpha) * x
    bl2 = hpAlpha * bl2 + (1.0 - hpAlpha) * bl1
    val baseline = bl2
    var y = x - baseline
    y = notch50(y)
    lpY = lpAlpha * y + (1.0 - lpAlpha) * lpY
    return lpY
  }

  private fun notch50(x: Double): Double {
    val y = b0 * x + b1 * nx1 + b2 * nx2 - a1 * ny1 - a2 * ny2
    nx2 = nx1
    nx1 = x
    ny2 = ny1
    ny1 = y
    return y
  }
}
