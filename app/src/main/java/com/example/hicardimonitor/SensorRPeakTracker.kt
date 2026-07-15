package com.example.hicardimonitor

/**
 * 센서 온칩 R-peak(BLE tag 0xFF) 기반 심박 추적 + 위치 재스냅.
 *
 * 센서 0xFF fiducial 은 개수·간격(RR)은 정확하지만 위치가 실제 QRS 보다 ~340ms 늦게(직전
 * QRS 를 뒤늦게) 보고된다(실측: rec.npz 에서 플래그가 QRS 로부터 +360ms 뒤, 스냅 전 ±60ms
 * 창 적중률 0%). 따라서 0xFF 는 "박동이 있었다"는 트리거로만 쓰고, 실제 R 위치는 플래그
 * 기준 backward 창([200,500]ms)에서 제곱미분 에너지 최대점 → 원신호 극점으로 재스냅한다.
 * (rec.npz 검증: 개수 64 보존, QRS 적중 100%@60ms, 중앙값 +2ms, 충돌 0.)
 *
 * backward 스냅이라 미래 샘플이 불필요해 스트리밍에서 즉시 확정된다. 반환 상대 인덱스는
 * 플래그보다 앞선(과거) 위치라 대개 음수다 — 소비자는 절대 인덱스로 환산해 처리한다.
 *
 * BPM 은 최근 구간(기본 8초) RR 중앙값 — T파 중복검출로 인한 2배 오류 제거.
 */
class SensorRPeakTracker(private val fs: Float) {

  private val refractory = (fs * 0.20f).toInt().coerceAtLeast(1)
  private val bpmWindowSamples = (fs * 8f).toLong().coerceAtLeast(1L)

  // 재스냅 backward 창 [200ms, 500ms] + 미분/정제 여유. 링은 이보다 넉넉히 유지.
  private val snapBackLo = (fs * 0.20f).toInt().coerceAtLeast(1)   // 50 @250Hz
  private val snapBackHi = (fs * 0.50f).toInt().coerceAtLeast(2)   // 125 @250Hz
  private val snapRefine = (fs * 0.02f).toInt().coerceAtLeast(1)   // ±20ms
  private val ringKeep = ((fs * 2f).toInt()).coerceAtLeast(snapBackHi + snapRefine + 8)

  private val ring = ArrayDeque<Float>()
  private var ringBase = 0L
  private var absCount = 0L
  private var lastFlagAbs = Long.MIN_VALUE / 4
  private val beats = ArrayDeque<Long>()

  fun reset() {
    ring.clear()
    ringBase = 0L
    absCount = 0L
    lastFlagAbs = Long.MIN_VALUE / 4
    beats.clear()
  }

  /**
   * 한 배치를 투입한다.
   * @param filtered 이 배치의 필터된 파형(mV)
   * @param flagIndices tag 0xFF로 표시된 배치 내 로컬 인덱스(원시 버스트, 미병합)
   * @return 이번 배치 시작 기준 상대 인덱스로 확정된 비트(재스냅 완료). backward 스냅이라 음수(과거)일 수 있다.
   */
  fun process(filtered: FloatArray, flagIndices: IntArray): IntArray {
    val batchStart = absCount
    for (v in filtered) ring.addLast(v)
    absCount += filtered.size
    while (ring.size > ringKeep) {
      ring.removeFirst()
      ringBase++
    }

    val confirmed = ArrayList<Int>(flagIndices.size)
    for (idx in flagIndices) {
      val flagAbs = batchStart + idx
      // refractory 는 신뢰도 높은 플래그 간격에 적용(버스트 병합).
      if (flagAbs - lastFlagAbs < refractory) continue
      lastFlagAbs = flagAbs
      val snapAbs = snapBackwardToQrs(flagAbs)
      // 재스냅 결과가 역행하면(잡음) 버린다 — 단조 증가 보장.
      if (beats.isNotEmpty() && snapAbs <= beats.last()) continue
      beats.addLast(snapAbs)
      confirmed.add((snapAbs - batchStart).toInt())
    }

    val cutoff = absCount - bpmWindowSamples
    while (beats.isNotEmpty() && beats.first() < cutoff) beats.removeFirst()
    return confirmed.toIntArray()
  }

  /** 최근 구간 RR 중앙값 기반 BPM. 비트가 부족하면 null. */
  fun bpm(): Int? {
    if (beats.size < 2) return null
    val ordered = beats.toLongArray()
    val rr = ArrayList<Float>(ordered.size - 1)
    for (i in 1 until ordered.size) {
      val sec = (ordered[i] - ordered[i - 1]) / fs
      if (sec in 0.3f..2.0f) rr.add(sec)
    }
    if (rr.isEmpty()) return null
    rr.sort()
    val median = if (rr.size % 2 == 0) {
      (rr[rr.size / 2 - 1] + rr[rr.size / 2]) / 2f
    } else {
      rr[rr.size / 2]
    }
    if (median <= 0f) return null
    return (60f / median).toInt().coerceIn(20, 220)
  }

  private fun ringValue(abs: Long): Float? {
    val i = (abs - ringBase).toInt()
    if (i < 0 || i >= ring.size) return null
    return ring[i]
  }

  /**
   * 0xFF 플래그(QRS 보다 늦게 보고됨)를 실제 R 위치로 재스냅.
   * flagAbs 기준 backward 창 [flagAbs-snapBackHi, flagAbs-snapBackLo] 에서
   *   1) 제곱미분 에너지 최대점(= QRS 최급경사, 베이스라인 드리프트 무관)을 찾고
   *   2) 그 ±snapRefine 에서 원신호 |진폭-베이스라인| 극점으로 정제.
   * 히스토리가 부족하면(스트림 초반) 플래그 위치를 그대로 반환.
   */
  private fun snapBackwardToQrs(flagAbs: Long): Long {
    val loC = flagAbs - snapBackHi
    val hiC = flagAbs - snapBackLo
    if (loC < ringBase + 4 || hiC >= ringBase + ring.size) return flagAbs

    // 창 평균 = 베이스라인 근사.
    var mean = 0f
    var cnt = 0
    var a = loC
    while (a <= hiC) {
      ringValue(a)?.let { mean += it; cnt++ }
      a++
    }
    if (cnt == 0) return flagAbs
    mean /= cnt

    // 1) 제곱미분 에너지 최대 후보.
    var bestC = hiC
    var bestE = -1f
    var c = loC
    while (c <= hiC) {
      val e = derivEnergy(c)
      if (e > bestE) { bestE = e; bestC = c }
      c++
    }

    // 2) 원신호 극점으로 정제.
    var bestR = bestC
    var bestAmp = -1f
    var r = bestC - snapRefine
    val rHi = bestC + snapRefine
    while (r <= rHi) {
      val v = ringValue(r)
      if (v != null) {
        val amp = kotlin.math.abs(v - mean)
        if (amp > bestAmp) { bestAmp = amp; bestR = r }
      }
      r++
    }
    return bestR
  }

  /** ±3 샘플(≈30ms) 평활한 중앙차분 제곱 에너지. */
  private fun derivEnergy(abs: Long): Float {
    var sum = 0f
    var cnt = 0
    for (k in -3..3) {
      val prev = ringValue(abs + k - 1)
      val next = ringValue(abs + k + 1)
      if (prev != null && next != null) {
        val d = (next - prev) * 0.5f
        sum += d * d
        cnt++
      }
    }
    return if (cnt > 0) sum / cnt else 0f
  }
}
