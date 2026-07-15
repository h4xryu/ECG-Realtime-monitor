package com.example.hicardimonitor

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 실제 녹화(rec.npz → CSV fixture)로 backward 재스냅을 검증한다.
 * 센서 0xFF 플래그(위치가 ~340ms 늦음)를 트래커에 스트리밍 투입했을 때,
 * 확정된 R-peak 가 독립 검출한 실제 QRS 에 정렬되는지 확인.
 */
class SensorRPeakTrackerTest {

  private fun readCsv(name: String): IntArray {
    val f = File("src/test/resources/$name")
    return f.readLines().filter { it.isNotBlank() }.map { it.trim().toDouble().toInt() }.toIntArray()
  }

  @Test
  fun resnapAlignsToQrs() {
    val adc = readCsv("rec_adc.csv")
    val flags = readCsv("rec_flags.csv").toHashSet()
    val qrs = readCsv("rec_qrs.csv")
    val fs = 250f

    val tracker = SensorRPeakTracker(fs)
    val snapped = ArrayList<Long>()
    var absBase = 0
    // BLE 프레임 모사: 11 샘플/배치로 스트리밍.
    val batch = 11
    var i = 0
    while (i < adc.size) {
      val end = minOf(i + batch, adc.size)
      val filt = FloatArray(end - i) { (adc[i + it] - 16384).toFloat() }
      val flagRel = ArrayList<Int>()
      for (k in i until end) if (flags.contains(k)) flagRel.add(k - i)
      val confirmed = tracker.process(filt, flagRel.toIntArray())
      for (rel in confirmed) snapped.add((absBase + rel).toLong())
      absBase += (end - i)
      i = end
    }

    // 1) 개수 보존 (플래그 64 ≈ 확정 64, 스트림 초반 1~2개 폴백 허용)
    assertTrue("count=${snapped.size} (expect ~${flags.size})", snapped.size >= flags.size - 2 && snapped.size <= flags.size)

    // 2) 각 확정 비트가 실제 QRS 60ms 이내
    val tol = (0.06f * fs).toInt()
    var within = 0
    val offs = ArrayList<Int>()
    for (s in snapped) {
      var best = Int.MAX_VALUE
      for (q in qrs) {
        val diff = (q - s).toInt()
        if (kotlin.math.abs(diff) < kotlin.math.abs(best)) best = diff
      }
      offs.add(best)
      if (kotlin.math.abs(best) <= tol) within++
    }
    val rate = within.toFloat() / snapped.size
    offs.sort()
    val medOff = offs[offs.size / 2]
    println("resnap: n=${snapped.size} within60ms=${(rate * 100).toInt()}% medianOffset=${medOff}samp (${medOff * 1000 / fs.toInt()}ms)")

    assertTrue("only ${(rate * 100).toInt()}% within 60ms of QRS", rate >= 0.90f)
    assertTrue("median offset ${medOff}samp too large", kotlin.math.abs(medOff) <= (0.03f * fs).toInt())
  }
}
