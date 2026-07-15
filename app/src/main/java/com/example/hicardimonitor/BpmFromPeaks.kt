package com.example.hicardimonitor

import kotlin.math.abs

/**
 * 피크 기반 BPM (PyQt pulse threshold 방식 단순화)
 */
object BpmFromPeaks {

  data class Result(val bpm: Int?, val peakCount: Int)

  /**
   * @param threshold 피크로 인정할 최소 진폭 (mV)
   * @param refractorySamples 피크 간 최소 간격
   */
  fun estimate(
    waveform: FloatArray,
    fs: Float,
    threshold: Float = 0.35f,
    refractorySamples: Int = (fs * 0.25f).toInt(),
  ): Result {
    if (waveform.size < 3) return Result(null, 0)

    val peaks = mutableListOf<Int>()
    var lastPeak = -refractorySamples
    for (i in 1 until waveform.size - 1) {
      val v = waveform[i]
      if (v > threshold && v >= waveform[i - 1] && v >= waveform[i + 1]) {
        if (i - lastPeak >= refractorySamples) {
          peaks.add(i)
          lastPeak = i
        }
      }
    }
    if (peaks.size < 2) return Result(null, peaks.size)

    val intervals = mutableListOf<Float>()
    for (i in 1 until peaks.size) {
      intervals.add((peaks[i] - peaks[i - 1]) / fs)
    }
    val avg = intervals.average().toFloat()
    if (avg <= 0f) return Result(null, peaks.size)
    val bpm = (60f / avg).toInt().coerceIn(20, 250)
    return Result(bpm, peaks.size)
  }
}
