package com.example.hicardimonitor

/**
 * 신호 리샘플 — Python resample_to_len 대응
 */
object SignalUtils {

  fun resampleToLen(ts: FloatArray, outLen: Int): FloatArray {
    if (ts.size == outLen) return ts.copyOf()
    if (outLen <= 0) return FloatArray(0)
    if (ts.size < 2) {
      val out = FloatArray(outLen)
      val v = if (ts.isNotEmpty()) ts[0] else 0f
      for (i in 0 until outLen) out[i] = v
      return out
    }
    // outLen==1 이면 아래 보간식의 분모(outLen-1)가 0 → NaN. 중앙 샘플로 대표.
    if (outLen == 1) return floatArrayOf(ts[ts.size / 2])
    val out = FloatArray(outLen)
    for (i in 0 until outLen) {
      val pos = i.toFloat() / (outLen - 1) * (ts.size - 1)
      val lo = pos.toInt().coerceIn(0, ts.size - 1)
      val hi = (lo + 1).coerceAtMost(ts.size - 1)
      val t = pos - lo
      out[i] = ts[lo] * (1 - t) + ts[hi] * t
    }
    return out
  }

  fun zscore(seg: FloatArray): FloatArray {
    var mean = 0f
    for (v in seg) mean += v
    mean /= seg.size
    var varSum = 0f
    for (v in seg) {
      val d = v - mean
      varSum += d * d
    }
    val std = kotlin.math.sqrt(varSum / seg.size).coerceAtLeast(1e-6f)
    return FloatArray(seg.size) { i -> (seg[i] - mean) / std }
  }
}
