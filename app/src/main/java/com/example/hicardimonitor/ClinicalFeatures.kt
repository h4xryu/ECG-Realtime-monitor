package com.example.hicardimonitor

import kotlin.math.sqrt

/**
 * HiCardi 학습 clinical 32-D 피처의 온디바이스 재구현.
 *
 * SSOT: PECG-HiCardi/src/data/clinical_features.py `compute_clinical_features`
 * (HiCardi 학습 레이아웃 — MIT-BIH premature_index 레이아웃 아님).
 *
 * 레이아웃:
 *   [0:10]  리듬  — pre/post RR, local-mean 대비 비율, past-only local(10)/global(20) RR 통계
 *   [10:13] 형태  — peak-to-peak, RMS, 중앙 20% 에너지
 *   [13:32] 파형  — 균등 간격 19 샘플
 *
 * 파이썬과의 정합: population std(ddof=0)+1e-8, excess kurtosis, linspace(...).astype(int).
 * 목표: max|feat_device - feat_py| < 1e-5.
 */
object ClinicalFeatures {

  const val DIM = 32
  private const val EPS = 1e-8

  /** [mean, std, skew, excess_kurtosis] — past-only RR 창(초). */
  private fun rrStats(rr: DoubleArray): DoubleArray {
    val n = rr.size
    if (n == 0) return doubleArrayOf(0.0, 0.0, 0.0, 0.0)
    if (n == 1) return doubleArrayOf(rr[0], 0.0, 0.0, 0.0)
    var mean = 0.0
    for (x in rr) mean += x
    mean /= n
    var m2 = 0.0
    var m3 = 0.0
    var m4 = 0.0
    for (x in rr) {
      val d = x - mean
      val d2 = d * d
      m2 += d2
      m3 += d2 * d
      m4 += d2 * d2
    }
    val std = sqrt(m2 / n) + EPS // ddof=0 + 1e-8 (numpy 기본)
    m3 /= n
    m4 /= n
    val skew = m3 / (std * std * std)
    val kurt = m4 / (std * std * std * std) - 3.0
    return doubleArrayOf(mean, std, skew, kurt)
  }

  /**
   * 비트 [i] 의 32-D 피처.
   * @param sampleIndices 시간 순 R-peak 절대 샘플 인덱스(전체 히스토리; local/global 은 과거만 사용).
   * @param i             대상 비트 위치. post_rr 을 위해 i < size-1 (다음 비트 known) 권장.
   * @param waveform      비트 [i] 의 501 샘플 raw mV crop.
   * @param fs            샘플레이트(Hz).
   */
  fun compute(sampleIndices: LongArray, i: Int, waveform: FloatArray, fs: Float): FloatArray {
    val n = sampleIndices.size
    val feat = FloatArray(DIM)

    val rrAll = if (n > 1) {
      DoubleArray(n - 1) { (sampleIndices[it + 1] - sampleIndices[it]).toDouble() / fs }
    } else {
      DoubleArray(0)
    }

    val preRr = if (i > 0) rrAll[i - 1] else 0.0
    val postRr = if (i < n - 1) rrAll[i] else 0.0

    val local = rrAll.copyOfRange(maxOf(0, i - 10), i.coerceIn(0, rrAll.size))
    val global = rrAll.copyOfRange(maxOf(0, i - 20), i.coerceIn(0, rrAll.size))
    val ls = rrStats(local) // [mean,std,skew,kurt]
    val gs = rrStats(global)
    val lm = ls[0]

    feat[0] = preRr.toFloat()
    feat[1] = postRr.toFloat()
    feat[2] = (preRr / (lm + EPS)).toFloat()
    feat[3] = (postRr / (lm + EPS)).toFloat()
    feat[4] = lm.toFloat()
    feat[5] = ls[1].toFloat()
    feat[6] = ls[2].toFloat()
    feat[7] = ls[3].toFloat()
    feat[8] = gs[0].toFloat()
    feat[9] = gs[1].toFloat()

    val seg = waveform
    val L = seg.size
    if (L > 0) {
      var mn = seg[0]
      var mx = seg[0]
      var sumSq = 0.0
      for (v in seg) {
        if (v < mn) mn = v
        if (v > mx) mx = v
        sumSq += v.toDouble() * v.toDouble()
      }
      feat[10] = mx - mn
      feat[11] = sqrt(sumSq / L).toFloat()

      val c0 = (L * 0.4).toInt()
      val c1 = (L * 0.6).toInt()
      if (c1 > c0) {
        var e = 0.0
        for (k in c0 until c1) e += seg[k].toDouble() * seg[k].toDouble()
        feat[12] = (e / (c1 - c0)).toFloat()
      }

      // linspace(0, L-1, 19).astype(int) — 양수라 truncation == floor.
      val last = (L - 1).coerceAtLeast(0).toDouble()
      for (k in 0 until 19) {
        val pos = (last * k / 18.0).toInt().coerceIn(0, L - 1)
        feat[13 + k] = seg[pos]
      }
    }

    return feat
  }
}
