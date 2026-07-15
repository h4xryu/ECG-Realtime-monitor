package com.example.hicardimonitor.session

import kotlin.math.sqrt

/**
 * 개인화 품질 지표 — validate_patient_conditioning.py 의 온디바이스 단일-환자 대응판.
 *
 * 원 스크립트는 여러 환자 임베딩으로 collapse / kNN 분리를 측정하지만, 기기에서는 한 사람의
 * 1분 윈도우들만 있으므로 "자신의 beat 를 얼마나 일관되게 맞추는가"로 재정의한다.
 *
 * - selfConsistency : 앞/뒤 30초 임베딩의 코사인 — 리듬이 안정적으로 재현되는가
 * - beatCoherence   : 각 2초 윈도우 ↔ pooled 임베딩 평균 코사인 — 자기 beat 정합도
 * - concentration   : 단위벡터 평균의 크기(R) ∈ [0,1] — 윈도우 방향 집중도
 * - effectiveRank   : 윈도우 임베딩 특이값 엔트로피 — 붕괴(collapse) 감지
 * - fitScore        : 0..100, coherence + consistency 가중 (지표화용 단일 값)
 */
data class PersonalizationQuality(
  val windowCount: Int,
  val selfConsistency: Float,
  val beatCoherence: Float,
  val concentration: Float,
  val effectiveRank: Float,
  val fitScore: Int,
) {
  val fitLabelKey: String
    get() = when {
      fitScore >= 80 -> "good"
      fitScore >= 55 -> "fair"
      else -> "weak"
    }

  companion object {
    /** 코사인(0.70..1.0)을 0..1 품질로 스트레치 — 동일 환자 윈도우는 높은 코사인에 몰림. */
    private fun stretch(c: Float): Float = ((c - 0.70f) / 0.30f).coerceIn(0f, 1f)

    fun from(encoding: PatientEncoding): PersonalizationQuality {
      val w = encoding.perWindow
      val pooled = encoding.pooled
      if (w.size < 2) {
        return PersonalizationQuality(w.size, 0f, 0f, 0f, 1f, 0)
      }
      val dim = w[0].size

      // beat coherence: 윈도우 ↔ pooled 평균 코사인
      var coh = 0f
      for (v in w) coh += CpcRunner.cosine(v, pooled)
      coh /= w.size

      // concentration R = ||mean(unit)||
      val mean = FloatArray(dim)
      for (v in w) for (d in 0 until dim) mean[d] += v[d]
      for (d in 0 until dim) mean[d] /= w.size
      var r = 0f
      for (d in 0 until dim) r += mean[d] * mean[d]
      r = sqrt(r).coerceIn(0f, 1f)

      // self-consistency: 앞/뒤 절반 평균 임베딩 코사인
      val mid = w.size / 2
      val a = meanVec(w.subList(0, mid), dim)
      val b = meanVec(w.subList(mid, w.size), dim)
      val self = CpcRunner.cosine(a, b)

      val eff = CpcRunner.effectiveRank(w)

      val fit = (100f * (0.5f * stretch(self) + 0.5f * stretch(coh))).toInt().coerceIn(0, 100)
      return PersonalizationQuality(
        windowCount = w.size,
        selfConsistency = self,
        beatCoherence = coh,
        concentration = r,
        effectiveRank = eff,
        fitScore = fit,
      )
    }

    private fun meanVec(list: List<FloatArray>, dim: Int): FloatArray {
      val out = FloatArray(dim)
      for (v in list) for (d in 0 until dim) out[d] += v[d]
      if (list.isNotEmpty()) for (d in 0 until dim) out[d] /= list.size
      return out
    }
  }
}
