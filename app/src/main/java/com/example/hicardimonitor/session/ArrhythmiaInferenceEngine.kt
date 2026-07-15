package com.example.hicardimonitor.session

import com.example.hicardimonitor.ArrhythmiaClassifierRunner
import com.example.hicardimonitor.ArrhythmiaConfig
import com.example.hicardimonitor.ArrhythmiaResult
import com.example.hicardimonitor.ClinicalFeatures

/**
 * 실시간 비트 중심(beat-centered) 부정맥 추론 오케스트레이터.
 *
 * 센서 온칩 R-peak(0xFF, [SensorRPeakTracker] 로 병합·스냅됨)마다 501 샘플 raw mV 를 크롭하고
 * clinical 32-D 를 계산해 [ArrhythmiaClassifierRunner] 로 6-class 확률을 낸다.
 *
 * post_rr(다음 R) 이 필요하므로 정확히 **1비트 지연**으로 확정한다(spec §5.2 허용).
 * 즉 비트 k 는 비트 k+1 이 확인되고 r_k+250 샘플이 도착한 뒤 emit 된다.
 *
 * 스레드: onBatch 는 BLE 워커 스레드에서 호출(추론도 여기서). onResult 콜백도 동일 스레드에서
 * 불리므로 UI 반영은 호출측에서 메인 스레드로 post 한다.
 */
class ArrhythmiaInferenceEngine(
  private val runner: ArrhythmiaClassifierRunner,
  private val fs: Float,
) {

  /** 개인화 patient_vec 공급자. null 이면 0-vector(개인화 미적용). */
  var patientVecProvider: () -> FloatArray? = { null }

  /** 추론 on/off. false 면 상태만 흘려보내 backlog 를 막는다. */
  var enabledProvider: () -> Boolean = { true }

  /** 현재 심박(BPM). 서맥 rule 판정에 사용. 0/미상이면 서맥 해제. */
  var bpmProvider: () -> Int = { 0 }

  /** (결과, 비트 절대 샘플 인덱스). BLE 워커 스레드에서 호출됨. */
  var onResult: ((ArrhythmiaResult, Long) -> Unit)? = null

  // ── raw mV circular buffer (절대 인덱싱) ──────────────────────────────
  private val ringCap = (fs * 6f).toInt().coerceAtLeast(ArrhythmiaConfig.BEAT_LEN * 2)
  private val ring = FloatArray(ringCap)
  private var ringBaseAbs = 0L // ring[0] 에 해당하는 절대 인덱스(oldest)
  private var ringCount = 0

  private var absCount = 0L // 지금까지 투입된 총 샘플 수 = 다음 샘플의 절대 인덱스

  // ── 확정 R-peak 절대 인덱스(시간순) + emit 프론티어 ────────────────────
  private val beatsAbs = ArrayList<Long>(64)
  private var emitPos = 0

  // 비트별 확률 EMA(지터 제거). null 이면 아직 시작 전.
  private var smoothed: FloatArray? = null

  // 서맥 rule 상태(히스테리시스): BPM < BRADYCARDIA_BPM_MAX 진입, ≥ BRADYCARDIA_CLEAR_BPM 해제.
  private var bradyActive = false

  fun reset() {
    ringBaseAbs = 0L
    ringCount = 0
    absCount = 0L
    beatsAbs.clear()
    emitPos = 0
    smoothed = null
    bradyActive = false
  }

  /**
   * 한 배치 투입.
   * @param mvSamples          이 배치의 raw mV 샘플(= rawCounts / ADC_SCALE_MV), filtered 와 동일 길이.
   * @param confirmedBeatsRel  [SensorRPeakTracker.process] 가 반환한 배치 시작 기준 확정 비트 상대 인덱스.
   */
  fun onBatch(mvSamples: FloatArray, confirmedBeatsRel: IntArray) {
    val batchStart = absCount
    for (v in mvSamples) push(v)
    absCount += mvSamples.size

    for (rel in confirmedBeatsRel) {
      val abs = batchStart + rel
      // 트래커는 refractory 로 단조 증가하는 비트를 보장 → 중복/역행만 방어.
      if (beatsAbs.isEmpty() || abs > beatsAbs[beatsAbs.size - 1]) beatsAbs.add(abs)
    }

    flush()
  }

  private fun flush() {
    val enabled = enabledProvider() && runner.isLoaded
    // 다음 비트(post_rr)가 있어야 확정.
    while (emitPos < beatsAbs.size - 1) {
      val r = beatsAbs[emitPos]
      // r+250 샘플이 아직 안 왔으면 대기.
      if (absCount < r + ArrhythmiaConfig.BEAT_POST + 1) break

      if (enabled) {
        val crop = crop(r)
        if (crop != null) {
          val clinical = ClinicalFeatures.compute(beatsAbs.toLongArray(), emitPos, crop, fs)
          val pv = patientVecProvider()
          val res = runner.infer(crop, pv, clinical)
          if (res != null) onResult?.invoke(postProcess(res), r)
        }
      }
      emitPos++
    }
    trim()
  }

  /**
   * 후처리: (1) 모델 6-로짓 → 표시 5클래스 매핑(서맥 로짓 폐기), (2) EMA 스무딩,
   * (3) 서맥 rule 판정. 서맥은 정의 자체가 심박수 기준(HR < 60)이라 모델(AUPRC 낮음)
   * 대신 BPM rule 로 판정해 [ArrhythmiaResult.bradycardia] 로 별도 전달한다.
   */
  private fun postProcess(res: ArrhythmiaResult): ArrhythmiaResult {
    // 모델 로짓에서 서맥([BRADY_LOGIT_INDEX])을 드롭 → CLASS_NAMES 순서의 5-vector.
    val raw = res.probabilities
    val probs = FloatArray(ArrhythmiaConfig.NUM_CLASSES)
    var j = 0
    for (i in raw.indices) {
      if (i == ArrhythmiaConfig.BRADY_LOGIT_INDEX) continue
      if (j >= probs.size) break
      probs[j++] = raw[i]
    }

    val prev = smoothed
    val sm = if (prev == null || prev.size != probs.size) {
      probs.copyOf()
    } else {
      FloatArray(probs.size) { SMOOTH_ALPHA * probs[it] + (1f - SMOOTH_ALPHA) * prev[it] }
    }
    smoothed = sm

    // 서맥 rule: BPM < 60 진입, ≥ 62 해제(히스테리시스). BPM 미상이면 해제.
    // rule 결과는 스무딩 없이 즉시 반영 — 판정 기준이 이미 중앙값 RR 기반이라 안정적.
    val bpm = bpmProvider()
    bradyActive = when {
      bpm <= 0 -> false
      bpm < ArrhythmiaConfig.BRADYCARDIA_BPM_MAX -> true
      bpm >= ArrhythmiaConfig.BRADYCARDIA_CLEAR_BPM -> false
      else -> bradyActive
    }

    return ArrhythmiaResult.from(sm, res.latencyMs, bradyActive)
  }

  /** [r-250, r+250] 501 샘플 crop. ring 밖으로 밀렸으면 null. */
  private fun crop(r: Long): FloatArray? {
    val startAbs = r - ArrhythmiaConfig.BEAT_PRE
    if (startAbs < ringBaseAbs) return null
    if (r + ArrhythmiaConfig.BEAT_POST >= ringBaseAbs + ringCount) return null
    val out = FloatArray(ArrhythmiaConfig.BEAT_LEN)
    for (k in 0 until ArrhythmiaConfig.BEAT_LEN) {
      out[k] = ring[((startAbs + k) % ringCap).toInt()]
    }
    return out
  }

  private fun push(v: Float) {
    if (ringCount < ringCap) {
      ring[((ringBaseAbs + ringCount) % ringCap).toInt()] = v
      ringCount++
    } else {
      ring[(ringBaseAbs % ringCap).toInt()] = v
      ringBaseAbs++
    }
  }

  /** beatsAbs 를 유계로 유지(emit 프론티어 뒤로 과거 RR 20개 이상 보존). */
  private fun trim() {
    while (beatsAbs.size > MAX_BEATS && emitPos > KEEP_BEHIND) {
      beatsAbs.removeAt(0)
      emitPos--
    }
  }

  companion object {
    private const val MAX_BEATS = 64
    private const val KEEP_BEHIND = 22 // global RR 은 과거 20개 → 여유 포함 보존
    /** 확률 EMA 계수(작을수록 더 부드럽고 느리게 반응). */
    private const val SMOOTH_ALPHA = 0.3f
  }
}
