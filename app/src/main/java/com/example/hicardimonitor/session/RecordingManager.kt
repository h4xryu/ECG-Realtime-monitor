package com.example.hicardimonitor.session

import com.example.hicardimonitor.ModelConfig
import com.example.hicardimonitor.data.DiagnosisRecord
import com.example.hicardimonitor.data.RecordSampleBlock

/**
 * 레코딩 — 시작하면 중지할 때까지 연속 기록. 중지 시점까지의 데이터로 레코드 생성.
 * 장시간 기록을 위해 per-sample 멀티라벨 flags 는 저장하지 않는다(메모리 상한).
 */
class RecordingManager {

  enum class State { IDLE, RECORDING, COMPLETE }

  @Volatile
  var state: State = State.IDLE
    private set

  /** (state, elapsedSec) — 시작/배치/중지 시 호출 */
  var stateListener: ((State, elapsedSec: Int) -> Unit)? = null

  private val filtered = ArrayList<Float>(ModelConfig.FS_HZ.toInt() * 60)
  private val rawAdc = ArrayList<Float>(ModelConfig.FS_HZ.toInt() * 60)
  private val rPeaks = ArrayList<Int>()
  private val dataLost = ArrayList<Int>()
  private val leadOff = ArrayList<Int>()
  private var startMs = 0L
  private var endMs = 0L
  private var lastBpm = 0
  private var lastPred = ""

  private val elapsedSec: Int
    get() = if (startMs == 0L) 0 else ((System.currentTimeMillis() - startMs) / 1000).toInt()

  fun start() {
    clear()
    state = State.RECORDING
    startMs = System.currentTimeMillis()
    stateListener?.invoke(state, 0)
  }

  /** 여기까지 기록하고 종료 — 데이터는 유지되어 buildRecord 로 넘어간다. */
  fun stop() {
    if (state != State.RECORDING) return
    endMs = System.currentTimeMillis()
    state = State.COMPLETE
    stateListener?.invoke(state, elapsedSec)
  }

  fun cancel() {
    state = State.IDLE
    clear()
    stateListener?.invoke(state, 0)
  }

  /** buildRecord 이후 버퍼 해제(리스너 호출 없음). */
  fun reset() {
    state = State.IDLE
    clear()
  }

  fun feed(
    filteredSamples: FloatArray,
    rawSamples: FloatArray,
    rPeakRelIndices: IntArray,
    isDataLoss: Boolean,
    isLeadOff: Boolean,
    bpm: Int,
    prediction: String,
  ) {
    if (state != State.RECORDING) return
    stateListener?.invoke(state, elapsedSec)
    lastBpm = bpm
    lastPred = prediction
    val base = filtered.size
    for (i in filteredSamples.indices) {
      filtered.add(filteredSamples[i])
      rawAdc.add(rawSamples.getOrElse(i) { filteredSamples[i] })
      if (isDataLoss) dataLost.add(base + i)
      if (isLeadOff) leadOff.add(base + i)
    }
    // 센서 온칩 R-peak(스냅 완료, 배치 시작 기준 상대 인덱스).
    // 음수 rel(직전 배치로 스냅)은 base+rel 이 버퍼 범위 안이면 유효 — 범위 밖은 스킵.
    val size = filtered.size
    for (rel in rPeakRelIndices) {
      val idx = base + rel
      if (idx in 0 until size) rPeaks.add(idx)
    }
  }

  fun hasSamples(): Boolean = filtered.isNotEmpty()

  fun buildRecord(id: String): DiagnosisRecord {
    val n = filtered.size
    val emptyFlags = FloatArray(0)
    return DiagnosisRecord(
      id = id,
      startedAtMs = startMs,
      endedAtMs = if (endMs != 0L) endMs else System.currentTimeMillis(),
      fs = ModelConfig.FS_HZ.toInt(),
      bpm = lastBpm,
      summary = lastPred,
      blocks = listOf(
        RecordSampleBlock(
          filtered = filtered.toFloatArray(),
          rawAdc = rawAdc.toFloatArray(),
          perSampleFlags = Array(n) { emptyFlags },
          rPeakIndices = rPeaks.toIntArray(),
          dataLossIndices = dataLost.toIntArray(),
          leadOffIndices = leadOff.toIntArray(),
        ),
      ),
      isShortTerm = true,
    )
  }

  private fun clear() {
    filtered.clear()
    rawAdc.clear()
    rPeaks.clear()
    dataLost.clear()
    leadOff.clear()
    startMs = 0L
    endMs = 0L
    lastBpm = 0
    lastPred = ""
  }
}

/** flags 배열 길이 상수 (StripRecordBuffer와 분리) */
object HiCardiLabelCount {
  const val NUM_CLASSES = 27
}
