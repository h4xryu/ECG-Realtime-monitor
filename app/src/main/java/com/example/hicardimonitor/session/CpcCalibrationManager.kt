package com.example.hicardimonitor.session

import com.example.hicardimonitor.ModelConfig
import com.example.hicardimonitor.SignalUtils
import com.example.hicardimonitor.data.DiagnosisRecord
import com.example.hicardimonitor.data.RecordSampleBlock
import kotlin.math.min

/**
 * 1분(15,000샘플 @ 250Hz) CPC 캘리브레이션 버퍼.
 * 완료 후 16개 2초 윈도우(50% overlap 슬라이딩)를 추출해 추론 입력으로 사용.
 */
class CpcCalibrationManager {

  enum class State { IDLE, RECORDING, COMPLETE }

  // bleExecutor(feed)에서 쓰고 메인(onDestroy 정리)·inferenceExecutor 에서 읽는다.
  @Volatile
  var state: State = State.IDLE
    private set

  private val buffer = FloatArray(ModelConfig.CPC_RECORD_SAMPLES)
  private val rawBuffer = FloatArray(ModelConfig.CPC_RECORD_SAMPLES)
  private var writePos = 0
  private var startMs = 0L

  var calibrationWindows: Array<FloatArray>? = null
    private set

  var progress: Float = 0f
    private set

  /** 지금까지 녹음된 필터 신호 (표시·strip 용). */
  fun recordedSignal(): FloatArray = buffer.copyOfRange(0, writePos.coerceAtMost(buffer.size))

  /**
   * 모델(CPC 인코더) 입력용 raw ECG. 학습 전처리와 동일하게 인코더에서 윈도우별 z-score 한다.
   * 표시용 필터(OnlineFilter)를 거치지 않은 원신호.
   */
  fun recordedRawSignal(): FloatArray = rawBuffer.copyOfRange(0, writePos.coerceAtMost(rawBuffer.size))

  var onProgress: ((remainingSec: Int, progress: Float) -> Unit)? = null
  var onComplete: (() -> Unit)? = null

  fun start() {
    writePos = 0
    progress = 0f
    calibrationWindows = null
    state = State.RECORDING
    startMs = System.currentTimeMillis()
  }

  fun cancel() {
    state = State.IDLE
    writePos = 0
    progress = 0f
    calibrationWindows = null
  }

  fun feed(filteredSamples: FloatArray, rawSamples: FloatArray): Boolean {
    if (state != State.RECORDING) return false
    for (i in filteredSamples.indices) {
      if (writePos >= buffer.size) break
      buffer[writePos] = filteredSamples[i]
      rawBuffer[writePos] = rawSamples.getOrElse(i) { filteredSamples[i] }
      writePos++
    }
    progress = writePos.toFloat() / buffer.size
    val elapsedSec = ((System.currentTimeMillis() - startMs) / 1000).toInt()
    val totalSec = (ModelConfig.CPC_RECORD_SAMPLES / ModelConfig.FS_HZ).toInt()
    val remaining = (totalSec - elapsedSec).coerceAtLeast(0)
    onProgress?.invoke(remaining, progress)
    if (writePos >= buffer.size) {
      finish()
      return true
    }
    return false
  }

  private fun finish() {
    // 모델 입력은 raw ECG (밴드패스 없이 인코더/러너에서 윈도우별 z-score).
    calibrationWindows = extractCalibrationWindows(rawBuffer)
    state = State.COMPLETE
    onComplete?.invoke()
  }

  fun buildRecord(id: String, summary: String = "개인화 측정"): DiagnosisRecord {
    val n = writePos.coerceAtMost(buffer.size)
    val flags = Array(n) { FloatArray(HiCardiLabelCount.NUM_CLASSES) }
    return DiagnosisRecord(
      id = id,
      startedAtMs = startMs,
      endedAtMs = System.currentTimeMillis(),
      fs = ModelConfig.FS_HZ.toInt(),
      bpm = 0,
      summary = summary,
      blocks = listOf(
        RecordSampleBlock(
          filtered = buffer.copyOfRange(0, n),
          rawAdc = rawBuffer.copyOfRange(0, n),
          perSampleFlags = flags,
          rPeakIndices = intArrayOf(),
          dataLossIndices = intArrayOf(),
          leadOffIndices = intArrayOf(),
        ),
      ),
      isShortTerm = false,
      isPersonalization = true,
    )
  }

  companion object {
    /**
     * 레거시 model.pte 조건화용 고정 개수(NUM_CALIB=59) 캘리브레이션 윈도우.
     * 2초 창(500) · 1초 hop(250) 로 60초(15000샘플)를 슬라이딩 → 59창(SSOT §6.1).
     * 주 CPC 경로(CpcRunner.encodePatient)는 이 함수를 쓰지 않고 신호 전체를 타일링한다.
     */
    fun extractCalibrationWindows(signal: FloatArray): Array<FloatArray> {
      val win = ModelConfig.CPC_WINDOW_SIZE
      val hop = win / 2
      val out = Array(ModelConfig.NUM_CALIB) { FloatArray(win) }
      if (signal.size < win) {
        for (i in out.indices) {
          out[i] = SignalUtils.resampleToLen(signal, win)
        }
        return out
      }
      var start = 0
      for (i in 0 until ModelConfig.NUM_CALIB) {
        val end = min(start + win, signal.size)
        val slice = signal.copyOfRange(start, end)
        out[i] = if (slice.size == win) slice else SignalUtils.resampleToLen(slice, win)
        start += hop
        if (start + win > signal.size) start = (signal.size - win).coerceAtLeast(0)
      }
      return out
    }
  }
}
