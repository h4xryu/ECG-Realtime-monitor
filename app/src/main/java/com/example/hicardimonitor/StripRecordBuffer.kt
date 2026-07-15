package com.example.hicardimonitor

/**
 * 스트립 뷰용 데이터 — eval_record_plots.py macro view
 */
data class StripBeat(
  val rPeakIndex: Int,
  val segmentStart: Int = 0,
  val segmentEnd: Int = 0,
  val predId: Int,
  val trueId: Int,
  val predProb: Float,
)

class StripRecordBuffer(
  private val fs: Float = ModelConfig.FS_HZ,
) {
  private val signal = ArrayList<Float>(4096)
  private val beats = ArrayList<StripBeat>()

  /** HiCardi: 30초/행 */
  var secondsPerRow: Float = 30f

  /** 고정 뷰포트 길이(초) — 기본 1분 */
  var visibleWindowSec: Float = 60f

  val sampleCount: Int get() = signal.size
  val beatList: List<StripBeat> get() = beats

  fun clear() {
    signal.clear()
    beats.clear()
  }

  fun appendSegment(segment: FloatArray, beat: StripBeat?) {
    synchronized(this) {
      val base = signal.size
      for (v in segment) signal.add(v)
      if (beat != null) {
        beats.add(
          beat.copy(
            rPeakIndex = base + segment.size / 2,
            segmentStart = base,
            segmentEnd = base + segment.size,
          ),
        )
      }
    }
  }

  fun signalArray(): FloatArray = synchronized(this) { signal.toFloatArray() }

  fun samplesPerRow(): Int = (secondsPerRow * fs).toInt().coerceAtLeast(1)

  fun visibleRowCount(): Int {
    val rows = (visibleWindowSec / secondsPerRow).toInt()
    return rows.coerceAtLeast(1)
  }

  fun dataRowCount(): Int {
    if (signal.isEmpty()) return 0
    val spr = samplesPerRow()
    return (signal.size + spr - 1) / spr
  }

  /** 뷰포트(최소 1분)와 실제 데이터 행 중 큰 값 */
  fun contentRowCount(): Int = maxOf(visibleRowCount(), dataRowCount())

  fun durationSec(): Float = if (signal.isEmpty()) 0f else signal.size / fs

  companion object {
    fun classToId(name: String): Int = HiCardiLabels.classIndex(name).coerceAtLeast(0)
    fun idToName(id: Int): String = HiCardiLabels.BEAT_CLASSES.getOrElse(id) { "Unknown" }
  }
}
