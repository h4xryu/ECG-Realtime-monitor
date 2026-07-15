package com.example.hicardimonitor.data

data class RecordSampleBlock(
  val filtered: FloatArray,
  val rawAdc: FloatArray,
  val perSampleFlags: Array<FloatArray>,
  val rPeakIndices: IntArray,
  val dataLossIndices: IntArray,
  val leadOffIndices: IntArray,
)

data class DiagnosisRecord(
  val id: String,
  val startedAtMs: Long,
  val endedAtMs: Long,
  val fs: Int,
  val bpm: Int,
  val summary: String,
  val blocks: List<RecordSampleBlock>,
  val isShortTerm: Boolean,
  val isPersonalization: Boolean = false,
  val reconnectEvents: List<Long> = emptyList(),
) {
  fun totalSamples(): Int = blocks.sumOf { it.rawAdc.size }

  fun mergedRawAdc(): FloatArray {
    val n = totalSamples()
    val out = FloatArray(n)
    var off = 0
    for (b in blocks) {
      b.rawAdc.copyInto(out, off)
      off += b.rawAdc.size
    }
    return out
  }

  fun mergedFiltered(): FloatArray {
    val n = totalSamples()
    val out = FloatArray(n)
    var off = 0
    for (b in blocks) {
      b.filtered.copyInto(out, off)
      off += b.filtered.size
    }
    return out
  }
}
