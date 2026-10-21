package com.example.hicardimonitor.session

import android.os.Handler
import android.os.Looper
import com.example.hicardimonitor.ArrhythmiaResult

/**
 * Buffers live ECG samples, the lead-off ratio, and the latest beat classification between
 * periodic flushes, so the BLE batch path (handleBleBatch, ~22 calls/s) only ever does a cheap
 * array append — the actual network send happens on [onFlush], off that hot path.
 *
 * "Signal quality" here is a simple proxy, not a true SNR estimate: the fraction of samples in
 * the flush window that weren't flagged lead-off.
 */
class LiveStreamAggregator(
  private val flushIntervalMs: Long = 2500L,
) {
  data class LivePayload(
    val signal: FloatArray,
    val leadOff: Boolean,
    val signalQuality: Float,
    val bpm: Int?,
    val latestBeatProbs: FloatArray?,
  )

  var onFlush: ((LivePayload) -> Unit)? = null

  private val mainHandler = Handler(Looper.getMainLooper())
  private val lock = Any()

  private var buffer = ArrayList<Float>(4096)
  private var goodSamples = 0
  private var totalSamples = 0
  private var lastLeadOff = false
  private var lastBpm: Int? = null
  private var latestBeatProbs: FloatArray? = null
  private var running = false

  fun start() {
    if (running) return
    running = true
    scheduleNext()
  }

  fun stop() {
    running = false
    synchronized(lock) {
      buffer = ArrayList(4096)
      goodSamples = 0
      totalSamples = 0
      latestBeatProbs = null
    }
  }

  /** Call from the BLE batch path for every batch while a live session is active. */
  fun onBatch(filtered: FloatArray, leadOff: Boolean) {
    synchronized(lock) {
      for (v in filtered) buffer.add(v)
      totalSamples += filtered.size
      if (!leadOff) goodSamples += filtered.size
      lastLeadOff = leadOff
    }
  }

  fun onBeatResult(result: ArrhythmiaResult) {
    synchronized(lock) { latestBeatProbs = result.probabilities.copyOf() }
  }

  fun onBpm(bpm: Int) {
    synchronized(lock) { lastBpm = bpm }
  }

  private fun scheduleNext() {
    mainHandler.postDelayed({
      if (!running) return@postDelayed
      flush()
      scheduleNext()
    }, flushIntervalMs)
  }

  private fun flush() {
    val payload: LivePayload?
    synchronized(lock) {
      if (buffer.isEmpty()) {
        payload = null
      } else {
        val quality = if (totalSamples > 0) goodSamples.toFloat() / totalSamples else 1f
        payload = LivePayload(
          signal = buffer.toFloatArray(),
          leadOff = lastLeadOff,
          signalQuality = quality,
          bpm = lastBpm,
          latestBeatProbs = latestBeatProbs,
        )
        buffer = ArrayList(4096)
        goodSamples = 0
        totalSamples = 0
      }
    }
    if (payload != null) onFlush?.invoke(payload)
  }
}
