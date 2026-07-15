package com.example.hicardimonitor

import com.example.hicardimonitor.data.DiagnosisRecord
import com.example.hicardimonitor.data.RecordSampleBlock
import com.example.hicardimonitor.data.RecordingExporter
import java.io.File
import org.junit.Test

/**
 * .mat/.npz 파일을 생성해 스크립트(파이썬 scipy/numpy)로 검증할 수 있게 남긴다.
 * 출력 위치: 시스템 프로퍼티 recordingExportDir (없으면 build/recording-export-test)
 */
class RecordingExporterTest {

  private fun sampleRecord(): DiagnosisRecord {
    val n = 2500 // 10초 @ 250Hz
    val raw = FloatArray(n) { i -> 16384f + (200f * kotlin.math.sin(i / 250.0 * 2 * Math.PI)).toFloat() }
    val filtered = FloatArray(n) { i -> raw[i] - 16384f }
    return DiagnosisRecord(
      id = "test1234",
      startedAtMs = 1752541200000L,
      endedAtMs = 1752541210000L,
      fs = 250,
      bpm = 72,
      summary = "Normal",
      blocks = listOf(
        RecordSampleBlock(
          filtered = filtered,
          rawAdc = raw,
          perSampleFlags = Array(n) { FloatArray(0) },
          rPeakIndices = intArrayOf(125, 375, 625, 875),
          dataLossIndices = intArrayOf(1000, 1001),
          leadOffIndices = intArrayOf(),
        ),
      ),
      isShortTerm = true,
    )
  }

  @Test
  fun exportMatAndNpz() {
    val outDir = File(System.getProperty("recordingExportDir") ?: "build/recording-export-test")
    outDir.mkdirs()
    val record = sampleRecord()

    RecordingExporter.export(
      record,
      RecordingExporter.Options(includeAdc = true, includeMv = true, format = RecordingExporter.Format.MAT),
      File(outDir, "test.mat"),
    )
    RecordingExporter.export(
      record,
      RecordingExporter.Options(includeAdc = true, includeMv = true, format = RecordingExporter.Format.NPZ),
      File(outDir, "test.npz"),
    )
    RecordingExporter.export(
      record,
      RecordingExporter.Options(includeAdc = false, includeMv = true, format = RecordingExporter.Format.NPZ),
      File(outDir, "test_mv_only.npz"),
    )

    assert(File(outDir, "test.mat").length() > 0)
    assert(File(outDir, "test.npz").length() > 0)
  }
}
