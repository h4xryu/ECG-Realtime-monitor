package com.example.hicardimonitor.data

import com.example.hicardimonitor.HiCardiSignal
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * 레코딩 내보내기 — MATLAB v5 .mat 또는 NumPy .npz.
 *
 * 컬럼(모두 float64, 샘플 정렬):
 *   ecg_adc   [n]  raw 16bit ADC (옵션)
 *   ecg_mv    [n]  (adc - 16384) / 32 mV (옵션)
 *   rpeak     [n]  센서 온칩 R-peak 플래그 0/1
 *   data_lost [n]  0/1
 *   lead_off  [n]  0/1
 *   t_sec     [n]  시작 기준 경과 초 (i / fs)
 *   ts_ms     [n]  epoch ms 일시 (start_ms + i * 1000/fs)
 *   fs        [1]  샘플링 주파수
 *   start_ms  [1]  시작 epoch ms
 *
 * 장시간 기록(수십 분)도 힙에 통째로 올리지 않도록 값 단위 스트리밍으로 쓴다.
 */
object RecordingExporter {

  enum class Format(val ext: String) { MAT("mat"), NPZ("npz") }

  data class Options(
    val includeAdc: Boolean,
    val includeMv: Boolean,
    val format: Format,
  )

  fun fileNameFor(record: DiagnosisRecord, format: Format): String {
    val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(record.startedAtMs))
    return "hicardi_rec_${stamp}.${format.ext}"
  }

  fun export(record: DiagnosisRecord, options: Options, outFile: File) {
    val raw = record.mergedRawAdc()
    val n = raw.size
    val rpeak = BooleanArray(n)
    val dataLost = BooleanArray(n)
    val leadOff = BooleanArray(n)
    var offset = 0
    for (block in record.blocks) {
      block.rPeakIndices.forEach { idx ->
        val g = offset + idx
        if (g in 0 until n) rpeak[g] = true
      }
      block.dataLossIndices.forEach { idx ->
        val g = offset + idx
        if (g in 0 until n) dataLost[g] = true
      }
      block.leadOffIndices.forEach { idx ->
        val g = offset + idx
        if (g in 0 until n) leadOff[g] = true
      }
      offset += block.rawAdc.size
    }

    val fs = record.fs.toDouble()
    val msPerSample = 1000.0 / fs
    val startMs = record.startedAtMs.toDouble()

    val columns = ArrayList<Column>()
    if (options.includeAdc) columns.add(Column("ecg_adc", n) { raw[it].toDouble() })
    if (options.includeMv) {
      columns.add(
        Column("ecg_mv", n) {
          (raw[it] - HiCardiSignal.BLE_ADC_OFFSET) / HiCardiSignal.COUNTS_PER_MV.toDouble()
        },
      )
    }
    columns.add(Column("rpeak", n) { if (rpeak[it]) 1.0 else 0.0 })
    columns.add(Column("data_lost", n) { if (dataLost[it]) 1.0 else 0.0 })
    columns.add(Column("lead_off", n) { if (leadOff[it]) 1.0 else 0.0 })
    columns.add(Column("t_sec", n) { it / fs })
    columns.add(Column("ts_ms", n) { startMs + it * msPerSample })
    columns.add(Column("fs", 1) { fs })
    columns.add(Column("start_ms", 1) { startMs })

    when (options.format) {
      Format.MAT -> writeMat(outFile, columns)
      Format.NPZ -> writeNpz(outFile, columns)
    }
  }

  /** 이름 있는 float64 1-D 배열. 값은 인덱스에서 지연 계산(스트리밍). */
  private class Column(val name: String, val n: Int, val value: (Int) -> Double)

  // ── MATLAB v5 ─────────────────────────────────────────────────────────────

  private fun writeMat(file: File, columns: List<Column>) {
    BufferedOutputStream(FileOutputStream(file), BUF_SIZE).use { out ->
      val header = ByteArray(128)
      val title = "MATLAB 5.0 MAT-file, HiCardiMonitor recording"
      title.toByteArray(Charsets.US_ASCII).copyInto(header, 0, 0, minOf(title.length, 116))
      header[124] = 0x00 // version 0x0100 (LE)
      header[125] = 0x01
      header[126] = 'I'.code.toByte()
      header[127] = 'M'.code.toByte()
      out.write(header)

      val chunk = ByteBuffer.allocate(BUF_SIZE).order(ByteOrder.LITTLE_ENDIAN)
      fun flushChunk() {
        out.write(chunk.array(), 0, chunk.position())
        chunk.clear()
      }
      fun putInt(v: Int) {
        if (chunk.remaining() < 4) flushChunk()
        chunk.putInt(v)
      }
      fun putDouble(v: Double) {
        if (chunk.remaining() < 8) flushChunk()
        chunk.putDouble(v)
      }

      for (col in columns) {
        val nameBytes = col.name.toByteArray(Charsets.US_ASCII)
        val namePadded = (nameBytes.size + 7) / 8 * 8
        val dataBytes = col.n * 8
        // miMATRIX payload: array flags(16) + dims(16) + name(8+pad) + data(8+bytes)
        val total = 16 + 16 + (8 + namePadded) + (8 + dataBytes)

        putInt(MI_MATRIX); putInt(total)
        putInt(MI_UINT32); putInt(8)
        putInt(MX_DOUBLE_CLASS); putInt(0)
        putInt(MI_INT32); putInt(8)
        putInt(col.n); putInt(1)
        putInt(MI_INT8); putInt(nameBytes.size)
        for (b in nameBytes) {
          if (chunk.remaining() < 1) flushChunk()
          chunk.put(b)
        }
        repeat(namePadded - nameBytes.size) {
          if (chunk.remaining() < 1) flushChunk()
          chunk.put(0)
        }
        putInt(MI_DOUBLE); putInt(dataBytes)
        for (i in 0 until col.n) putDouble(col.value(i))
      }
      flushChunk()
    }
  }

  private const val MI_INT8 = 1
  private const val MI_INT32 = 5
  private const val MI_UINT32 = 6
  private const val MI_DOUBLE = 9
  private const val MI_MATRIX = 14
  private const val MX_DOUBLE_CLASS = 6
  private const val BUF_SIZE = 1 shl 16

  // ── NumPy .npz ────────────────────────────────────────────────────────────

  private fun writeNpz(file: File, columns: List<Column>) {
    ZipOutputStream(BufferedOutputStream(FileOutputStream(file), BUF_SIZE)).use { zip ->
      for (col in columns) {
        zip.putNextEntry(ZipEntry("${col.name}.npy"))
        writeNpy(zip, col)
        zip.closeEntry()
      }
    }
  }

  /** NPY v1.0 — dtype '<f8', C order, shape (n,). */
  private fun writeNpy(out: OutputStream, col: Column) {
    val dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (${col.n},), }"
    // magic(6) + version(2) + headerLen(2) + header 가 64의 배수, 끝은 '\n'
    val base = 6 + 2 + 2
    val padded = ((base + dict.length + 1 + 63) / 64) * 64 - base
    val header = dict.padEnd(padded - 1, ' ') + "\n"

    out.write(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte()))
    out.write(byteArrayOf(1, 0))
    out.write(byteArrayOf((header.length and 0xFF).toByte(), ((header.length shr 8) and 0xFF).toByte()))
    out.write(header.toByteArray(Charsets.US_ASCII))

    val chunk = ByteBuffer.allocate(BUF_SIZE).order(ByteOrder.LITTLE_ENDIAN)
    for (i in 0 until col.n) {
      if (chunk.remaining() < 8) {
        out.write(chunk.array(), 0, chunk.position())
        chunk.clear()
      }
      chunk.putDouble(col.value(i))
    }
    out.write(chunk.array(), 0, chunk.position())
  }
}
