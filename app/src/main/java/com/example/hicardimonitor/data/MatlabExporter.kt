package com.example.hicardimonitor.data

import com.example.hicardimonitor.HiCardiLabels
import com.example.hicardimonitor.ModelConfig
import com.example.hicardimonitor.session.HiCardiLabelCount
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * MATLAB v5 호환 .mat (7개 변수).
 * v7.3(HDF5) 대신 v5 double 배열 — scipy.io.loadmat 호환 목표.
 */
object MatlabExporter {

  fun export(record: DiagnosisRecord, outFile: File) {
    val raw = record.mergedRawAdc()
    val n = raw.size
    val numClasses = HiCardiLabelCount.NUM_CLASSES

    val dEcg = DoubleArray(n) { i -> raw[i].toDouble() }
    val finalFlag = Array(n) { DoubleArray(numClasses) }
    val rpk = DoubleArray(n)
    val dataLost = DoubleArray(n)
    val leadOff = DoubleArray(n)

    var offset = 0
    for (block in record.blocks) {
      block.rPeakIndices.forEach { idx ->
        val g = offset + idx
        if (g in 0 until n) rpk[g] = 1.0
      }
      block.dataLossIndices.forEach { idx ->
        val g = offset + idx
        if (g in 0 until n) dataLost[g] = 1.0
      }
      block.leadOffIndices.forEach { idx ->
        val g = offset + idx
        if (g in 0 until n) leadOff[g] = 1.0
      }
      for (i in block.perSampleFlags.indices) {
        val g = offset + i
        if (g !in 0 until n) continue
        val flags = block.perSampleFlags[i]
        for (c in 0 until minOf(flags.size, numClasses)) {
          if (flags[c] >= 0.5f) finalFlag[g][c] = 1.0
        }
      }
      offset += block.rawAdc.size
    }

    val fs = doubleArrayOf(record.fs.toDouble())
    val rthm = HiCardiLabels.rhythmLabelMatrix()

    writeMatV5(
      outFile,
      linkedMapOf(
        "dECG" to MatArray(dEcg, intArrayOf(n, 1)),
        "final_flag" to MatArray(flatten2d(finalFlag), intArrayOf(n, numClasses)),
        "Rpk_flag" to MatArray(rpk, intArrayOf(n, 1)),
        "fs" to MatArray(fs, intArrayOf(1, 1)),
        "Rthm_label" to MatArray(flatten2d(rthm), intArrayOf(numClasses, 3)),
        "data_lost" to MatArray(dataLost, intArrayOf(n, 1)),
        "LeadOff" to MatArray(leadOff, intArrayOf(n, 1)),
      ),
    )
  }

  private fun flatten2d(rows: Array<DoubleArray>): DoubleArray {
    if (rows.isEmpty()) return doubleArrayOf()
    val cols = rows[0].size
    val out = DoubleArray(rows.size * cols)
    var k = 0
    for (row in rows) {
      for (v in row) out[k++] = v
    }
    return out
  }

  private data class MatArray(val data: DoubleArray, val dims: IntArray)

  private fun writeMatV5(file: File, variables: Map<String, MatArray>) {
    val buf = ArrayList<Byte>(65536)
    fun putBytes(b: ByteArray) { b.forEach { buf.add(it) } }
    fun putIntLE(v: Int) {
      putBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array())
    }
    fun putDoubleLE(v: Double) {
      putBytes(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(v).array())
    }

    val header = ByteArray(128)
    val title = "MATLAB 5.0 MAT-file, HiCardiMonitor, Platform: Android"
    title.toByteArray(Charsets.US_ASCII).copyInto(header, 0, 0, minOf(title.length, 116))
    putBytes(header)

    for ((name, arr) in variables) {
      val nameBytes = name.padEnd(8, '\u0000').take(8).toByteArray(Charsets.US_ASCII)
      val rows = arr.dims[0]
      val cols = arr.dims.getOrElse(1) { 1 }
      val dataBytes = arr.data.size * 8
      val matrixPayload = 8 + 8 + 8 + dataBytes
      val varPayload = 8 + 8 + matrixPayload

      putIntLE(14)
      putIntLE(varPayload)
      putIntLE(6)
      putIntLE(8)
      putIntLE(0)
      putIntLE(0)
      putBytes(nameBytes)

      putIntLE(14)
      putIntLE(matrixPayload)
      putIntLE(6)
      putIntLE(8)
      putIntLE(6)
      putIntLE(8)
      putIntLE(rows)
      putIntLE(cols)
      putIntLE(9)
      putIntLE(dataBytes)
      for (v in arr.data) putDoubleLE(v)
    }

    FileOutputStream(file).use { it.write(buf.toByteArray()) }
  }

  private fun List<Byte>.toByteArray(): ByteArray = ByteArray(size) { this[it] }
}
