package com.example.hicardimonitor.data

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.pdf.PdfDocument
import com.example.hicardimonitor.HiCardiSignal
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs

/**
 * 임상 ECG strip-view PDF 리포트.
 *
 * ISPLAB 슬라이드와 동일한 표기: "HiCardi ECG Strip View (25 mm/s, 10 mm/mV)".
 * A4 세로, dpi 기준 mm 방안지(1mm 얇은선/5mm 굵은선, 핑크), 행 단위로 전체 기록을 페이지네이션.
 * 1페이지 상단에 요약 통계(Samples/Fs/Start/Duration/R-peaks/Avg HR/data_lost/lead_off/ECG mV).
 *
 * 신호는 raw ADC → `(adc-16384)/1000` mV(SSOT, 스케일 불변)로 그린다. 행별 median 을 빼
 * 베이스라인을 0 으로 정렬(드리프트 무관, 진폭·1mV 마커 정확도 보존). PdfDocument 는
 * pt(1/72in) 단위 = 72dpi → 1mm = 72/25.4 pt.
 */
object RecordingPdfExporter {

  private const val PT_PER_MM = 72f / 25.4f
  private const val MM_PER_SEC = 25f
  private const val MM_PER_MV = 10f
  private const val PAGE_W = 595f          // A4 세로 pt
  private const val PAGE_H = 842f
  private const val MARGIN = 28f           // ≈10mm
  private const val MV_HALF = 2f           // 행당 ±2mV

  fun fileNameFor(record: DiagnosisRecord): String {
    val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(record.startedAtMs))
    return "hicardi_strip_${stamp}.pdf"
  }

  fun export(record: DiagnosisRecord, outFile: File) {
    val raw = record.mergedRawAdc()
    val n = raw.size
    val fs = record.fs.toFloat().coerceAtLeast(1f)
    val mv = FloatArray(n) { (raw[it] - HiCardiSignal.BLE_ADC_OFFSET) / HiCardiSignal.COUNTS_PER_MV }

    var rpeaks = 0
    var dataLost = 0
    var leadOff = 0
    for (b in record.blocks) {
      rpeaks += b.rPeakIndices.size
      dataLost += b.dataLossIndices.size
      leadOff += b.leadOffIndices.size
    }
    val durSec = if (n > 1) (n - 1) / fs else 0f
    val avgHr = if (durSec > 0f && rpeaks > 1) rpeaks / (durSec / 60f) else 0f
    var mvMin = 0f
    var mvMax = 0f
    if (n > 0) {
      mvMin = mv[0]; mvMax = mv[0]
      for (v in mv) { if (v < mvMin) mvMin = v; if (v > mvMax) mvMax = v }
    }

    val pxPerSec = MM_PER_SEC * PT_PER_MM
    val pxPerMv = MM_PER_MV * PT_PER_MM
    val gutter = 13f * PT_PER_MM               // 좌측 mV 축 + 1mV 마커
    val plotLeft = MARGIN + gutter
    val plotRight = PAGE_W - MARGIN
    val plotW = plotRight - plotLeft
    val secPerRow = plotW / pxPerSec
    val samplesPerRow = (secPerRow * fs).toInt().coerceAtLeast(1)
    val rowH = MV_HALF * 2f * pxPerMv          // ±2mV 세로 크기
    val rowGap = 7f * PT_PER_MM
    val rowBlock = rowH + rowGap

    val startStr = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date(record.startedAtMs))
    val title = "HiCardi ECG Strip View (25 mm/s, 10 mm/mV) - Start: $startStr"
    val summary = listOf(
      "Samples    : $n",
      "Fs         : ${"%.1f".format(fs)} Hz",
      "Start time : $startStr",
      "Duration   : ${"%.2f".format(durSec)} s",
      "R-peaks    : $rpeaks",
      "Avg HR     : ${"%.1f".format(avgHr)} bpm",
      "data_lost  : $dataLost samples",
      "lead_off   : $leadOff samples",
      "ECG (mV)   : min=${"%.3f".format(mvMin)} max=${"%.3f".format(mvMax)}",
    )

    // ── paints ────────────────────────────────────────────────────────────
    val minorPaint = Paint().apply { color = Color.rgb(0xF4, 0xCF, 0xCF); strokeWidth = 0.4f }
    val majorPaint = Paint().apply { color = Color.rgb(0xE0, 0x9A, 0x9A); strokeWidth = 0.8f }
    val wavePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.rgb(0x10, 0x10, 0x10); strokeWidth = 0.8f; style = Paint.Style.STROKE
    }
    val calPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.rgb(0x30, 0x30, 0x30); strokeWidth = 1f; style = Paint.Style.STROKE
    }
    val axisText = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x60, 0x60, 0x60); textSize = 6f }
    val timeText = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x80, 0x50, 0x50); textSize = 6.5f }
    val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLACK; textSize = 10f; isFakeBoldText = true }
    val summaryPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.rgb(0x20, 0x20, 0x20); textSize = 9f; typeface = android.graphics.Typeface.MONOSPACE
    }
    val footPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x90, 0x90, 0x90); textSize = 7f }

    val headerH = 22f
    val summaryH = summary.size * 11f + 10f

    // 총 행 수 → 페이지 수 사전 계산
    val totalRows = ((n + samplesPerRow - 1) / samplesPerRow).coerceAtLeast(1)
    fun rowsForPage(page: Int): Int {
      val top = MARGIN + headerH + (if (page == 0) summaryH else 0f)
      val avail = PAGE_H - MARGIN - top
      return (avail / rowBlock).toInt().coerceAtLeast(1)
    }
    var pages = 0
    run { var r = 0; var pg = 0; while (r < totalRows) { r += rowsForPage(pg); pg++ }; pages = pg }

    val doc = PdfDocument()
    var rowIdx = 0
    var page = 0
    while (rowIdx < totalRows) {
      val info = PdfDocument.PageInfo.Builder(PAGE_W.toInt(), PAGE_H.toInt(), page + 1).create()
      val p = doc.startPage(info)
      val c = p.canvas
      c.drawColor(Color.WHITE)

      // 헤더
      c.drawText(title, MARGIN, MARGIN + 12f, titlePaint)
      c.drawText("page ${page + 1} / $pages", PAGE_W - MARGIN - 60f, MARGIN + 12f, footPaint)

      var rowTop = MARGIN + headerH
      if (page == 0) {
        var ty = rowTop + 10f
        for (line in summary) { c.drawText(line, MARGIN, ty, summaryPaint); ty += 11f }
        rowTop += summaryH
      }

      val rowsThis = rowsForPage(page)
      var drawn = 0
      while (drawn < rowsThis && rowIdx < totalRows) {
        val startSample = rowIdx * samplesPerRow
        drawRow(
          c, mv, startSample, samplesPerRow, fs,
          plotLeft, plotRight, rowTop, rowH, pxPerSec, pxPerMv,
          minorPaint, majorPaint, wavePaint, calPaint, axisText, timeText,
        )
        rowTop += rowBlock
        rowIdx++
        drawn++
      }

      doc.finishPage(p)
      page++
    }

    FileOutputStream(outFile).use { doc.writeTo(it) }
    doc.close()
  }

  private fun drawRow(
    c: Canvas,
    mv: FloatArray,
    startSample: Int,
    samplesPerRow: Int,
    fs: Float,
    plotLeft: Float,
    plotRight: Float,
    rowTop: Float,
    rowH: Float,
    pxPerSec: Float,
    pxPerMv: Float,
    minorPaint: Paint,
    majorPaint: Paint,
    wavePaint: Paint,
    calPaint: Paint,
    axisText: Paint,
    timeText: Paint,
  ) {
    val centerY = rowTop + rowH / 2f
    val bottom = rowTop + rowH
    val mmX = PT_PER_MM
    val mmY = PT_PER_MM

    // 방안지 — 1mm minor
    var gx = plotLeft
    while (gx <= plotRight + 0.1f) { c.drawLine(gx, rowTop, gx, bottom, minorPaint); gx += mmX }
    var gy = rowTop
    while (gy <= bottom + 0.1f) { c.drawLine(plotLeft, gy, plotRight, gy, minorPaint); gy += mmY }
    // 5mm major (0mV 기준 상·하 대칭으로 정렬)
    gx = plotLeft
    while (gx <= plotRight + 0.1f) { c.drawLine(gx, rowTop, gx, bottom, majorPaint); gx += mmX * 5f }
    var yUp = centerY
    while (yUp >= rowTop - 0.1f) { c.drawLine(plotLeft, yUp, plotRight, yUp, majorPaint); yUp -= mmY * 5f }
    var yDn = centerY + mmY * 5f
    while (yDn <= bottom + 0.1f) { c.drawLine(plotLeft, yDn, plotRight, yDn, majorPaint); yDn += mmY * 5f }

    // mV 축 라벨 (-2..2, 1mV 간격)
    axisText.textAlign = Paint.Align.RIGHT
    for (m in -2..2) {
      val y = centerY - m * pxPerMv
      if (y in rowTop - 1f..bottom + 1f) c.drawText("%.1f".format(m.toFloat()), plotLeft - 2f, y + 2f, axisText)
    }

    // 1mV 캘리브레이션 마커 (좌측 gutter)
    val calX = plotLeft - 10f * PT_PER_MM
    val calTop = centerY - pxPerMv
    val calW = 0.12f * pxPerSec
    c.drawLine(calX, centerY, calX, calTop, calPaint)
    c.drawLine(calX, calTop, calX + calW, calTop, calPaint)
    c.drawLine(calX + calW, calTop, calX + calW, centerY, calPaint)
    axisText.textAlign = Paint.Align.LEFT
    c.drawText("1 mV", calX, calTop - 2f, axisText)

    // 시각 라벨
    val t0 = startSample / fs
    c.drawText(formatTime(t0), plotLeft + 2f, rowTop + 8f, timeText)

    // 행 baseline = median (드리프트 제거)
    val end = (startSample + samplesPerRow).coerceAtMost(mv.size)
    if (end - startSample < 2) return
    val slice = mv.copyOfRange(startSample, end)
    val sorted = slice.copyOf(); sorted.sort()
    val base = sorted[sorted.size / 2]

    // 파형
    var prevX = plotLeft
    var prevY = centerY - clampMv(slice[0] - base) * pxPerMv
    for (i in 1 until slice.size) {
      val x = plotLeft + (i / fs) * pxPerSec
      if (x > plotRight) break
      val y = centerY - clampMv(slice[i] - base) * pxPerMv
      c.drawLine(prevX, prevY, x, y, wavePaint)
      prevX = x; prevY = y
    }
  }

  private fun clampMv(v: Float): Float = if (v > MV_HALF) MV_HALF else if (v < -MV_HALF) -MV_HALF else v

  private fun formatTime(sec: Float): String {
    val s = sec.toInt()
    return "${s / 60}:${(s % 60).toString().padStart(2, '0')}"
  }
}
