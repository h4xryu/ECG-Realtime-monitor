package com.example.hicardimonitor

/**
 * Sweep 방식 실시간 ECG — 고정 X축 창에서 쓰기 헤드가 래핑되며
 * [0..writeIdx] 검은색(현재 주기), [writeIdx+gap..끝] 회색(이전 주기).
 */
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.min

class EcgPlotView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
) : View(context, attrs) {

  var settings = GraphSettings(
    xWindowSec = ModelConfig.SWEEP_WINDOW_SEC,
    sampleRateHz = ModelConfig.FS_HZ,
  )
    set(value) {
      field = value
      resizeSweepBuffer()
      invalidate()
    }

  var sweepEnabled: Boolean = true

  private val displaySmoother = EcgDisplaySmoother(ModelConfig.FS_HZ)

  private var sweepBuffer = FloatArray(sweepCapacity())
  private var rPeakMarks = BooleanArray(sweepCapacity())
  /** true 면 이 인덱스에서 이전 샘플과 선을 잇지 않는다(패킷 유실 경계). */
  private var breakMarks = BooleanArray(sweepCapacity())
  private var writeIdx = 0
  private var hasWrapped = false

  // 자동 스케일 상태(EMA): 새 피크는 즉시 포함, 축소는 서서히 → 깜빡임/클리핑 방지.
  private var autoYMin = Float.NaN
  private var autoYMax = Float.NaN

  // 배터리 캐시(30초마다 갱신).
  private var batteryPct = -1
  private var lastBatteryReadMs = 0L

  private val plotLeft = 44f
  private val plotRight = 12f
  private val plotTop = 16f
  private val plotBottom = 28f

  private val gridMinorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_grid_minor)
    strokeWidth = 0.5f
  }
  private val gridMajorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_grid)
    strokeWidth = 1f
  }
  private val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_axis)
    strokeWidth = 1.5f
    style = Paint.Style.STROKE
  }
  private val zeroPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_axis)
    strokeWidth = 1f
    pathEffect = DashPathEffect(floatArrayOf(6f, 4f), 0f)
  }
  private val lineBlackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_waveform)
    strokeWidth = 2f
    style = Paint.Style.STROKE
  }
  private val lineGrayPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.GRAY
    strokeWidth = 2f
    style = Paint.Style.STROKE
  }
  private val headPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.RED
    style = Paint.Style.FILL
  }
  private val rPeakPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.CYAN
    style = Paint.Style.FILL
  }
  private val rPeakLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.CYAN
    strokeWidth = 1.5f
    style = Paint.Style.STROKE
  }
  private val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_axis)
    strokeWidth = 1f
  }
  private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_secondary)
    textSize = 11f * resources.displayMetrics.scaledDensity
    typeface = android.graphics.Typeface.MONOSPACE
  }
  private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_placeholder)
    textSize = 13f * resources.displayMetrics.scaledDensity
    textAlign = Paint.Align.CENTER
    typeface = android.graphics.Typeface.MONOSPACE
  }
  private val calPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_secondary)
    strokeWidth = 1.6f * resources.displayMetrics.density
    style = Paint.Style.STROKE
    strokeCap = Paint.Cap.SQUARE
  }
  private val cornerTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_tertiary)
    textSize = 10f * resources.displayMetrics.scaledDensity
    typeface = android.graphics.Typeface.MONOSPACE
  }
  private val batteryStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_tertiary)
    strokeWidth = 1f
    style = Paint.Style.STROKE
  }
  private val batteryFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_tertiary)
    style = Paint.Style.FILL
  }
  private val plotBgPaint = Paint().apply {
    color = ContextCompat.getColor(context, R.color.ecg_bg)
    style = Paint.Style.FILL
  }
  private val bgColor = ContextCompat.getColor(context, R.color.bg_primary)

  fun appendSamples(samples: FloatArray) = appendSamples(samples, IntArray(0), false)

  fun appendSamples(samples: FloatArray, rPeakRel: IntArray) = appendSamples(samples, rPeakRel, false)

  /**
   * @param rPeakRel 이 배치 내 R-peak 위치(batch-local 인덱스). 음수/범위밖은 무시.
   * @param gapBefore 직전 배치와 사이에 패킷 유실이 있었으면 true — 첫 샘플에서 선을 끊는다.
   */
  fun appendSamples(samples: FloatArray, rPeakRel: IntArray, gapBefore: Boolean) {
    if (!settings.showChannelEcg) return
    if (!sweepEnabled) return
    val size = sweepBuffer.size
    val startIdx = writeIdx
    val smooth = settings.smoothDisplay
    for (i in samples.indices) {
      sweepBuffer[writeIdx] = if (smooth) displaySmoother.process(samples[i]) else samples[i]
      rPeakMarks[writeIdx] = false
      breakMarks[writeIdx] = gapBefore && i == 0
      writeIdx++
      if (writeIdx >= size) {
        writeIdx = 0
        hasWrapped = true
      }
    }
    // R-peak 마커: backward 재스냅으로 rel 이 음수(과거, ≈-90샘플)일 수 있어
    // 배치 시작 인덱스 기준으로 버퍼 위치를 직접 계산해 마킹한다.
    for (rel in rPeakRel) {
      val raw = startIdx + rel
      if (raw < 0 && !hasWrapped) continue // 초반: 아직 안 쓴 영역이면 스킵
      val pos = ((raw % size) + size) % size
      rPeakMarks[pos] = true
    }
    invalidate()
  }

  fun clear() {
    writeIdx = 0
    hasWrapped = false
    sweepBuffer.fill(0f)
    rPeakMarks.fill(false)
    breakMarks.fill(false)
    autoYMin = Float.NaN
    autoYMax = Float.NaN
    displaySmoother.reset()
    invalidate()
  }

  override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
    super.onSizeChanged(w, h, oldw, oldh)
    // 캘리브레이션 모드: 가시 창(초) = 플롯 폭 / (mm/s × px/mm). 폭이 정해질 때 sweep 용량 재계산.
    if (settings.calibrated && w > 0) {
      val plotWpx = (w - plotLeft - plotRight).coerceAtLeast(1f)
      val sec = (plotWpx / (settings.mmPerSec * pxPerMmX())).coerceIn(0.5f, 30f)
      if (kotlin.math.abs(sec - settings.xWindowSec) > 0.02f) {
        settings = settings.copy(xWindowSec = sec)
      }
    }
  }

  fun currentWriteIndex(): Int = writeIdx

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    canvas.drawColor(bgColor)

    val w = width.toFloat()
    val h = height.toFloat()
    val left = plotLeft
    val right = w - plotRight
    val top = plotTop
    val bottom = h - plotBottom
    val plotW = right - left
    val plotH = bottom - top
    val midY = top + plotH / 2f

    canvas.drawRect(left, top, right, bottom, plotBgPaint)

    if (settings.showGrid) {
      if (settings.calibrated) drawGridMm(canvas, left, top, right, bottom)
      else drawGrid(canvas, left, top, right, bottom)
    }

    val (yMin, yMax) = when {
      // 임상 캘리브레이션: 세로 게인 mm/mV 고정, 베이스라인(0mV) 중앙.
      settings.calibrated -> {
        val half = (plotH / (settings.mmPerMv * pxPerMmY())) / 2f
        -half to half
      }
      settings.autoScaleY -> autoRange()
      else -> settings.yMin to settings.yMax
    }
    val yRange = (yMax - yMin).coerceAtLeast(1e-3f)
    val zeroY = bottom - ((0f - yMin) / yRange) * plotH
    if (zeroY in top..bottom) canvas.drawLine(left, zeroY, right, zeroY, zeroPaint)
    canvas.drawRect(left, top, right, bottom, axisPaint)
    drawTicks(canvas, left, top, right, bottom)

    labelPaint.textAlign = Paint.Align.RIGHT
    canvas.drawText("mV", left - 6f, top + 12f, labelPaint)

    // 스케일/배터리는 하단 여백(파형과 겹치지 않음) → 여기서 그린다. 1mV 마커는 파형 위에 올리려고 마지막에.
    if (settings.showScaleReadout) drawScaleReadout(canvas, right, h, plotH, yRange, plotW)
    if (settings.showBattery) drawBattery(canvas, left, h)

    val n = sweepBuffer.size
    if (writeIdx == 0 && !hasWrapped) {
      canvas.drawText("waiting for data", left + plotW / 2f, midY, textPaint)
      return
    }

    val gapSamples = (ModelConfig.SWEEP_GAP_SEC * settings.sampleRateHz).toInt().coerceAtLeast(1)
    val blackEnd = if (hasWrapped) writeIdx else writeIdx.coerceAtLeast(1)
    val grayStart = (writeIdx + gapSamples) % n

    if (hasWrapped && grayStart != writeIdx) {
      drawSweepSegment(canvas, grayStart, n, left, top, bottom, plotW, plotH, yMin, yRange, lineGrayPaint)
      if (grayStart > 0) {
        drawSweepSegment(canvas, 0, grayStart, left, top, bottom, plotW, plotH, yMin, yRange, lineGrayPaint)
      }
    }
    drawSweepSegment(canvas, 0, blackEnd, left, top, bottom, plotW, plotH, yMin, yRange, lineBlackPaint)

    if (settings.showRPeak) {
      drawRPeakMarkers(canvas, left, top, bottom, plotW, plotH, yMin, yRange)
    }

    if (blackEnd > 0) {
      val headX = left + (blackEnd - 1).toFloat() / (n - 1).coerceAtLeast(1) * plotW
      val headY = bottom - ((sweepBuffer[blackEnd - 1] - yMin) / yRange) * plotH
      canvas.drawCircle(headX, headY.coerceIn(top, bottom), 4f, headPaint)
    }

    // 1mV 캘리브레이션 마커는 파형 위에 표시(좌측, 레퍼런스 방식).
    if (settings.showCalMarker) drawCalMarker(canvas, left, zeroY, top, bottom, plotH, yRange)
  }

  /**
   * 표시 y창 자동 스케일 — 현재 sweep 버퍼(채워진 구간)의 min/max에 10% 여백.
   * ecgplot.py 가 ylim 고정 없이 auto-scale 하는 것과 동일한 취지(형태·R-peak 항상 포함).
   * EMA: 새 진폭은 즉시 포함(피크 클리핑 방지), 축소는 서서히(깜빡임 방지).
   */
  private fun autoRange(): Pair<Float, Float> {
    val end = if (hasWrapped) sweepBuffer.size else writeIdx
    if (end < 2) return settings.yMin to settings.yMax
    var mn = Float.POSITIVE_INFINITY
    var mx = Float.NEGATIVE_INFINITY
    for (i in 0 until end) {
      val v = sweepBuffer[i]
      if (v < mn) mn = v
      if (v > mx) mx = v
    }
    if (!mn.isFinite() || !mx.isFinite()) return settings.yMin to settings.yMax
    val span = (mx - mn).coerceAtLeast(MIN_SPAN)
    val pad = span * 0.1f
    val targetMin = mn - pad
    val targetMax = mx + pad
    autoYMin = if (autoYMin.isNaN()) targetMin
    else if (targetMin < autoYMin) targetMin else autoYMin + SHRINK * (targetMin - autoYMin)
    autoYMax = if (autoYMax.isNaN()) targetMax
    else if (targetMax > autoYMax) targetMax else autoYMax + SHRINK * (targetMax - autoYMax)
    return autoYMin to autoYMax
  }

  private fun pxPerMmY(): Float = (resources.displayMetrics.ydpi / 25.4f).coerceIn(4f, 40f)
  private fun pxPerMmX(): Float = (resources.displayMetrics.xdpi / 25.4f).coerceIn(4f, 40f)

  /** 좌측 1 mV 캘리브레이션 마커(현재 게인 기준 실제 높이). */
  private fun drawCalMarker(
    canvas: Canvas,
    left: Float,
    zeroY: Float,
    top: Float,
    bottom: Float,
    plotH: Float,
    yRange: Float,
  ) {
    val pxPerMv = plotH / yRange
    val barH = pxPerMv.coerceIn(10f, plotH * 0.9f)
    val x = left + 12f
    val baseY = zeroY.coerceIn(top + barH, bottom)
    val topY = baseY - barH
    val serif = 4f
    canvas.drawLine(x, baseY, x, topY, calPaint)
    canvas.drawLine(x - serif, baseY, x + serif, baseY, calPaint)
    canvas.drawLine(x - serif, topY, x + serif, topY, calPaint)
    cornerTextPaint.textAlign = Paint.Align.LEFT
    canvas.drawText("1 mV", x + serif + 3f, (topY + barH * 0.5f) + cornerTextPaint.textSize * 0.35f, cornerTextPaint)
  }

  /** 우하단 mm/mV · mm/s (캘리브레이션 모드는 설정값, 아니면 현재 표시 스케일에서 실측). */
  private fun drawScaleReadout(canvas: Canvas, right: Float, h: Float, plotH: Float, yRange: Float, plotW: Float) {
    val mmPerMv: Float
    val mmPerSec: Float
    if (settings.calibrated) {
      mmPerMv = settings.mmPerMv
      mmPerSec = settings.mmPerSec
    } else {
      mmPerMv = (plotH / yRange) / pxPerMmY()
      val secondsShown = (sweepBuffer.size / settings.sampleRateHz).coerceAtLeast(1e-3f)
      mmPerSec = (plotW / secondsShown) / pxPerMmX()
    }
    cornerTextPaint.textAlign = Paint.Align.RIGHT
    canvas.drawText("%.1f mm/mV, %.1f mm/s".format(mmPerMv, mmPerSec), right, h - 5f, cornerTextPaint)
  }

  /** 임상 ECG 방안지 그리드 — 1mm 얇은 선, 5mm 굵은 선 (dpi 기준). */
  private fun drawGridMm(canvas: Canvas, left: Float, top: Float, right: Float, bottom: Float) {
    val mmX = pxPerMmX()
    val mmY = pxPerMmY()
    // 1mm minor
    var x = left
    while (x <= right) {
      canvas.drawLine(x, top, x, bottom, gridMinorPaint)
      x += mmX
    }
    var y = top
    while (y <= bottom) {
      canvas.drawLine(left, y, right, y, gridMinorPaint)
      y += mmY
    }
    // 5mm major
    x = left
    while (x <= right) {
      canvas.drawLine(x, top, x, bottom, gridMajorPaint)
      x += mmX * 5f
    }
    y = top
    while (y <= bottom) {
      canvas.drawLine(left, y, right, y, gridMajorPaint)
      y += mmY * 5f
    }
  }

  /** 좌하단 배터리 잔량(폰 기준, 30초 캐시). */
  private fun drawBattery(canvas: Canvas, left: Float, h: Float) {
    val pct = batteryPercent()
    if (pct < 0) return
    val by = h - 5f
    val bx = left
    val bw = 16f
    val bh = 8f
    val iconTop = by - bh
    canvas.drawRect(bx, iconTop, bx + bw, by, batteryStrokePaint)
    canvas.drawRect(bx + bw, iconTop + 2f, bx + bw + 2f, by - 2f, batteryFillPaint)
    val fillW = ((bw - 2f) * (pct / 100f)).coerceAtLeast(0f)
    canvas.drawRect(bx + 1f, iconTop + 1f, bx + 1f + fillW, by - 1f, batteryFillPaint)
    cornerTextPaint.textAlign = Paint.Align.LEFT
    canvas.drawText("$pct%", bx + bw + 6f, by, cornerTextPaint)
  }

  private fun batteryPercent(): Int {
    val now = System.currentTimeMillis()
    if (batteryPct < 0 || now - lastBatteryReadMs > 30_000) {
      lastBatteryReadMs = now
      batteryPct = try {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
        bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
      } catch (_: Throwable) {
        -1
      }
    }
    return batteryPct
  }

  private fun drawRPeakMarkers(
    canvas: Canvas,
    left: Float,
    top: Float,
    bottom: Float,
    plotW: Float,
    plotH: Float,
    yMin: Float,
    yRange: Float,
  ) {
    val n = sweepBuffer.size
    val xStep = plotW / (n - 1).coerceAtLeast(1)
    for (i in 0 until n) {
      if (!rPeakMarks[i]) continue
      val x = left + i * xStep
      val y = (bottom - ((sweepBuffer[i] - yMin) / yRange) * plotH).coerceIn(top, bottom)
      val markTop = (y - 12f).coerceAtLeast(top)
      canvas.drawLine(x, markTop, x, y, rPeakLinePaint)
      canvas.drawCircle(x, markTop, 3f, rPeakPaint)
    }
  }

  private fun drawSweepSegment(
    canvas: Canvas,
    startIdx: Int,
    endIdx: Int,
    left: Float,
    top: Float,
    bottom: Float,
    plotW: Float,
    plotH: Float,
    yMin: Float,
    yRange: Float,
    paint: Paint,
  ) {
    val n = sweepBuffer.size
    if (endIdx - startIdx < 2) return
    val xStep = plotW / (n - 1).coerceAtLeast(1)
    var prevX = left + startIdx * xStep
    var prevY = bottom - ((sweepBuffer[startIdx] - yMin) / yRange) * plotH
    for (i in startIdx + 1 until endIdx) {
      val px = left + i * xStep
      val py = (bottom - ((sweepBuffer[i] - yMin) / yRange) * plotH).coerceIn(top, bottom)
      // 패킷 유실 경계에서는 잇지 않고 새 구간을 시작(가파른 연결선 방지).
      if (!breakMarks[i]) {
        canvas.drawLine(prevX, prevY.coerceIn(top, bottom), px, py, paint)
      }
      prevX = px
      prevY = py
    }
  }

  private companion object {
    /** 자동 스케일 최소 창 폭(앱 단위) — 평탄 구간에서 과도 확대 방지. */
    const val MIN_SPAN = 2f
    /** 창 축소 EMA 계수(작을수록 서서히 축소). */
    const val SHRINK = 0.05f
  }

  private fun sweepCapacity(): Int {
    return (settings.xWindowSec * settings.sampleRateHz).toInt().coerceAtLeast(2)
  }

  private fun resizeSweepBuffer() {
    val cap = sweepCapacity()
    val next = FloatArray(cap)
    val nextMarks = BooleanArray(cap)
    val nextBreaks = BooleanArray(cap)
    val copyN = min(cap, sweepBuffer.size)
    for (i in 0 until copyN) {
      next[i] = sweepBuffer[i]
      if (i < rPeakMarks.size) nextMarks[i] = rPeakMarks[i]
      if (i < breakMarks.size) nextBreaks[i] = breakMarks[i]
    }
    sweepBuffer = next
    rPeakMarks = nextMarks
    breakMarks = nextBreaks
    if (writeIdx >= cap) {
      writeIdx = 0
      hasWrapped = true
    }
  }

  private fun drawGrid(canvas: Canvas, left: Float, top: Float, right: Float, bottom: Float) {
    val minor = 20f
    var x = left
    while (x <= right) {
      canvas.drawLine(x, top, x, bottom, gridMinorPaint)
      x += minor
    }
    var y = top
    while (y <= bottom) {
      canvas.drawLine(left, y, right, y, gridMinorPaint)
      y += minor
    }
    val major = 40f
    x = left
    while (x <= right) {
      canvas.drawLine(x, top, x, bottom, gridMajorPaint)
      x += major
    }
    y = top
    while (y <= bottom) {
      canvas.drawLine(left, y, right, y, gridMajorPaint)
      y += major
    }
  }

  private fun drawTicks(canvas: Canvas, left: Float, top: Float, right: Float, bottom: Float) {
    val major = 40f
    val tickLen = 5f
    var x = left
    while (x <= right) {
      canvas.drawLine(x, bottom, x, bottom + tickLen, tickPaint)
      x += major
    }
    var y = top
    while (y <= bottom) {
      canvas.drawLine(left - tickLen, y, left, y, tickPaint)
      y += major
    }
  }
}

typealias EcgView = EcgPlotView
