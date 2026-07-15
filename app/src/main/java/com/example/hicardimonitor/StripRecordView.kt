package com.example.hicardimonitor

/**
 * Macro Strip View — eval_record_plots.py plot_record_strip
 *
 * 기본 1분(뷰포트) 고정 표시, 빈 행은 그리드+시간 라벨만.
 * 1분 초과 시 NestedScrollView 내부 스크롤 + follow live.
 */
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import androidx.core.content.ContextCompat
import androidx.core.widget.NestedScrollView
import kotlin.math.max
import kotlin.math.min

class StripRecordView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
) : View(context, attrs) {

  var buffer = StripRecordBuffer()
    set(value) {
      field = value
      requestLayout()
      invalidate()
    }

  var fs: Float = ModelConfig.FS_HZ
    set(value) {
      field = value
      requestLayout()
      invalidate()
    }

  /** HiCardi 실시간 스트립 */
  var hicardiMode: Boolean = true
    set(value) {
      field = value
      requestLayout()
      invalidate()
    }

  var scrollHost: NestedScrollView? = null
  var followLive: Boolean = true

  /** 탭한 지점의 샘플 인덱스 콜백 (설정 시 뷰가 터치를 받는다). */
  var onBeatTap: ((sampleIndex: Int) -> Unit)? = null

  // 탭만 감지하고 드래그는 소비하지 않아 부모 ScrollView 스크롤을 방해하지 않는다.
  private val tapDetector = GestureDetector(
    context,
    object : GestureDetector.SimpleOnGestureListener() {
      override fun onDown(e: MotionEvent): Boolean = true
      override fun onSingleTapUp(e: MotionEvent): Boolean {
        fireTap(e.x, e.y)
        return true
      }
    },
  )

  override fun onTouchEvent(event: MotionEvent): Boolean {
    if (onBeatTap == null) return super.onTouchEvent(event)
    tapDetector.onTouchEvent(event)
    return true
  }

  private fun fireTap(x: Float, y: Float) {
    val spr = buffer.samplesPerRow()
    val row = ((y - padV) / rowHeight).toInt().coerceAtLeast(0)
    val plotW = (width - 2 * padH).coerceAtLeast(1f)
    val fracX = ((x - padH) / plotW).coerceIn(0f, 1f)
    val idx = row * spr + (fracX * spr).toInt()
    if (idx in 0 until buffer.sampleCount) onBeatTap?.invoke(idx)
  }

  private val rowHeight get() = 120f
  private val padH = 4f
  private val padV = 6f
  private val density get() = resources.displayMetrics.scaledDensity

  private val wavePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.black)
    strokeWidth = 1.4f
    style = Paint.Style.STROKE
  }
  private val gridPaint = Paint().apply {
    color = ContextCompat.getColor(context, R.color.ecg_grid_minor)
    strokeWidth = 0.5f
  }
  private val emptyGridPaint = Paint().apply {
    color = ContextCompat.getColor(context, R.color.ecg_grid_minor)
    strokeWidth = 0.5f
    alpha = 140
  }
  private val segDividerPaint = Paint().apply {
    color = ContextCompat.getColor(context, R.color.ecg_grid)
    strokeWidth = 1f
  }
  private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_secondary)
    textSize = 11f * density
    typeface = android.graphics.Typeface.MONOSPACE
  }
  private val emptyHintPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_placeholder)
    textSize = 10f * density
    typeface = android.graphics.Typeface.MONOSPACE
    textAlign = Paint.Align.CENTER
  }
  private val predLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    textSize = 11f * density
    typeface = android.graphics.Typeface.MONOSPACE
    isFakeBoldText = true
    textAlign = Paint.Align.CENTER
  }
  private val mislabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_error)
    textSize = 10f * density
    typeface = android.graphics.Typeface.MONOSPACE
    isFakeBoldText = true
    textAlign = Paint.Align.CENTER
  }
  private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.FILL
  }

  private val classFillColors = intArrayOf(
    0,
    ContextCompat.getColor(context, R.color.strip_s),
    ContextCompat.getColor(context, R.color.strip_v),
    ContextCompat.getColor(context, R.color.strip_f),
  )
  private val classTextColors = intArrayOf(
    ContextCompat.getColor(context, R.color.text_secondary),
    ContextCompat.getColor(context, R.color.text_error),
    ContextCompat.getColor(context, R.color.text_success),
    ContextCompat.getColor(context, R.color.text_accent),
  )

  fun computeViewportHeight(): Int {
    val rows = drawRowCount()
    return (rows * rowHeight + padV * 2).toInt()
  }

  private fun drawRowCount(): Int {
    if (hicardiMode) {
      if (buffer.sampleCount == 0) return 1
      return buffer.dataRowCount().coerceAtLeast(1)
    }
    return buffer.contentRowCount()
  }

  fun notifyContentChanged() {
    requestLayout()
    invalidate()
    scrollHost?.post { scrollToLatest() }
  }

  fun scrollToLatest() {
    val host = scrollHost ?: return
    val child = host.getChildAt(0) ?: return
    val maxScroll = (child.height - host.height).coerceAtLeast(0)
    if (followLive) {
      host.smoothScrollTo(0, maxScroll)
    }
  }

  override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
    val w = MeasureSpec.getSize(widthMeasureSpec)
    val h = (drawRowCount() * rowHeight + padV * 2).toInt()
    setMeasuredDimension(w, h)
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    canvas.drawColor(ContextCompat.getColor(context, R.color.ecg_bg))

    val sig = buffer.signalArray()
    val w = width.toFloat()
    val spr = buffer.samplesPerRow()
    val rows = drawRowCount()
    val viewportRows = buffer.visibleRowCount()

    for (row in 0 until rows) {
      val top = padV + row * rowHeight
      val bottom = top + rowHeight - 6f
      val left = padH
      val right = w - padH
      val plotH = bottom - top

      val start = row * spr
      val end = min(start + spr, sig.size)
      val isEmptyRow = start >= sig.size
      val grid = if (isEmptyRow) emptyGridPaint else gridPaint

      var gy = top
      while (gy < bottom) {
        canvas.drawLine(left, gy, right, gy, grid)
        gy += plotH / 4f
      }

      val t0 = start / fs
      labelPaint.textAlign = Paint.Align.LEFT
      canvas.drawText(
        formatTime(t0),
        left + 2f,
        top + 12f,
        labelPaint,
      )

      if (isEmptyRow) {
        if (row == 0 && sig.isEmpty()) {
          emptyHintPaint.textAlign = Paint.Align.CENTER
          val hint = if (hicardiMode) {
            "waiting for BLE data — press Connect"
          } else {
            "strip: 0–${buffer.visibleWindowSec.toInt()}s window"
          }
          canvas.drawText(hint, w / 2f, top + plotH / 2f, emptyHintPaint)
        }
        continue
      }

      val seg = sig.copyOfRange(start, end)
      var sMin = seg.minOrNull() ?: 0f
      var sMax = seg.maxOrNull() ?: 1f
      val range = (sMax - sMin).coerceAtLeast(0.5f)
      sMin -= range * 0.12f
      sMax += range * 0.12f
      val yRange = sMax - sMin

      for (b in buffer.beatList) {
        if (b.segmentEnd <= start || b.segmentStart >= end) continue

        val bStart = max(b.rPeakIndex - (fs / 3f).toInt(), start)
        val bEnd = min(b.rPeakIndex + (fs / 3f).toInt(), end)

        val color = classFillColors.getOrElse(b.predId) { 0 }
        if (color != 0) {
          val x0 = left + (bStart - start).toFloat() / seg.size * (right - left)
          val x1 = left + (bEnd - start).toFloat() / seg.size * (right - left)
          fillPaint.color = color
          fillPaint.alpha = (max(0.25f, min(0.75f, b.predProb)) * 255).toInt()
          canvas.drawRect(x0, top + 2f, x1, bottom - 2f, fillPaint)
        }

      }

      if (seg.size >= 2) {
        val xStep = (right - left) / (seg.size - 1)
        var px = left
        var py = bottom - ((seg[0] - sMin) / yRange) * plotH
        for (i in 1 until seg.size) {
          val nx = left + i * xStep
          val ny = bottom - ((seg[i] - sMin) / yRange) * plotH
          canvas.drawLine(px, py, nx, ny, wavePaint)
          px = nx
          py = ny
        }
      }

      for (b in buffer.beatList) {
        if (b.rPeakIndex < start || b.rPeakIndex >= end) continue

        val cx = left + (b.rPeakIndex - start).toFloat() / seg.size * (right - left)

        if (b.predId != b.trueId && !hicardiMode) {
          val ly = bottom - ((sig[b.rPeakIndex] - sMin) / yRange) * plotH - 4f
          canvas.drawText(StripRecordBuffer.idToName(b.trueId), cx, ly, mislabelPaint)
        }
      }
    }

    if (!hicardiMode) {
      labelPaint.textAlign = Paint.Align.RIGHT
      canvas.drawText("S / V / F = color | red text = true label", w - padH, 12f, labelPaint)
    }
  }

  private fun formatTime(sec: Float): String {
    val s = sec.toInt()
    return "${s / 60}:${(s % 60).toString().padStart(2, '0')}"
  }
}
