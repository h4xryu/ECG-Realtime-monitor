package com.example.hicardimonitor.ui.widget

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.hicardimonitor.ArrhythmiaConfig
import com.example.hicardimonitor.R

/**
 * 표시 클래스(NUM_CLASSES, 서맥 제외 5) 확률 막대그래프.
 *
 * 각 클래스별 세로 막대(확률 0..1), 상단에 % 값, 하단에 라벨.
 * argmax(대표 예측) 막대는 강조색.
 */
class ArrhythmiaBarView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
  defStyle: Int = 0,
) : View(context, attrs, defStyle) {

  private var probs: FloatArray = FloatArray(ArrhythmiaConfig.NUM_CLASSES)
  private var topIndex: Int = -1
  private var hasData = false

  private val density = resources.displayMetrics.density
  private fun dp(v: Float) = v * density

  private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG)
  private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_grid_minor)
  }
  private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_secondary)
    textSize = dp(11f)
    textAlign = Paint.Align.CENTER
  }
  private val valuePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_primary)
    textSize = dp(11f)
    textAlign = Paint.Align.CENTER
  }
  private val colorAccent = ContextCompat.getColor(context, R.color.text_accent)
  private val colorMuted = ContextCompat.getColor(context, R.color.text_tertiary)

  fun setData(probabilities: FloatArray, top: Int) {
    probs = FloatArray(ArrhythmiaConfig.NUM_CLASSES) { probabilities.getOrElse(it) { 0f } }
    topIndex = top
    hasData = true
    invalidate()
  }

  fun clear() {
    hasData = false
    topIndex = -1
    probs = FloatArray(ArrhythmiaConfig.NUM_CLASSES)
    invalidate()
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val n = ArrhythmiaConfig.NUM_CLASSES
    val topPad = dp(18f) // 값 텍스트 공간
    val bottomPad = dp(20f) // 라벨 공간
    val leftPad = dp(4f)
    val rightPad = dp(4f)

    val plotTop = topPad
    val plotBottom = height - bottomPad
    val plotHeight = (plotBottom - plotTop).coerceAtLeast(1f)
    val plotLeft = leftPad
    val plotRight = width - rightPad
    val plotWidth = (plotRight - plotLeft).coerceAtLeast(1f)

    val slot = plotWidth / n
    val barW = slot * 0.52f

    // 막대(트랙 + 값)
    for (i in 0 until n) {
      val cx = plotLeft + slot * (i + 0.5f)
      val left = cx - barW / 2f
      val right = cx + barW / 2f

      // 트랙(전체 높이 배경)
      trackPaint.style = Paint.Style.FILL
      canvas.drawRoundRect(left, plotTop, right, plotBottom, dp(3f), dp(3f), trackPaint)

      val p = probs[i].coerceIn(0f, 1f)
      val barTop = plotBottom - p * plotHeight
      // 대표 예측(argmax)만 강조, 나머지는 흐리게. (임계선 없음)
      barPaint.color = if (!hasData) colorMuted else colorAccent
      barPaint.alpha = if (!hasData) 90 else if (i == topIndex) 255 else 150
      canvas.drawRoundRect(left, barTop, right, plotBottom, dp(3f), dp(3f), barPaint)
      barPaint.alpha = 255

      // 값 %
      if (hasData) {
        valuePaint.typeface = if (i == topIndex) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        val pct = (p * 100f)
        val txt = if (pct >= 99.5f) "100" else String.format("%.0f", pct)
        canvas.drawText(txt, cx, (barTop - dp(4f)).coerceAtLeast(plotTop + dp(9f)), valuePaint)
      }

      // 라벨
      labelPaint.typeface = if (i == topIndex) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
      labelPaint.color = if (hasData && i == topIndex) {
        ContextCompat.getColor(context, R.color.text_primary)
      } else {
        ContextCompat.getColor(context, R.color.text_secondary)
      }
      canvas.drawText(ArrhythmiaConfig.labelKo(i), cx, plotBottom + dp(14f), labelPaint)
    }
  }
}
