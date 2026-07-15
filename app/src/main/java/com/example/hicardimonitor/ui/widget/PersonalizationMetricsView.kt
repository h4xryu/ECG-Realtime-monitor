package com.example.hicardimonitor.ui.widget

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.hicardimonitor.R
import com.example.hicardimonitor.session.PersonalizationQuality

/**
 * 개인화 품질 지표화 — 말 없이 숫자·막대만.
 *
 *   FIT  = 0..100 종합 (색: good/fair/weak)
 *   SELF = 앞/뒤 30초 임베딩 코사인 (자기 리듬 재현성)
 *   BEAT = 윈도우 ↔ pooled 코사인 (자기 beat 정합)
 *   RANK = 임베딩 effective rank (붕괴 감지)
 */
class PersonalizationMetricsView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
) : View(context, attrs) {

  private var quality: PersonalizationQuality? = null
  private val density = resources.displayMetrics.density

  private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_secondary)
    textSize = 12f * density
    typeface = Typeface.MONOSPACE
  }
  private val valuePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_primary)
    textSize = 12f * density
    typeface = Typeface.MONOSPACE
    textAlign = Paint.Align.RIGHT
  }
  private val bigPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    textSize = 40f * density
    typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
  }
  private val barBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.border_primary)
  }
  private val barFg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_accent)
  }
  private val goodColor = ContextCompat.getColor(context, R.color.text_success)
  private val fairColor = ContextCompat.getColor(context, R.color.text_warning)
  private val weakColor = ContextCompat.getColor(context, R.color.text_error)

  fun setQuality(q: PersonalizationQuality?) {
    quality = q
    requestLayout()
    invalidate()
  }

  override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
    val w = MeasureSpec.getSize(widthMeasureSpec)
    val h = ((if (quality == null) 60f else 210f) * density).toInt()
    setMeasuredDimension(w, h)
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val q = quality ?: run {
      labelPaint.textAlign = Paint.Align.LEFT
      canvas.drawText("—", 0f, 30f * density, labelPaint)
      return
    }
    val fitColor = when (q.fitLabelKey) {
      "good" -> goodColor
      "fair" -> fairColor
      else -> weakColor
    }
    // FIT 대형 숫자
    labelPaint.textAlign = Paint.Align.LEFT
    canvas.drawText("FIT", 0f, 18f * density, labelPaint)
    bigPaint.color = fitColor
    canvas.drawText(q.fitScore.toString(), 0f, 56f * density, bigPaint)

    // 하위 3개 막대
    var y = 92f * density
    drawBar(canvas, "SELF", frac(q.selfConsistency), fmt2(q.selfConsistency), y)
    y += 38f * density
    drawBar(canvas, "BEAT", frac(q.beatCoherence), fmt2(q.beatCoherence), y)
    y += 38f * density
    val rankFrac = if (q.windowCount > 1) (q.effectiveRank / q.windowCount).coerceIn(0f, 1f) else 0f
    drawBar(canvas, "RANK", rankFrac, q.effectiveRank.toInt().toString(), y)
  }

  private fun drawBar(canvas: Canvas, label: String, frac: Float, value: String, y: Float) {
    labelPaint.textAlign = Paint.Align.LEFT
    canvas.drawText(label, 0f, y - 4f * density, labelPaint)
    val barLeft = 56f * density
    val barRight = width - 52f * density
    val barTop = y - 16f * density
    val barBottom = y + 2f * density
    val r = 3f * density
    canvas.drawRoundRect(barLeft, barTop, barRight, barBottom, r, r, barBg)
    val fillRight = barLeft + (barRight - barLeft) * frac.coerceIn(0f, 1f)
    canvas.drawRoundRect(barLeft, barTop, fillRight.coerceAtLeast(barLeft + 1f), barBottom, r, r, barFg)
    valuePaint.textAlign = Paint.Align.RIGHT
    canvas.drawText(value, width.toFloat(), y - 2f * density, valuePaint)
  }

  // 코사인 지표(0.7..1.0)를 막대 길이로 스트레치 — MetricsView 표시 전용.
  private fun frac(cos: Float): Float = ((cos - 0.70f) / 0.30f).coerceIn(0f, 1f)
  private fun fmt2(v: Float): String = String.format("%.2f", v)
}
