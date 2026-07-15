package com.example.hicardimonitor.ui.widget

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.hicardimonitor.R

/**
 * 측정 타임라인 — 1초 = 눈금 하나.
 * 경과한 눈금은 잉크로 채워지고, 10초 단위 눈금은 길게 표시된다.
 * 링 카운트다운 대신 룰러(자) 메타포를 쓰는 HiCardi 고유 표현.
 */
class TickTimelineView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
) : View(context, attrs) {

  var totalSeconds: Int = 30
    set(value) {
      field = value.coerceAtLeast(1)
      invalidate()
    }

  /** 0 = 시작, 1 = 종료 */
  var elapsedFraction: Float = 0f
    set(value) {
      field = value.coerceIn(0f, 1f)
      invalidate()
    }

  private val density = resources.displayMetrics.density

  private val donePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_primary)
    strokeWidth = 2f * density
    strokeCap = Paint.Cap.ROUND
  }
  private val restPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.border_primary)
    strokeWidth = 2f * density
    strokeCap = Paint.Cap.ROUND
  }
  private val headPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.text_accent)
    strokeWidth = 2.5f * density
    strokeCap = Paint.Cap.ROUND
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val n = totalSeconds
    val w = width.toFloat() - paddingLeft - paddingRight
    if (w <= 0f || n <= 0) return

    val baseY = height * 0.78f
    val shortLen = 10f * density
    val longLen = 18f * density

    val doneCount = (elapsedFraction * n).toInt()
    val step = if (n > 1) w / (n - 1) else 0f

    for (i in 0 until n) {
      val x = paddingLeft + i * step
      val major = i % 10 == 0 || i == n - 1
      val len = if (major) longLen else shortLen
      val paint = when {
        i == doneCount && elapsedFraction > 0f && elapsedFraction < 1f -> headPaint
        i < doneCount || elapsedFraction >= 1f -> donePaint
        else -> restPaint
      }
      canvas.drawLine(x, baseY, x, baseY - len, paint)
    }
  }
}
