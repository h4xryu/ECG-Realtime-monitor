package com.example.hicardimonitor.ui.widget

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.hicardimonitor.R
import kotlin.math.abs

/**
 * Patient embedding 지문 — 256-dim 벡터를 발산형 히트맵 한 줄로 표시.
 * 음수=파랑, 0=옅음, 양수=빨강 (|max| 정규화). 개인화가 될수록 고유 패턴을 가진다.
 */
class EmbeddingView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
) : View(context, attrs) {

  private var values: FloatArray = FloatArray(0)
  private val cellPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
  private val negColor = ContextCompat.getColor(context, R.color.text_accent)
  private val posColor = ContextCompat.getColor(context, R.color.text_error)
  private val zeroColor = Color.rgb(245, 244, 240)

  fun setEmbedding(v: FloatArray) {
    values = v
    invalidate()
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val n = values.size
    if (n == 0) return
    var maxAbs = 1e-6f
    for (x in values) maxAbs = maxOf(maxAbs, abs(x))
    val w = width.toFloat()
    val h = height.toFloat()
    val cw = w / n
    for (i in 0 until n) {
      val t = (values[i] / maxAbs).coerceIn(-1f, 1f)
      cellPaint.color = if (t >= 0f) lerp(zeroColor, posColor, t) else lerp(zeroColor, negColor, -t)
      val x = i * cw
      canvas.drawRect(x, 0f, x + cw + 1f, h, cellPaint)
    }
  }

  private fun lerp(a: Int, b: Int, t: Float): Int {
    val r = (Color.red(a) + (Color.red(b) - Color.red(a)) * t).toInt()
    val g = (Color.green(a) + (Color.green(b) - Color.green(a)) * t).toInt()
    val bl = (Color.blue(a) + (Color.blue(b) - Color.blue(a)) * t).toInt()
    return Color.rgb(r.coerceIn(0, 255), g.coerceIn(0, 255), bl.coerceIn(0, 255))
  }
}
