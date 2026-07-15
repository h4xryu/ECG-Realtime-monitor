package com.example.hicardimonitor.ui.widget

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.hicardimonitor.R

/**
 * 2초 윈도우 파형 + 선택적 CAM 오버레이.
 *
 * cam 이 설정되면 각 구간을 중요도(0..1)로 jet 채색 — visualize_cpc.plot_gradcam_windows 의
 * "빨강 = 임베딩을 좌우" 의미를 그대로 따른다(여기선 occlusion saliency).
 */
class SaliencyWaveView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
) : View(context, attrs) {

  private var signal: FloatArray = FloatArray(0)
  private var cam: FloatArray? = null

  private val plainPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = ContextCompat.getColor(context, R.color.ecg_waveform)
    strokeWidth = 2.2f
    style = Paint.Style.STROKE
  }
  private val camPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    strokeWidth = 3f
    style = Paint.Style.STROKE
  }
  private val gridPaint = Paint().apply {
    color = ContextCompat.getColor(context, R.color.ecg_grid_minor)
    strokeWidth = 0.5f
  }

  fun setSignal(s: FloatArray) {
    signal = s
    invalidate()
  }

  fun setCam(c: FloatArray?) {
    cam = c
    invalidate()
  }

  val hasCam: Boolean get() = cam != null

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    canvas.drawColor(ContextCompat.getColor(context, R.color.ecg_bg))
    val n = signal.size
    val w = width.toFloat()
    val h = height.toFloat()
    val pad = 8f

    var gy = pad
    while (gy < h - pad) {
      canvas.drawLine(0f, gy, w, gy, gridPaint)
      gy += (h - 2 * pad) / 4f
    }
    if (n < 2) return

    var sMin = signal[0]
    var sMax = signal[0]
    for (v in signal) {
      if (v < sMin) sMin = v
      if (v > sMax) sMax = v
    }
    val range = (sMax - sMin).coerceAtLeast(1e-3f)
    val plotH = h - 2 * pad
    val xStep = w / (n - 1)
    val c = cam
    var px = 0f
    var py = pad + plotH - (signal[0] - sMin) / range * plotH
    for (i in 1 until n) {
      val nx = i * xStep
      val ny = pad + plotH - (signal[i] - sMin) / range * plotH
      if (c != null && c.size == n) {
        camPaint.color = jet(((c[i] + c[i - 1]) * 0.5f).coerceIn(0f, 1f))
        canvas.drawLine(px, py, nx, ny, camPaint)
      } else {
        canvas.drawLine(px, py, nx, ny, plainPaint)
      }
      px = nx
      py = ny
    }
  }

  companion object {
    /** jet colormap 0(파랑)→0.5(초록)→1(빨강). */
    fun jet(t: Float): Int {
      val x = t.coerceIn(0f, 1f)
      val r = (1.5f - kotlin.math.abs(4f * x - 3f)).coerceIn(0f, 1f)
      val g = (1.5f - kotlin.math.abs(4f * x - 2f)).coerceIn(0f, 1f)
      val b = (1.5f - kotlin.math.abs(4f * x - 1f)).coerceIn(0f, 1f)
      return Color.rgb((r * 255).toInt(), (g * 255).toInt(), (b * 255).toInt())
    }
  }
}
