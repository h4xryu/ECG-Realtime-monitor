package com.example.hicardimonitor.session

import android.content.Context
import android.util.Log
import com.example.hicardimonitor.ModelConfig
import com.example.hicardimonitor.SignalUtils
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * CPC 개인화 인코더 러너 (cpc_dsconv, export_to_pte.py).
 *
 * 모델 계약: 입력 [1,1,500] (2 s @ 250 Hz, z-score) → 출력 [1,256] pooled patient embedding.
 * 1분 캘리브레이션은 2 s 윈도우 50% overlap 으로 타일링해 윈도우별 임베딩을 평균한다(§Eq.7 pooling).
 *
 * Grad-CAM 은 autograd 가 필요하지만 .pte 는 추론 전용이므로, 동일한 의도
 * ("어느 구간이 임베딩을 좌우하는가")를 forward-only occlusion saliency 로 근사한다.
 */
class CpcRunner(private val context: Context) {

  private var module: Module? = null
  var lastError: String? = null
    private set

  val isLoaded: Boolean get() = module != null

  fun load(): Boolean {
    return try {
      ensureNative()
      val path = copyAssetToFile(ModelConfig.CPC_PTE_FILE).absolutePath
      module = Module.load(path)
      Log.i(TAG, "cpc load OK: $path")
      true
    } catch (e: Throwable) {
      module = null
      lastError = e.message ?: e.toString()
      Log.e(TAG, "cpc load FAIL: $lastError", e)
      false
    }
  }

  fun close() {
    try {
      module?.destroy()
    } catch (_: Throwable) {
    }
    module = null
  }

  /** 단일 2 s 윈도우 → 256-dim 임베딩 (미정규화). 실패 시 null. */
  fun embedWindow(window: FloatArray): FloatArray? {
    val m = module ?: return null
    return try {
      val x = fitWindow(window)
      val t = Tensor.fromBlob(x, longArrayOf(1, 1, ModelConfig.CPC_WINDOW_SIZE.toLong()))
      val out = m.forward(EValue.from(t))
      out[0].toTensor().getDataAsFloatArray()
    } catch (e: Throwable) {
      lastError = e.message ?: e.toString()
      null
    }
  }

  /**
   * 1분 신호를 타일링해 patient embedding 을 만든다.
   * @return pooled(정규화) + 윈도우별 임베딩(정규화). 실패 시 null.
   */
  fun encodePatient(signal: FloatArray): PatientEncoding? {
    val win = ModelConfig.CPC_WINDOW_SIZE
    if (signal.size < win) return null
    val hop = ModelConfig.CPC_EMBED_HOP.coerceAtLeast(1)
    // 학습(patient_encoder_cpc.forward)은 per-window pooled_embedding 을 그대로 mean-pool 한다
    // — L2 정규화 없음. 앱이 L2(단위벡터)를 걸면 AdaIN 이 OOD patient_vec 을 받아 포화(AF 오탐).
    // 따라서 raw 임베딩을 그대로 평균해 학습과 동일한 patient_vec 을 만든다.
    val perWindow = ArrayList<FloatArray>()
    var start = 0
    while (start + win <= signal.size) {
      val emb = embedWindow(signal.copyOfRange(start, start + win)) ?: return null
      perWindow.add(emb)
      start += hop
    }
    if (perWindow.isEmpty()) return null
    val dim = perWindow[0].size
    val pooled = FloatArray(dim)
    for (v in perWindow) for (d in 0 until dim) pooled[d] += v[d]
    for (d in 0 until dim) pooled[d] /= perWindow.size
    return PatientEncoding(pooled, perWindow)
  }

  /**
   * Occlusion saliency — baseline 대비 각 구간을 0으로 가릴 때 임베딩 코사인 감소량.
   * 값이 클수록 그 구간이 개인화 임베딩을 강하게 결정한다. 반환 길이 = window.size, 0..1 정규화.
   */
  fun occlusionSaliency(window: FloatArray, maskWidth: Int = 25, hop: Int = 12): FloatArray? {
    val base = embedWindow(window)?.let { l2normalize(it) } ?: return null
    val n = window.size
    val sal = FloatArray(n)
    val hitCount = IntArray(n)
    var pos = 0
    while (pos < n) {
      val end = (pos + maskWidth).coerceAtMost(n)
      val masked = window.copyOf()
      for (i in pos until end) masked[i] = 0f
      val e = embedWindow(masked)?.let { l2normalize(it) }
      if (e != null) {
        val drop = (1f - cosine(base, e)).coerceAtLeast(0f)
        for (i in pos until end) {
          sal[i] += drop
          hitCount[i]++
        }
      }
      pos += hop
    }
    var mx = 0f
    for (i in 0 until n) {
      if (hitCount[i] > 0) sal[i] /= hitCount[i]
      if (sal[i] > mx) mx = sal[i]
    }
    if (mx > 1e-8f) for (i in 0 until n) sal[i] /= mx
    return sal
  }

  private fun fitWindow(window: FloatArray): FloatArray {
    val sized = if (window.size == ModelConfig.CPC_WINDOW_SIZE) {
      window.copyOf()
    } else {
      SignalUtils.resampleToLen(window, ModelConfig.CPC_WINDOW_SIZE)
    }
    // InstanceNorm 이 stem 에 있어 스케일 불변이지만, 윈도우 간 비교를 위해 z-score 로 통일.
    return SignalUtils.zscore(sized)
  }

  private fun ensureNative() {
    if (!NativeLoader.isInitialized()) NativeLoader.init(SystemDelegate())
  }

  private fun copyAssetToFile(name: String): File {
    val outFile = File(context.filesDir, name)
    context.assets.open(name).use { input ->
      outFile.outputStream().use { output -> input.copyTo(output) }
    }
    return outFile
  }

  companion object {
    private const val TAG = "CpcRunner"

    fun l2normalize(v: FloatArray): FloatArray {
      var s = 0f
      for (x in v) s += x * x
      val norm = sqrt(s).coerceAtLeast(1e-8f)
      return FloatArray(v.size) { v[it] / norm }
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
      val n = minOf(a.size, b.size)
      var dot = 0f
      var na = 0f
      var nb = 0f
      for (i in 0 until n) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
      }
      return dot / (sqrt(na) * sqrt(nb) + 1e-8f)
    }

    fun rawNorm(v: FloatArray): Float {
      var s = 0f
      for (x in v) s += x * x
      return sqrt(s)
    }

    /**
     * effective rank — 윈도우별 임베딩 특이값 엔트로피의 exp.
     * 값이 낮으면 임베딩이 붕괴(collapse), 높으면 다양성 확보.
     * SVD 대신 Gram 행렬 고유값(power-iteration 없이 Jacobi) 근사.
     */
    fun effectiveRank(vectors: List<FloatArray>): Float {
      if (vectors.size < 2) return 1f
      val dim = vectors[0].size
      // 중심화
      val mean = FloatArray(dim)
      for (v in vectors) for (d in 0 until dim) mean[d] += v[d]
      for (d in 0 until dim) mean[d] /= vectors.size
      val centered = vectors.map { v -> FloatArray(dim) { v[it] - mean[it] } }
      // N×N Gram 행렬(윈도우 수는 작음) 고유값 = 특이값^2
      val n = centered.size
      val gram = Array(n) { i ->
        DoubleArray(n) { j ->
          var s = 0.0
          for (d in 0 until dim) s += centered[i][d] * centered[j][d]
          s
        }
      }
      val eig = jacobiEigenvalues(gram)
      val sv = eig.map { sqrt(max(0.0, it)) }
      val sum = sv.sum() + 1e-12
      var h = 0.0
      for (s in sv) {
        val p = s / sum
        if (p > 1e-12) h -= p * ln(p)
      }
      return kotlin.math.exp(h).toFloat()
    }

    /** 대칭행렬 Jacobi 고유값(작은 N 전용). */
    private fun jacobiEigenvalues(a0: Array<DoubleArray>, iters: Int = 60): DoubleArray {
      val n = a0.size
      val a = Array(n) { a0[it].copyOf() }
      for (sweep in 0 until iters) {
        var off = 0.0
        for (p in 0 until n) for (q in p + 1 until n) off += a[p][q] * a[p][q]
        if (off < 1e-12) break
        for (p in 0 until n) {
          for (q in p + 1 until n) {
            if (kotlin.math.abs(a[p][q]) < 1e-14) continue
            val app = a[p][p]
            val aqq = a[q][q]
            val apq = a[p][q]
            val phi = 0.5 * kotlin.math.atan2(2 * apq, aqq - app)
            val c = kotlin.math.cos(phi)
            val s = kotlin.math.sin(phi)
            for (k in 0 until n) {
              val akp = a[k][p]
              val akq = a[k][q]
              a[k][p] = c * akp - s * akq
              a[k][q] = s * akp + c * akq
            }
            for (k in 0 until n) {
              val apk = a[p][k]
              val aqk = a[q][k]
              a[p][k] = c * apk - s * aqk
              a[q][k] = s * apk + c * aqk
            }
          }
        }
      }
      return DoubleArray(n) { a[it][it] }
    }
  }
}

/** pooled patient embedding + 윈도우별 임베딩(모두 L2 정규화). */
data class PatientEncoding(
  val pooled: FloatArray,
  val perWindow: List<FloatArray>,
)
