package com.example.hicardimonitor

import android.content.Context
import android.util.Log
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import kotlin.math.exp

/**
 * 6-class ResUMamba 분류기 ExecuTorch 러너.
 *
 * 계약: forward(x[1,1,501], patient_vec[1,256], clinical[1,32]) → logits[1,6]
 *   - x 는 raw mV (z-score/밴드패스 없음).
 *   - sigmoid → 멀티라벨, argmax → 대표 클래스.
 *
 * classifier.pte 가 assets 에 없으면 isLoaded=false 로 남고 추론은 조용히 건너뛴다.
 */
class ArrhythmiaClassifierRunner(private val context: Context) {

  private var module: Module? = null
  var lastError: String? = null
    private set

  val isLoaded: Boolean get() = module != null

  fun load(): Boolean {
    return try {
      ensureNative()
      val asset = ArrhythmiaConfig.CLASSIFIER_PTE_FILE
      if (!assetExists(asset)) {
        lastError = "assets/$asset 없음 (resumamba 환경에서 export 후 배치)"
        Log.w(TAG, lastError!!)
        return false
      }
      val path = copyAssetToFile(asset).absolutePath
      module = Module.load(path)
      Log.i(TAG, "classifier load OK: $path")
      true
    } catch (e: Throwable) {
      module = null
      lastError = e.message ?: e.toString()
      Log.e(TAG, "classifier load FAIL: $lastError", e)
      false
    }
  }

  /**
   * 단일 비트 추론.
   * @param beatMv     501 샘플 raw mV crop.
   * @param patientVec 256-dim CPC patient embedding (null 이면 0-vector = 개인화 미적용).
   * @param clinical   32-dim clinical feats.
   */
  fun infer(beatMv: FloatArray, patientVec: FloatArray?, clinical: FloatArray): ArrhythmiaResult? {
    val m = module ?: return null
    return try {
      val x = fit(beatMv, ArrhythmiaConfig.BEAT_LEN)
      val p = fit(patientVec ?: FloatArray(ArrhythmiaConfig.PATIENT_DIM), ArrhythmiaConfig.PATIENT_DIM)
      val c = fit(clinical, ArrhythmiaConfig.CLINICAL_DIM)

      val xt = Tensor.fromBlob(x, longArrayOf(1, 1, ArrhythmiaConfig.BEAT_LEN.toLong()))
      val pt = Tensor.fromBlob(p, longArrayOf(1, ArrhythmiaConfig.PATIENT_DIM.toLong()))
      val ct = Tensor.fromBlob(c, longArrayOf(1, ArrhythmiaConfig.CLINICAL_DIM.toLong()))

      val t0 = System.nanoTime()
      val out = m.forward(EValue.from(xt), EValue.from(pt), EValue.from(ct))
      val latencyMs = (System.nanoTime() - t0) / 1_000_000f

      val logits = out[0].toTensor().getDataAsFloatArray()
      val probs = FloatArray(logits.size) { 1f / (1f + exp(-logits[it])) }
      ArrhythmiaResult.from(probs, latencyMs)
    } catch (e: Throwable) {
      lastError = e.message ?: e.toString()
      Log.e(TAG, "classifier infer FAIL: $lastError", e)
      null
    }
  }

  fun close() {
    try {
      module?.destroy()
    } catch (_: Throwable) {
    }
    module = null
  }

  /** 길이를 target 으로 맞춤(부족하면 0 패딩, 초과하면 절단). 정상 경로에선 그대로 통과. */
  private fun fit(v: FloatArray, target: Int): FloatArray {
    if (v.size == target) return v
    val out = FloatArray(target)
    System.arraycopy(v, 0, out, 0, minOf(v.size, target))
    return out
  }

  private fun ensureNative() {
    if (!NativeLoader.isInitialized()) NativeLoader.init(SystemDelegate())
  }

  private fun assetExists(name: String): Boolean = try {
    context.assets.open(name).use { true }
  } catch (_: Throwable) {
    false
  }

  private fun copyAssetToFile(name: String): File {
    val outFile = File(context.filesDir, name)
    context.assets.open(name).use { input ->
      outFile.outputStream().use { output -> input.copyTo(output) }
    }
    return outFile
  }

  companion object {
    private const val TAG = "ArrhythmiaRunner"
  }
}

/**
 * 멀티라벨 추론 결과.
 * @param probabilities 표시 클래스별 sigmoid 확률(size = [ArrhythmiaConfig.NUM_CLASSES], 서맥 제외).
 *                      러너 직후에는 모델 로짓 그대로(6)일 수 있고, 엔진 후처리에서 5 로 매핑된다.
 * @param topIndex      argmax 인덱스(대표 예측 클래스).
 * @param activeIndices threshold 이상인 클래스들.
 * @param bradycardia   BPM rule 서맥 판정(모델과 무관, 엔진 후처리에서 설정).
 */
class ArrhythmiaResult(
  val probabilities: FloatArray,
  val topIndex: Int,
  val activeIndices: List<Int>,
  val latencyMs: Float,
  val bradycardia: Boolean = false,
) {
  val topName: String get() = ArrhythmiaConfig.name(topIndex)
  val topLabelKo: String get() = ArrhythmiaConfig.labelKo(topIndex)
  val topProb: Float get() = probabilities.getOrElse(topIndex) { 0f }

  companion object {
    fun from(probs: FloatArray, latencyMs: Float, bradycardia: Boolean = false): ArrhythmiaResult {
      var top = 0
      for (i in probs.indices) if (probs[i] > probs[top]) top = i
      val active = probs.indices.filter { probs[it] >= ArrhythmiaConfig.THRESHOLD }
      return ArrhythmiaResult(probs, top, active, latencyMs, bradycardia)
    }
  }
}
