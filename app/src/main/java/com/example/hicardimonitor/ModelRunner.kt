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
 * HiCardi ExecuTorch 추론 — beat window [1,1,W] + optional calib [1,NUM_CALIB,1,500].
 * CPC 임베딩은 calib 윈도우 forward 결과를 평균해 조건화에 사용.
 *
 * 레거시 경로다. 6-class 개인화는 CpcRunner + ArrhythmiaClassifierRunner 가 담당하며,
 * 이쪽 model.pte 가 현재 NUM_CALIB(59)과 다른 calib 개수로 export 되었을 수 있으므로
 * calib forward 가 shape 로 실패하면 무조건부 forward 로 폴백한다(추론이 죽지 않게).
 */
class ModelRunner(private val context: Context) {

  private var module: Module? = null
  var lastError: String? = null
    private set

  val isLoaded: Boolean
    get() = module != null

  fun load(): Boolean {
    return try {
      ensureNative()
      val path = copyAssetToFile(ModelConfig.PTE_FILE).absolutePath
      module = Module.load(path)
      Log.i(TAG, "load OK: $path")
      true
    } catch (e: Throwable) {
      module = null
      lastError = e.message ?: e.toString()
      Log.e(TAG, "load FAIL: $lastError", e)
      false
    }
  }

  fun infer(
    waveform: FloatArray,
    calibrationWindows: Array<FloatArray>? = null,
    @Suppress("UNUSED_PARAMETER") patientEmbedding: FloatArray? = null,
  ): InferenceResult? {
    val m = module ?: return null
    return try {
      val input = preprocess(waveform)
      val xTensor = Tensor.fromBlob(input, longArrayOf(1, 1, ModelConfig.WINDOW_SIZE.toLong()))
      if (calibrationWindows != null && calibrationWindows.size == ModelConfig.NUM_CALIB) {
        val calibFlat = FloatArray(ModelConfig.NUM_CALIB * ModelConfig.CPC_WINDOW_SIZE)
        calibrationWindows.forEachIndexed { i, w ->
          fitCalibWindow(w).copyInto(calibFlat, i * ModelConfig.CPC_WINDOW_SIZE)
        }
        val calibTensor = Tensor.fromBlob(
          calibFlat,
          longArrayOf(1, ModelConfig.NUM_CALIB.toLong(), 1, ModelConfig.CPC_WINDOW_SIZE.toLong()),
        )
        val t0 = System.nanoTime()
        try {
          val out = m.forward(EValue.from(xTensor), EValue.from(calibTensor))
          inferFromOutputs(out, t0, false)
        } catch (calibErr: Throwable) {
          // model.pte 가 이 calib 개수를 안 받으면(shape mismatch) 무조건부 forward 로 폴백.
          if (!calibShapeWarned) {
            calibShapeWarned = true
            Log.w(TAG, "calib forward failed (calib=${ModelConfig.NUM_CALIB}), falling back to no-calib: ${calibErr.message}")
          }
          val out = m.forward(EValue.from(xTensor))
          inferFromOutputs(out, System.nanoTime(), true)
        }
      } else {
        val t0 = System.nanoTime()
        val out = m.forward(EValue.from(xTensor))
        inferFromOutputs(out, t0, false)
      }
    } catch (e: Throwable) {
      lastError = e.message ?: e.toString()
      null
    }
  }

  @Volatile private var calibShapeWarned = false

  /**
   * 16개 CPC 윈도우로 patient embedding 추출 (온디바이스 forward만).
   * 출력 텐서가 128-dim이면 직접 사용, 아니면 beat logits 평균을 fallback.
   */
  fun extractPatientEmbedding(calibrationWindows: Array<FloatArray>): FloatArray? {
    val m = module ?: return null
    return try {
      val calibFlat = FloatArray(ModelConfig.NUM_CALIB * ModelConfig.CPC_WINDOW_SIZE)
      calibrationWindows.forEachIndexed { i, w ->
        fitCalibWindow(w).copyInto(calibFlat, i * ModelConfig.CPC_WINDOW_SIZE)
      }
      val calibTensor = Tensor.fromBlob(
        calibFlat,
        longArrayOf(1, ModelConfig.NUM_CALIB.toLong(), 1, ModelConfig.CPC_WINDOW_SIZE.toLong()),
      )
      val dummy = FloatArray(ModelConfig.WINDOW_SIZE)
      val xTensor = Tensor.fromBlob(dummy, longArrayOf(1, 1, ModelConfig.WINDOW_SIZE.toLong()))
      val outputs = try {
        m.forward(EValue.from(calibTensor))
      } catch (_: Throwable) {
        m.forward(EValue.from(xTensor), EValue.from(calibTensor))
      }
      val data = outputs[0].toTensor().getDataAsFloatArray()
      when {
        data.size == ModelConfig.PATIENT_EMBEDDING_DIM -> data
        data.size > ModelConfig.PATIENT_EMBEDDING_DIM -> {
          val perWin = data.size / ModelConfig.NUM_CALIB
          FloatArray(ModelConfig.PATIENT_EMBEDDING_DIM) { d ->
            var sum = 0f
            for (i in 0 until ModelConfig.NUM_CALIB) sum += data[i * perWin + d]
            sum / ModelConfig.NUM_CALIB
          }
        }
        else -> {
          FloatArray(ModelConfig.PATIENT_EMBEDDING_DIM) { i ->
            data.getOrElse(i % data.size) { 0f }
          }
        }
      }
    } catch (e: Throwable) {
      lastError = e.message ?: e.toString()
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

  private fun inferFromOutputs(outputs: Array<EValue>, t0: Long, usedFallback: Boolean): InferenceResult {
    val latencyMs = (System.nanoTime() - t0) / 1_000_000f
    val logits = outputs[0].toTensor().getDataAsFloatArray()
    val probs = decodeProbabilities(logits)
    val top = probs.indices.maxByOrNull { probs[it] } ?: 0
    return InferenceResult(
      className = ModelConfig.CLASS_NAMES.getOrElse(top) { "?" },
      classIndex = top,
      confidence = probs[top],
      probabilities = probs,
      latencyMs = latencyMs,
      usedFallback = usedFallback,
      patientEmbedding = if (outputs.size > 1) {
        outputs[1].toTensor().getDataAsFloatArray()
      } else {
        null
      },
    )
  }

  /**
   * 학습 파이프라인(cpc_ecg data.transforms)과 동일한 모델 입력 전처리.
   * 250 Hz 리샘플(BLE는 이미 250 Hz) + 윈도우별 z-score. 밴드패스/노치 없음.
   * (표시용 필터 OnlineFilter 와 분리 — 모델은 raw ECG 를 받는다.)
   */
  private fun preprocess(waveform: FloatArray): FloatArray {
    val sized = if (waveform.size != ModelConfig.WINDOW_SIZE) {
      SignalUtils.resampleToLen(waveform, ModelConfig.WINDOW_SIZE)
    } else {
      waveform.copyOf()
    }
    return SignalUtils.zscore(sized)
  }

  /** CPC 캘리브레이션 2 s 윈도우: 500 샘플 맞춤 + 윈도우별 z-score (학습과 동일). */
  private fun fitCalibWindow(w: FloatArray): FloatArray {
    val sized = if (w.size == ModelConfig.CPC_WINDOW_SIZE) {
      w
    } else {
      SignalUtils.resampleToLen(w, ModelConfig.CPC_WINDOW_SIZE)
    }
    return SignalUtils.zscore(sized)
  }

  private fun decodeProbabilities(logits: FloatArray): FloatArray {
    return when (ModelConfig.OUTPUT_MODE) {
      ModelConfig.OutputMode.SOFTMAX -> softmax(logits)
      ModelConfig.OutputMode.SIGMOID -> FloatArray(logits.size) { i -> 1f / (1f + exp(-logits[i])) }
    }
  }

  private fun softmax(logits: FloatArray): FloatArray {
    val max = logits.maxOrNull() ?: 0f
    val exps = FloatArray(logits.size) { i -> exp(logits[i] - max) }
    val sum = exps.sum()
    return FloatArray(logits.size) { i -> exps[i] / sum }
  }

  private fun ensureNative() {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(SystemDelegate())
    }
  }

  private fun copyAssetToFile(name: String): File {
    val outFile = File(context.filesDir, name)
    context.assets.open(name).use { input ->
      outFile.outputStream().use { output -> input.copyTo(output) }
    }
    return outFile
  }

  companion object {
    private const val TAG = "ModelRunner"
  }
}

data class InferenceResult(
  val className: String,
  val classIndex: Int,
  val confidence: Float,
  val probabilities: FloatArray,
  val latencyMs: Float,
  val usedFallback: Boolean,
  val patientEmbedding: FloatArray? = null,
) {
  val topClass: String get() = className
}
