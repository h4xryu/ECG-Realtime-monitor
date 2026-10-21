package com.example.hicardimonitor.data.network

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.example.hicardimonitor.data.DiagnosisRecord
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors

/** Uploads a locally-saved DiagnosisRecord's signal + metadata to /api/ecg/upload. */
object EcgUploadRepository {
  // Mirrors the server-side cap in app/api/ecg/upload/route.ts (UploadSchema.signal.max).
  private const val MAX_SAMPLES = 2_000_000

  private val bg = Executors.newSingleThreadExecutor { r -> Thread(r, "EcgUpload").apply { isDaemon = true } }
  private val mainHandler = Handler(Looper.getMainLooper())

  fun upload(context: Context, record: DiagnosisRecord, onResult: (ApiResult<Unit>) -> Unit) {
    val token = SessionManager.getToken(context)
    if (token == null) {
      mainHandler.post { onResult(ApiResult.Failure("로그인이 필요합니다")) }
      return
    }

    bg.execute {
      val signal = record.mergedFiltered()
      if (signal.size > MAX_SAMPLES) {
        mainHandler.post {
          onResult(ApiResult.Failure("레코딩이 너무 길어서 한 번에 업로드할 수 없습니다 (${signal.size} samples)"))
        }
        return@execute
      }

      val signalArray = JSONArray()
      for (v in signal) signalArray.put(v.toDouble())

      val body = JSONObject().apply {
        // Lets the server de-duplicate a retried upload of this same local record (WorkManager
        // retry queue, for example) instead of saving it twice.
        put("clientRecordId", record.id)
        put("deviceId", JSONObject.NULL)
        put("recordedAt", record.startedAtMs)
        put("fs", record.fs)
        put("bpm", record.bpm)
        put("summary", record.summary)
        put("signal", signalArray)
      }

      val result = ApiClient.post(context, "ecg/upload", body, token)
      val mapped: ApiResult<Unit> = when (result) {
        is ApiResult.Success -> ApiResult.Success(Unit)
        is ApiResult.Failure -> result
      }
      mainHandler.post { onResult(mapped) }
    }
  }
}
