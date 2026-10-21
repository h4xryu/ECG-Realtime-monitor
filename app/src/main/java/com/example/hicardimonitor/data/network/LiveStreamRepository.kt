package com.example.hicardimonitor.data.network

import android.content.Context
import com.example.hicardimonitor.ArrhythmiaConfig
import com.example.hicardimonitor.ModelConfig
import com.example.hicardimonitor.session.LiveStreamAggregator
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors

/**
 * Best-effort POST of a [LiveStreamAggregator.LivePayload] to /api/ecg/live. Silent on failure
 * (no retry, no UI) — a dropped live frame just means the dashboard's next poll shows slightly
 * stale data; it must never affect the on-device monitor.
 */
object LiveStreamRepository {
  private val bg = Executors.newSingleThreadExecutor { r -> Thread(r, "LiveStream").apply { isDaemon = true } }

  fun send(context: Context, payload: LiveStreamAggregator.LivePayload) {
    val token = SessionManager.getToken(context) ?: return
    bg.execute {
      val signalArray = JSONArray()
      for (v in payload.signal) signalArray.put(v.toDouble())

      val body = JSONObject().apply {
        put("ts", System.currentTimeMillis())
        put("fs", ModelConfig.FS_HZ.toInt())
        put("signal", signalArray)
        put("leadOff", payload.leadOff)
        put("signalQuality", payload.signalQuality.toDouble())
        put("bpm", payload.bpm)
        val beatProbs = payload.latestBeatProbs
        if (beatProbs != null) {
          val probsArray = JSONArray()
          for (p in beatProbs) probsArray.put(p.toDouble())
          val classesArray = JSONArray()
          for (name in ArrhythmiaConfig.CLASS_NAMES) classesArray.put(name)
          put(
            "latestBeat",
            JSONObject().apply {
              put("probs", probsArray)
              put("classes", classesArray)
            },
          )
        } else {
          put("latestBeat", JSONObject.NULL)
        }
      }

      // Fire-and-forget: ignore the result entirely, this must never surface to the UI.
      ApiClient.post(context, "ecg/live", body, token)
    }
  }
}
