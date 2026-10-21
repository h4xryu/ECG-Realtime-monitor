package com.example.hicardimonitor.data.network

import android.content.Context
import android.os.Handler
import android.os.Looper
import org.json.JSONObject
import java.util.concurrent.Executors

/**
 * Signup/login against the backend's /api/auth routes; persists the resulting
 * session via SessionManager. Network calls run on a background executor; callbacks
 * are delivered on the main thread.
 */
object AuthRepository {
  private val bg = Executors.newSingleThreadExecutor { r -> Thread(r, "AuthRepository").apply { isDaemon = true } }
  private val mainHandler = Handler(Looper.getMainLooper())

  fun signUp(context: Context, username: String, password: String, onResult: (ApiResult<String>) -> Unit) {
    bg.execute {
      val body = JSONObject().apply {
        put("username", username)
        put("password", password)
      }
      deliver(context, ApiClient.post(context, "auth/signup", body), onResult)
    }
  }

  fun login(context: Context, username: String, password: String, onResult: (ApiResult<String>) -> Unit) {
    bg.execute {
      val body = JSONObject().apply {
        put("username", username)
        put("password", password)
      }
      deliver(context, ApiClient.post(context, "auth/login", body), onResult)
    }
  }

  /** Confirms the stored token is still accepted by the server; clears it locally if not. */
  fun validateSession(context: Context, onResult: (Boolean) -> Unit) {
    val token = SessionManager.getToken(context)
    if (token == null) {
      mainHandler.post { onResult(false) }
      return
    }
    bg.execute {
      val valid = ApiClient.get(context, "auth/me", token) is ApiResult.Success
      if (!valid) SessionManager.clear(context)
      mainHandler.post { onResult(valid) }
    }
  }

  fun logout(context: Context) {
    SessionManager.clear(context)
  }

  private fun deliver(
    context: Context,
    result: ApiResult<JSONObject>,
    onResult: (ApiResult<String>) -> Unit,
  ) {
    when (result) {
      is ApiResult.Success -> {
        val token = result.value.optString("token")
        val username = result.value.optJSONObject("user")?.optString("username").orEmpty()
        if (token.isNotEmpty()) {
          SessionManager.saveSession(context, token, username)
          mainHandler.post { onResult(ApiResult.Success(username)) }
        } else {
          mainHandler.post { onResult(ApiResult.Failure("서버 응답이 올바르지 않습니다")) }
        }
      }
      is ApiResult.Failure -> mainHandler.post { onResult(result) }
    }
  }
}
