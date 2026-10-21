package com.example.hicardimonitor.data.network

import android.content.Context
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

sealed class ApiResult<out T> {
  data class Success<T>(val value: T) : ApiResult<T>()
  data class Failure(val message: String, val httpCode: Int? = null) : ApiResult<Nothing>()
}

/** Thin JSON/HTTP helper over OkHttp for the backend under ApiConfig.getBaseUrl(). */
object ApiClient {
  private val JSON = "application/json; charset=utf-8".toMediaType()

  private val client = OkHttpClient.Builder()
    .connectTimeout(10, TimeUnit.SECONDS)
    .readTimeout(30, TimeUnit.SECONDS)
    .writeTimeout(30, TimeUnit.SECONDS)
    .build()

  /** Runs on the calling thread — callers must invoke this off the main thread. */
  fun post(context: Context, path: String, body: JSONObject, authToken: String? = null): ApiResult<JSONObject> =
    execute(buildRequest(context, path, body, authToken))

  /** Runs on the calling thread — callers must invoke this off the main thread. */
  fun get(context: Context, path: String, authToken: String? = null): ApiResult<JSONObject> =
    execute(buildRequest(context, path, null, authToken))

  private fun buildRequest(context: Context, path: String, body: JSONObject?, authToken: String?): Request {
    val url = ApiConfig.getBaseUrl(context) + path
    val builder = Request.Builder().url(url)
    if (authToken != null) builder.addHeader("Authorization", "Bearer $authToken")
    if (body != null) {
      builder.post(body.toString().toRequestBody(JSON))
    } else {
      builder.get()
    }
    return builder.build()
  }

  private fun execute(request: Request): ApiResult<JSONObject> {
    return try {
      client.newCall(request).execute().use { response ->
        val text = response.body?.string().orEmpty()
        val json = if (text.isNotBlank()) JSONObject(text) else JSONObject()
        if (response.isSuccessful) {
          ApiResult.Success(json)
        } else {
          val message = json.optString("error").ifBlank { "요청이 실패했습니다 (${response.code})" }
          ApiResult.Failure(message, response.code)
        }
      }
    } catch (e: IOException) {
      ApiResult.Failure("서버에 연결할 수 없습니다: ${e.message}")
    } catch (e: Exception) {
      ApiResult.Failure("알 수 없는 오류: ${e.message}")
    }
  }
}
