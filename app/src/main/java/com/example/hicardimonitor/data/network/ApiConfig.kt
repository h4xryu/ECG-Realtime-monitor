package com.example.hicardimonitor.data.network

import android.content.Context

/**
 * Backend base URL, editable at runtime instead of hardcoded at build time — the dev
 * server's reachable address changes depending on emulator vs. `adb reverse` vs. LAN.
 * Must end with "/" and point at the Next.js app's "/api" prefix, e.g.
 * "http://10.0.2.2:3000/api/".
 */
object ApiConfig {
  private const val PREFS = "hicardi_api_config"
  private const val KEY_BASE_URL = "base_url"

  // Android emulator's alias for the host machine's localhost.
  const val DEFAULT_BASE_URL = "http://10.0.2.2:3000/api/"

  private fun prefs(context: Context) =
    context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

  fun getBaseUrl(context: Context): String =
    prefs(context).getString(KEY_BASE_URL, DEFAULT_BASE_URL) ?: DEFAULT_BASE_URL

  fun setBaseUrl(context: Context, url: String) {
    val trimmed = url.trim()
    val normalized = if (trimmed.endsWith("/")) trimmed else "$trimmed/"
    prefs(context).edit().putString(KEY_BASE_URL, normalized).apply()
  }
}
