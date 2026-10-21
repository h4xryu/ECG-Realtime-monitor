package com.example.hicardimonitor.data.network

import android.content.Context

/**
 * Stores the signed-in session (JWT + username) in a private SharedPreferences file.
 *
 * Plain SharedPreferences, not EncryptedSharedPreferences — acceptable for this MVP
 * since the token only grants access to this one user's own uploaded ECG metadata on
 * the dev backend, but switch to androidx.security EncryptedSharedPreferences before
 * this ships against real patient data.
 */
object SessionManager {
  private const val PREFS = "hicardi_session"
  private const val KEY_TOKEN = "token"
  private const val KEY_USERNAME = "username"

  private fun prefs(context: Context) =
    context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

  fun saveSession(context: Context, token: String, username: String) {
    prefs(context).edit()
      .putString(KEY_TOKEN, token)
      .putString(KEY_USERNAME, username)
      .apply()
  }

  fun getToken(context: Context): String? = prefs(context).getString(KEY_TOKEN, null)

  fun getUsername(context: Context): String? = prefs(context).getString(KEY_USERNAME, null)

  fun isLoggedIn(context: Context): Boolean = getToken(context) != null

  fun clear(context: Context) {
    prefs(context).edit().clear().apply()
  }
}
