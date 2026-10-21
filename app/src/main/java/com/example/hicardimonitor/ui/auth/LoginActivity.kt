package com.example.hicardimonitor.ui.auth

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.hicardimonitor.R
import com.example.hicardimonitor.data.network.ApiResult
import com.example.hicardimonitor.data.network.AuthRepository

/** Logs an existing account in. Finishes with RESULT_OK once a session is saved. */
class LoginActivity : AppCompatActivity() {

  private val signupLauncher = registerForActivityResult(
    ActivityResultContracts.StartActivityForResult(),
  ) { result ->
    if (result.resultCode == Activity.RESULT_OK) {
      setResult(Activity.RESULT_OK)
      finish()
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_login)

    val usernameEdit = findViewById<EditText>(R.id.usernameEdit)
    val passwordEdit = findViewById<EditText>(R.id.passwordEdit)
    val errorText = findViewById<TextView>(R.id.loginErrorText)
    val submitButton = findViewById<Button>(R.id.loginSubmitButton)
    val goSignupText = findViewById<TextView>(R.id.goSignupText)

    goSignupText.setOnClickListener {
      signupLauncher.launch(Intent(this, SignupActivity::class.java))
    }

    submitButton.setOnClickListener {
      val username = usernameEdit.text.toString().trim()
      val password = passwordEdit.text.toString()
      if (username.isEmpty() || password.isEmpty()) {
        errorText.text = getString(R.string.login_error_blank)
        errorText.visibility = View.VISIBLE
        return@setOnClickListener
      }
      errorText.visibility = View.GONE
      submitButton.isEnabled = false

      AuthRepository.login(this, username, password) { result ->
        submitButton.isEnabled = true
        when (result) {
          is ApiResult.Success -> {
            setResult(Activity.RESULT_OK)
            finish()
          }
          is ApiResult.Failure -> {
            errorText.text = result.message
            errorText.visibility = View.VISIBLE
          }
        }
      }
    }
  }
}
