package com.example.hicardimonitor.ui.auth

import android.app.Activity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.hicardimonitor.R
import com.example.hicardimonitor.data.network.ApiResult
import com.example.hicardimonitor.data.network.AuthRepository

/** Creates a new account. Finishes with RESULT_OK once a session is saved (signup logs in). */
class SignupActivity : AppCompatActivity() {

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_signup)

    val usernameEdit = findViewById<EditText>(R.id.usernameEdit)
    val passwordEdit = findViewById<EditText>(R.id.passwordEdit)
    val passwordConfirmEdit = findViewById<EditText>(R.id.passwordConfirmEdit)
    val errorText = findViewById<TextView>(R.id.signupErrorText)
    val submitButton = findViewById<Button>(R.id.signupSubmitButton)

    submitButton.setOnClickListener {
      val username = usernameEdit.text.toString().trim()
      val password = passwordEdit.text.toString()
      val passwordConfirm = passwordConfirmEdit.text.toString()

      if (password != passwordConfirm) {
        errorText.text = getString(R.string.signup_password_mismatch)
        errorText.visibility = View.VISIBLE
        return@setOnClickListener
      }
      errorText.visibility = View.GONE
      submitButton.isEnabled = false

      AuthRepository.signUp(this, username, password) { result ->
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
