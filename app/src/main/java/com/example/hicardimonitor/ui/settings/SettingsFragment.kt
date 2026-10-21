package com.example.hicardimonitor.ui.settings

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import com.example.hicardimonitor.R
import com.example.hicardimonitor.data.network.ApiConfig
import com.example.hicardimonitor.data.network.AuthRepository
import com.example.hicardimonitor.data.network.SessionManager
import com.example.hicardimonitor.ui.MainActivity
import com.example.hicardimonitor.ui.auth.LoginActivity
import com.example.hicardimonitor.ui.auth.SignupActivity

class SettingsFragment : Fragment() {

  private val loginLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
    refreshAccountUi()
  }
  private val signupLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
    refreshAccountUi()
  }

  private var accountStatusText: TextView? = null

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?,
  ): View? = inflater.inflate(R.layout.fragment_settings, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val coord = (requireActivity() as MainActivity).coordinator
    val modelStatus = view.findViewById<TextView>(R.id.modelStatusText)
    val classifierStatus = view.findViewById<TextView>(R.id.classifierStatusText)
    val inferenceCheck = view.findViewById<CheckBox>(R.id.inferenceCheck)
    val personalizationCheck = view.findViewById<CheckBox>(R.id.personalizationCheck)

    modelStatus.text = if (coord.modelRunner.isLoaded) {
      "model.pte: OK"
    } else {
      "model.pte: FAIL — ${coord.modelRunner.lastError ?: "missing assets/model.pte"}"
    }
    classifierStatus.text = if (coord.arrhythmiaRunner.isLoaded) {
      "classifier.pte: OK (6-class)"
    } else {
      "classifier.pte: 미로드 — ${coord.arrhythmiaRunner.lastError ?: "assets/classifier.pte 필요"}"
    }

    inferenceCheck.isChecked = coord.enableInference
    inferenceCheck.setOnCheckedChangeListener { _, checked ->
      coord.enableInference = checked
    }
    personalizationCheck.isChecked = coord.applyPersonalization
    personalizationCheck.setOnCheckedChangeListener { _, checked ->
      coord.applyPersonalization = checked
    }

    accountStatusText = view.findViewById(R.id.accountStatusText)
    val serverUrlEdit = view.findViewById<EditText>(R.id.serverUrlEdit)
    val loginButton = view.findViewById<Button>(R.id.loginButton)
    val signupButton = view.findViewById<Button>(R.id.signupButton)
    val logoutButton = view.findViewById<Button>(R.id.logoutButton)

    serverUrlEdit.setText(ApiConfig.getBaseUrl(requireContext()))
    serverUrlEdit.setOnFocusChangeListener { _, hasFocus ->
      if (!hasFocus) {
        val url = serverUrlEdit.text.toString()
        if (url.isNotBlank()) ApiConfig.setBaseUrl(requireContext(), url)
      }
    }

    loginButton.setOnClickListener {
      loginLauncher.launch(android.content.Intent(requireContext(), LoginActivity::class.java))
    }
    signupButton.setOnClickListener {
      signupLauncher.launch(android.content.Intent(requireContext(), SignupActivity::class.java))
    }
    logoutButton.setOnClickListener {
      AuthRepository.logout(requireContext())
      refreshAccountUi()
    }

    refreshAccountUi()
  }

  private fun refreshAccountUi() {
    val view = view ?: return
    val context = context ?: return
    val loginButton = view.findViewById<Button>(R.id.loginButton)
    val signupButton = view.findViewById<Button>(R.id.signupButton)
    val logoutButton = view.findViewById<Button>(R.id.logoutButton)

    val loggedIn = SessionManager.isLoggedIn(context)
    accountStatusText?.text = if (loggedIn) {
      getString(R.string.cloud_account_logged_in, SessionManager.getUsername(context))
    } else {
      getString(R.string.cloud_account_logged_out)
    }
    loginButton.visibility = if (loggedIn) View.GONE else View.VISIBLE
    signupButton.visibility = if (loggedIn) View.GONE else View.VISIBLE
    logoutButton.visibility = if (loggedIn) View.VISIBLE else View.GONE
  }
}
