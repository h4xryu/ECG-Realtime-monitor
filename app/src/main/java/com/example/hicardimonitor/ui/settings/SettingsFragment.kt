package com.example.hicardimonitor.ui.settings

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.example.hicardimonitor.R
import com.example.hicardimonitor.ui.MainActivity

class SettingsFragment : Fragment() {

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
  }
}
