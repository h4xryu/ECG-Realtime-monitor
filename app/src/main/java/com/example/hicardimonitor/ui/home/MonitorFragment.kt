package com.example.hicardimonitor.ui.home

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.example.hicardimonitor.ArrhythmiaConfig
import com.example.hicardimonitor.ArrhythmiaResult
import com.example.hicardimonitor.GraphSettings
import com.example.hicardimonitor.ModelConfig
import com.example.hicardimonitor.R
import com.example.hicardimonitor.ui.MainActivity
import com.example.hicardimonitor.ui.widget.ArrhythmiaBarView

class MonitorFragment : Fragment() {

  private lateinit var ecgView: com.example.hicardimonitor.EcgPlotView
  private lateinit var bpmOverlay: TextView
  private lateinit var banner: TextView
  private lateinit var gainEdit: EditText
  private lateinit var speedEdit: EditText
  private lateinit var gridCheck: CheckBox
  private lateinit var bpmCheck: CheckBox
  private lateinit var rPeakCheck: CheckBox
  private lateinit var aiPredClass: TextView
  private lateinit var aiPredProb: TextView
  private lateinit var aiBarView: ArrhythmiaBarView
  private lateinit var aiStatus: TextView

  private val arrhythmiaListener: (ArrhythmiaResult) -> Unit = { res ->
    if (isAdded) updateArrhythmia(res)
  }

  // onPlotBatch 는 단일 슬롯 — 해제 시 내 것일 때만 null 처리(측정 액티비티와의 순서 경합 방지).
  private val plotListener: (com.example.hicardimonitor.BleSampleBatch, IntArray) -> Unit = { batch, beatsRel ->
    if (isAdded) ecgView.appendSamples(batch.filtered, beatsRel, batch.gapBefore)
  }

  private val connListener: (Boolean) -> Unit = { connected ->
    if (isAdded) banner.visibility = if (connected) View.GONE else View.VISIBLE
  }
  private val bpmListener: (Int) -> Unit = { bpm ->
    if (isAdded && bpmCheck.isChecked) {
      bpmOverlay.text = "$bpm BPM"
      bpmOverlay.visibility = View.VISIBLE
    }
  }
  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?,
  ): View? = inflater.inflate(R.layout.fragment_monitor, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val coord = (requireActivity() as MainActivity).coordinator

    ecgView = view.findViewById(R.id.ecgView)
    bpmOverlay = view.findViewById(R.id.bpmOverlay)
    banner = view.findViewById(R.id.connectionBanner)
    gainEdit = view.findViewById(R.id.gainEdit)
    speedEdit = view.findViewById(R.id.speedEdit)
    gridCheck = view.findViewById(R.id.gridCheck)
    bpmCheck = view.findViewById(R.id.bpmCheck)
    rPeakCheck = view.findViewById(R.id.rPeakCheck)
    aiPredClass = view.findViewById(R.id.aiPredClass)
    aiPredProb = view.findViewById(R.id.aiPredProb)
    aiBarView = view.findViewById(R.id.aiBarView)
    aiStatus = view.findViewById(R.id.aiStatus)

    applyGraphSettings()

    val applyWatcher = object : TextWatcher {
      override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
      override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
      override fun afterTextChanged(s: Editable?) { applyGraphSettings() }
    }
    gainEdit.addTextChangedListener(applyWatcher)
    speedEdit.addTextChangedListener(applyWatcher)

    val checkboxListener = { _: android.widget.CompoundButton, _: Boolean -> applyGraphSettings() }
    gridCheck.setOnCheckedChangeListener(checkboxListener)
    bpmCheck.setOnCheckedChangeListener { _, checked ->
      bpmOverlay.visibility = if (checked) View.VISIBLE else View.GONE
    }
    rPeakCheck.setOnCheckedChangeListener(checkboxListener)

    bpmOverlay.visibility = if (bpmCheck.isChecked) View.VISIBLE else View.GONE
  }

  // 리스너·플롯 구독은 onResume/onPause 로 — ViewPager2 에서 다른 탭이 앞에 있는 동안
  // 오프스크린 모니터가 배치마다 invalidate/레이아웃을 돌려 메인 스레드를 잡아먹지 않게 한다.
  override fun onResume() {
    super.onResume()
    val coord = (requireActivity() as MainActivity).coordinator
    coord.addConnectionListener(connListener)
    coord.addBpmListener(bpmListener)
    coord.addArrhythmiaListener(arrhythmiaListener)
    coord.lastArrhythmia?.let { updateArrhythmia(it) } ?: refreshAiStatus()
    coord.onPlotBatch = plotListener
    coord.setContinuousMonitor()
    banner.visibility = if (coord.bleSource.isConnected) View.GONE else View.VISIBLE
  }

  override fun onPause() {
    val coord = (activity as? MainActivity)?.coordinator
    if (coord != null) {
      coord.removeConnectionListener(connListener)
      coord.removeBpmListener(bpmListener)
      coord.removeArrhythmiaListener(arrhythmiaListener)
      if (coord.onPlotBatch === plotListener) coord.onPlotBatch = null
    }
    super.onPause()
  }

  private fun updateArrhythmia(res: ArrhythmiaResult) {
    // 서맥은 모델 클래스가 아닌 BPM rule 판정 — 활성 시 헤드라인만 서맥으로 표기하고
    // 막대는 모델 5클래스를 그대로 보여준다.
    if (res.bradycardia) {
      aiPredClass.text = ArrhythmiaConfig.BRADY_LABEL_KO
      aiPredProb.text = "(BPM 기준 판정)"
    } else {
      aiPredClass.text = res.topLabelKo
      aiPredProb.text = "(추론확률 %.1f%%)".format(res.topProb * 100f)
    }
    aiBarView.setData(res.probabilities, res.topIndex)
    aiStatus.visibility = View.GONE
  }

  private fun refreshAiStatus() {
    val coord = (activity as? MainActivity)?.coordinator ?: return
    aiBarView.clear()
    val msg = when {
      !coord.enableInference -> R.string.ai_status_disabled
      !coord.arrhythmiaRunner.isLoaded && coord.arrhythmiaRunner.lastError != null ->
        R.string.ai_status_no_model
      else -> R.string.ai_status_waiting
    }
    aiStatus.setText(msg)
    aiStatus.visibility = View.VISIBLE
  }

  private fun applyGraphSettings() {
    if (!::ecgView.isInitialized) return
    // 임상 캘리브레이션: 게인(mm/mV)·속도(mm/s) 고정 스케일. 가시 창(초)은 뷰가 폭에서 산출.
    val gain = gainEdit.text.toString().toFloatOrNull()?.coerceIn(2.5f, 80f) ?: 20f
    val speed = speedEdit.text.toString().toFloatOrNull()?.coerceIn(6.25f, 100f) ?: 25f
    ecgView.settings = ecgView.settings.copy(
      calibrated = true,
      mmPerMv = gain,
      mmPerSec = speed,
      sampleRateHz = ModelConfig.FS_HZ,
      showGrid = gridCheck.isChecked,
      showRPeak = rPeakCheck.isChecked,
      showCalMarker = true,
      showScaleReadout = true,
      showBattery = true,
    )
    ecgView.invalidate()
  }

  override fun onDestroyView() {
    val coord = (activity as? MainActivity)?.coordinator
    coord?.removeConnectionListener(connListener)
    coord?.removeBpmListener(bpmListener)
    coord?.removeArrhythmiaListener(arrhythmiaListener)
    if (coord?.onPlotBatch === plotListener) coord.onPlotBatch = null
    super.onDestroyView()
  }
}
