package com.example.hicardimonitor.ui.measurement

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageButton
import android.widget.RadioButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.example.hicardimonitor.EcgPlotView
import com.example.hicardimonitor.GraphSettings
import com.example.hicardimonitor.ModelConfig
import com.example.hicardimonitor.R
import com.example.hicardimonitor.data.DiagnosisRecord
import com.example.hicardimonitor.data.RecordMatActions
import com.example.hicardimonitor.data.RecordingExporter
import com.example.hicardimonitor.session.CpcCalibrationManager
import com.example.hicardimonitor.session.RecordingManager
import com.example.hicardimonitor.ui.MainActivity
import com.example.hicardimonitor.ui.record.RecordDetailActivity
import com.example.hicardimonitor.ui.widget.TickTimelineView
import java.io.File
import java.util.concurrent.Executors

class EcgMeasurementActivity : AppCompatActivity() {

  private val coordinator get() = MainActivity.requireCoordinator()

  private lateinit var countText: TextView
  private lateinit var timeline: TickTimelineView
  private lateinit var ecgView: EcgPlotView
  private lateinit var bpmText: TextView
  private var totalSec = 60
  private var mode: String = MODE_RECORDING

  private val bg = Executors.newSingleThreadExecutor { r ->
    Thread(r, "RecordingExport").apply { isDaemon = true }
  }
  private var pendingExportFile: File? = null

  private val bpmListener: (Int) -> Unit = { bpm ->
    runOnUiThread { bpmText.text = "$bpm BPM" }
  }

  // onPlotBatch 는 단일 슬롯 — 정리 시 내 것일 때만 null (모니터 프래그먼트와의 순서 경합 방지).
  private val plotListener: (com.example.hicardimonitor.BleSampleBatch, IntArray) -> Unit = { batch, beatsRel ->
    ecgView.appendSamples(batch.filtered, beatsRel, batch.gapBefore)
  }

  private val exportLauncher = registerForActivityResult(
    ActivityResultContracts.StartActivityForResult(),
  ) { result ->
    val staged = pendingExportFile
    pendingExportFile = null
    val uri = result.data?.data
    if (uri == null || staged == null) {
      Toast.makeText(this, R.string.recording_export_canceled, Toast.LENGTH_LONG).show()
      staged?.delete()
      finish()
      return@registerForActivityResult
    }
    bg.execute {
      val error = try {
        RecordMatActions.writeToUri(this, uri, staged)
        null
      } catch (e: Exception) {
        e.message ?: "unknown"
      } finally {
        staged.delete()
      }
      runOnUiThread {
        if (error == null) {
          Toast.makeText(this, R.string.recording_export_saved, Toast.LENGTH_SHORT).show()
        } else {
          Toast.makeText(this, "내보내기 실패: $error", Toast.LENGTH_LONG).show()
        }
        finish()
      }
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_ecg_measurement)

    mode = intent.getStringExtra(EXTRA_MODE) ?: MODE_RECORDING
    totalSec = intent.getIntExtra(EXTRA_SECONDS, 60)

    countText = findViewById(R.id.measureCountText)
    timeline = findViewById(R.id.measureTimeline)
    ecgView = findViewById(R.id.measureEcgView)
    bpmText = findViewById(R.id.measureBpmText)

    findViewById<TextView>(R.id.measureModeLabel).setText(
      if (mode == MODE_CPC) R.string.measure_mode_cpc else R.string.measure_mode_short_term,
    )
    findViewById<TextView>(R.id.measureStatusHint).setText(
      if (mode == MODE_CPC) R.string.measuring_hint_cpc else R.string.recording_hint,
    )

    ecgView.settings = GraphSettings(
      sampleRateHz = ModelConfig.FS_HZ,
      showGrid = true,
      showRPeak = true,
      // 임상 캘리브레이션(20mm/mV·25mm/s) — 모니터와 동일 스케일.
      calibrated = true,
      mmPerMv = 20f,
      mmPerSec = 25f,
      showCalMarker = true,
      showScaleReadout = true,
    )

    findViewById<ImageButton>(R.id.closeButton).setOnClickListener { onCloseTapped() }
    onBackPressedDispatcher.addCallback(this) { onCloseTapped() }

    coordinator.addBpmListener(bpmListener)
    coordinator.onPlotBatch = plotListener

    when (mode) {
      MODE_CPC -> setupCpc()
      else -> setupRecording()
    }
  }

  // ── CPC 1분 캘리브레이션 (기존 흐름 유지) ─────────────────────────────────
  private fun setupCpc() {
    countText.text = totalSec.toString()
    timeline.totalSeconds = totalSec
    coordinator.cpcManager.onProgress = progress@{ remaining, progress ->
      runOnUiThread {
        countText.text = remaining.toString()
        timeline.elapsedFraction = progress
      }
    }
    // 인코딩 완료 후 저장된 개인화 레코드의 strip 상세로 이동.
    coordinator.onCpcCalibrationComplete = complete@{ recordId ->
      runOnUiThread {
        if (recordId != null) {
          startActivity(
            Intent(this, RecordDetailActivity::class.java)
              .putExtra(RecordDetailActivity.EXTRA_ID, recordId),
          )
        }
        finish()
      }
    }
    coordinator.startCpcCalibration()
  }

  // ── 레코딩: 중지할 때까지 연속 기록 ───────────────────────────────────────
  private fun setupRecording() {
    countText.text = formatElapsed(0)
    findViewById<TextView>(R.id.measureCountCaption).setText(R.string.recording_elapsed)
    timeline.visibility = View.GONE
    findViewById<View>(R.id.stopRecordBar).visibility = View.VISIBLE

    coordinator.recordingManager.stateListener = listener@{ state, elapsed ->
      if (state != RecordingManager.State.RECORDING) return@listener
      runOnUiThread { countText.text = formatElapsed(elapsed) }
    }

    val stopButton = findViewById<Button>(R.id.stopRecordButton)
    stopButton.setOnClickListener {
      stopButton.isEnabled = false
      coordinator.stopRecording { record ->
        if (record == null) {
          Toast.makeText(this, R.string.recording_empty_discarded, Toast.LENGTH_SHORT).show()
          finish()
        } else {
          showSaveDialog(record)
        }
      }
    }
    coordinator.startRecording()
  }

  private fun onCloseTapped() {
    if (mode != MODE_CPC &&
      coordinator.recordingManager.state == RecordingManager.State.RECORDING
    ) {
      AlertDialog.Builder(this)
        .setTitle(R.string.recording_discard_title)
        .setMessage(R.string.recording_discard_message)
        .setPositiveButton(R.string.recording_discard_confirm) { _, _ ->
          coordinator.cancelActiveMeasurement()
          finish()
        }
        .setNegativeButton(R.string.recording_keep_recording, null)
        .show()
      return
    }
    coordinator.cancelActiveMeasurement()
    finish()
  }

  // ── 저장/내보내기 다이얼로그 ─────────────────────────────────────────────
  private fun showSaveDialog(record: DiagnosisRecord) {
    val view = layoutInflater.inflate(R.layout.dialog_recording_save, null)
    val adcCheck = view.findViewById<CheckBox>(R.id.exportAdcCheck)
    val mvCheck = view.findViewById<CheckBox>(R.id.exportMvCheck)
    val matRadio = view.findViewById<RadioButton>(R.id.exportFormatMat)

    val durSec = ((record.endedAtMs - record.startedAtMs) / 1000).toInt()
    view.findViewById<TextView>(R.id.saveSummaryText).text =
      "${formatElapsed(durSec)} · ${record.totalSamples()} 샘플 · ${record.fs} Hz"

    val dialog = AlertDialog.Builder(this)
      .setTitle(R.string.recording_save_title)
      .setView(view)
      .setPositiveButton(R.string.recording_export, null) // 아래에서 검증 후 처리
      .setNeutralButton(R.string.recording_save_only) { _, _ -> saveOnly(record) }
      .setNegativeButton(R.string.recording_delete) { _, _ -> finish() }
      .setCancelable(false)
      .create()
    dialog.show()

    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
      if (!adcCheck.isChecked && !mvCheck.isChecked) {
        Toast.makeText(this, R.string.recording_need_signal, Toast.LENGTH_SHORT).show()
        return@setOnClickListener
      }
      val options = RecordingExporter.Options(
        includeAdc = adcCheck.isChecked,
        includeMv = mvCheck.isChecked,
        format = if (matRadio.isChecked) RecordingExporter.Format.MAT else RecordingExporter.Format.NPZ,
      )
      dialog.dismiss()
      saveAndExport(record, options)
    }
  }

  private fun saveOnly(record: DiagnosisRecord) {
    bg.execute {
      coordinator.saveRecording(record)
      runOnUiThread {
        Toast.makeText(this, R.string.recording_saved, Toast.LENGTH_SHORT).show()
        finish()
      }
    }
  }

  private fun saveAndExport(record: DiagnosisRecord, options: RecordingExporter.Options) {
    Toast.makeText(this, R.string.recording_exporting, Toast.LENGTH_SHORT).show()
    val fileName = RecordingExporter.fileNameFor(record, options.format)
    bg.execute {
      coordinator.saveRecording(record)
      val staged = File(cacheDir, fileName)
      val error = try {
        RecordingExporter.export(record, options, staged)
        null
      } catch (e: Exception) {
        staged.delete()
        e.message ?: "unknown"
      }
      runOnUiThread {
        if (isFinishing || isDestroyed) {
          staged.delete()
          return@runOnUiThread
        }
        if (error != null) {
          Toast.makeText(this, "export 실패: $error", Toast.LENGTH_LONG).show()
          finish()
          return@runOnUiThread
        }
        pendingExportFile = staged
        exportLauncher.launch(RecordMatActions.createSaveIntent(fileName))
      }
    }
  }

  private fun formatElapsed(sec: Int): String {
    val h = sec / 3600
    val m = (sec % 3600) / 60
    val s = sec % 60
    return if (h > 0) {
      "$h:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}"
    } else {
      "$m:${s.toString().padStart(2, '0')}"
    }
  }

  override fun onDestroy() {
    val coord = MainActivity.peekCoordinator()
    if (coord != null) {
      coord.removeBpmListener(bpmListener)
      if (coord.onPlotBatch === plotListener) coord.onPlotBatch = null
      coord.cpcManager.onProgress = null
      coord.cpcManager.onComplete = null
      coord.onCpcCalibrationComplete = null
      coord.recordingManager.stateListener = null
      if (coord.cpcManager.state == CpcCalibrationManager.State.RECORDING ||
        coord.recordingManager.state == RecordingManager.State.RECORDING
      ) {
        coord.cancelActiveMeasurement()
      } else if (coord.bleSource.isConnected) {
        coord.setContinuousMonitor()
      }
    }
    bg.shutdown()
    super.onDestroy()
  }

  companion object {
    const val EXTRA_MODE = "measure_mode"
    const val EXTRA_SECONDS = "measure_seconds"
    const val MODE_RECORDING = "recording"
    const val MODE_CPC = "cpc"
  }
}
