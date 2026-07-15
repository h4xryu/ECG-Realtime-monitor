package com.example.hicardimonitor.ui.record

import android.os.Bundle
import android.view.LayoutInflater
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.hicardimonitor.ModelConfig
import com.example.hicardimonitor.R
import com.example.hicardimonitor.StripRecordBuffer
import com.example.hicardimonitor.StripRecordView
import com.example.hicardimonitor.data.DiagnosisRecord
import com.example.hicardimonitor.data.RecordMatActions
import com.example.hicardimonitor.data.RecordRepository
import com.example.hicardimonitor.session.PersonalizationQuality
import com.example.hicardimonitor.ui.MainActivity
import com.example.hicardimonitor.ui.widget.EmbeddingView
import com.example.hicardimonitor.ui.widget.PersonalizationMetricsView
import com.example.hicardimonitor.ui.widget.SaliencyWaveView
import java.io.File
import java.util.concurrent.Executors

class RecordDetailActivity : AppCompatActivity() {

  private lateinit var record: DiagnosisRecord
  private lateinit var matFile: File
  private var pendingSaveFile: File? = null

  private lateinit var signal: FloatArray
  private lateinit var rawSignal: FloatArray
  private val bg = Executors.newSingleThreadExecutor { r -> Thread(r, "RecordDetail").apply { isDaemon = true } }
  private val fs get() = ModelConfig.FS_HZ

  // 개인화 대표 윈도우(임베딩 CAM 대상) + CAM 캐시
  private var repWindow: FloatArray? = null
  private var embeddingCamOn = false
  private var embeddingCam: FloatArray? = null

  private val saveMatLauncher = registerForActivityResult(
    ActivityResultContracts.StartActivityForResult(),
  ) { result ->
    val uri = result.data?.data ?: return@registerForActivityResult
    val file = pendingSaveFile ?: return@registerForActivityResult
    pendingSaveFile = null
    try {
      RecordMatActions.writeToUri(this, uri, file)
      Toast.makeText(this, R.string.save_mat_success, Toast.LENGTH_SHORT).show()
    } catch (e: Exception) {
      Toast.makeText(this, "저장 실패: ${e.message}", Toast.LENGTH_SHORT).show()
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_record_detail)

    // 엣지-투-엣지: 상태바/내비바 인셋만큼 상·하단 패딩 (상단 strip 잘림 방지)
    val root = findViewById<android.view.View>(R.id.detailRoot)
    ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
      v.setPadding(v.paddingLeft, bars.top, v.paddingRight, bars.bottom)
      insets
    }

    val id = intent.getStringExtra(EXTRA_ID) ?: run { finish(); return }
    val repo = RecordRepository(this)
    record = repo.load(id) ?: run {
      Toast.makeText(this, "기록 없음", Toast.LENGTH_SHORT).show()
      finish()
      return
    }
    matFile = repo.matFileFor(id)
    signal = record.mergedFiltered()
    // 모델(CPC 인코더/CAM) 입력은 학습과 동일하게 raw ECG. 표시(strip)는 필터 신호 유지.
    rawSignal = record.mergedRawAdc()

    val strip = findViewById<StripRecordView>(R.id.detailStripView)
    val buf = StripRecordBuffer(fs = fs)
    buf.secondsPerRow = 30f
    buf.visibleWindowSec = (record.totalSamples() / fs).coerceAtLeast(60f)
    if (signal.isNotEmpty()) buf.appendSegment(signal, null)
    strip.buffer = buf
    strip.fs = fs
    strip.hicardiMode = true
    strip.onBeatTap = { idx -> showBeatDetail(idx) }

    findViewById<TextView>(R.id.detailDiagnosisText).text = detailSummary(record)

    findViewById<Button>(R.id.saveMatButton).setOnClickListener { exportAndSaveMat() }
    findViewById<Button>(R.id.shareMatButton).setOnClickListener { exportAndShareMat() }

    if (record.isPersonalization) setupPersonalization()
  }

  // ── 개인화 지표·임베딩·CAM ────────────────────────────────────────────────
  private fun setupPersonalization() {
    val section = findViewById<android.view.View>(R.id.personalizationSection)
    section.visibility = android.view.View.VISIBLE
    val metricsView = findViewById<PersonalizationMetricsView>(R.id.metricsView)
    val embeddingView = findViewById<EmbeddingView>(R.id.embeddingView)
    val embeddingWave = findViewById<SaliencyWaveView>(R.id.embeddingWave)
    val camButton = findViewById<Button>(R.id.embeddingCamButton)
    camButton.isEnabled = false

    val coord = MainActivity.peekCoordinator()
    if (coord == null || !coord.cpcRunner.isLoaded || rawSignal.size < ModelConfig.CPC_WINDOW_SIZE) {
      metricsView.setQuality(null)
      return
    }
    val runner = coord.cpcRunner

    bg.execute {
      val enc = coord.encodeSignal(rawSignal) ?: return@execute
      val quality = PersonalizationQuality.from(enc)
      // pooled 에 가장 가까운(가장 대표적인) 윈도우를 CAM 대상으로 선택
      var bestJ = 0
      var bestSim = -2f
      for (j in enc.perWindow.indices) {
        val s = com.example.hicardimonitor.session.CpcRunner.cosine(enc.perWindow[j], enc.pooled)
        if (s > bestSim) { bestSim = s; bestJ = j }
      }
      val start = (bestJ * ModelConfig.CPC_EMBED_HOP).coerceIn(0, (rawSignal.size - ModelConfig.CPC_WINDOW_SIZE).coerceAtLeast(0))
      val win = rawSignal.copyOfRange(start, (start + ModelConfig.CPC_WINDOW_SIZE).coerceAtMost(rawSignal.size))
      repWindow = win
      runOnUiThread {
        metricsView.setQuality(quality)
        embeddingView.setEmbedding(enc.pooled)
        embeddingWave.setSignal(win)
        camButton.isEnabled = true
      }
    }

    camButton.setOnClickListener {
      val win = repWindow ?: return@setOnClickListener
      if (embeddingCamOn) {
        embeddingCamOn = false
        embeddingWave.setCam(null)
        camButton.setText(R.string.cam_show)
        return@setOnClickListener
      }
      val cached = embeddingCam
      if (cached != null) {
        embeddingCamOn = true
        embeddingWave.setCam(cached)
        camButton.setText(R.string.cam_hide)
        return@setOnClickListener
      }
      camButton.isEnabled = false
      bg.execute {
        val cam = runner.occlusionSaliency(win)
        runOnUiThread {
          camButton.isEnabled = true
          if (cam != null) {
            embeddingCam = cam
            embeddingCamOn = true
            embeddingWave.setCam(cam)
            camButton.setText(R.string.cam_hide)
          }
        }
      }
    }
  }

  // ── strip 탭 → 2초 윈도우 확대 + CAM ──────────────────────────────────────
  private fun showBeatDetail(sampleIndex: Int) {
    if (rawSignal.size < 2) return
    val win = ModelConfig.CPC_WINDOW_SIZE
    val start = (sampleIndex - win / 2).coerceIn(0, (rawSignal.size - win).coerceAtLeast(0))
    val end = (start + win).coerceAtMost(rawSignal.size)
    // 인코더/CAM 이 실제로 처리하는 raw ECG 윈도우를 표시.
    val window = rawSignal.copyOfRange(start, end)

    val view = LayoutInflater.from(this).inflate(R.layout.dialog_beat_detail, null)
    val wave = view.findViewById<SaliencyWaveView>(R.id.beatWaveView)
    val timeText = view.findViewById<TextView>(R.id.beatTimeText)
    val camButton = view.findViewById<Button>(R.id.beatCamButton)
    wave.setSignal(window)
    timeText.text = formatTime(sampleIndex / fs)

    val runner = MainActivity.peekCoordinator()?.cpcRunner
    var camOn = false
    var camCache: FloatArray? = null
    camButton.isEnabled = runner != null && runner.isLoaded && window.size >= win / 2
    camButton.setOnClickListener click@{
      val r = runner ?: return@click
      if (camOn) {
        camOn = false
        wave.setCam(null)
        camButton.setText(R.string.cam_show)
        return@click
      }
      camCache?.let {
        camOn = true
        wave.setCam(it)
        camButton.setText(R.string.cam_hide)
        return@click
      }
      camButton.isEnabled = false
      bg.execute {
        val cam = r.occlusionSaliency(window)
        runOnUiThread {
          camButton.isEnabled = true
          if (cam != null) {
            camCache = cam
            camOn = true
            wave.setCam(cam)
            camButton.setText(R.string.cam_hide)
          }
        }
      }
    }

    AlertDialog.Builder(this)
      .setView(view)
      .setPositiveButton(R.string.close, null)
      .show()
  }

  private fun formatTime(sec: Float): String {
    val s = sec.toInt()
    return "${s / 60}:${(s % 60).toString().padStart(2, '0')}"
  }

  private fun exportMatOrThrow() = RecordMatActions.ensureMatFile(record, matFile)

  private fun exportAndSaveMat() {
    try {
      exportMatOrThrow()
      pendingSaveFile = matFile
      saveMatLauncher.launch(RecordMatActions.createSaveIntent(RecordMatActions.displayFileName(record)))
    } catch (e: Exception) {
      Toast.makeText(this, "export 실패: ${e.message}", Toast.LENGTH_SHORT).show()
    }
  }

  private fun exportAndShareMat() {
    try {
      exportMatOrThrow()
      RecordMatActions.shareMat(this, matFile, RecordMatActions.displayFileName(record))
    } catch (e: Exception) {
      Toast.makeText(this, "export 실패: ${e.message}", Toast.LENGTH_SHORT).show()
    }
  }

  private fun detailSummary(record: DiagnosisRecord): String {
    return if (record.isPersonalization) {
      getString(R.string.record_type_personalization_detail)
    } else {
      "진단: ${record.summary}\nBPM: ${record.bpm}"
    }
  }

  override fun onDestroy() {
    bg.shutdownNow()
    super.onDestroy()
  }

  companion object {
    const val EXTRA_ID = "record_id"
  }
}
