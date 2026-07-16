package com.example.hicardimonitor

import android.Manifest
import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.hicardimonitor.data.DiagnosisRecord
import com.example.hicardimonitor.data.MatlabExporter
import com.example.hicardimonitor.data.RecordRepository
import com.example.hicardimonitor.data.RecordSampleBlock
import com.example.hicardimonitor.session.ArrhythmiaInferenceEngine
import com.example.hicardimonitor.session.CpcCalibrationManager
import com.example.hicardimonitor.session.CpcRunner
import com.example.hicardimonitor.session.HiCardiLabelCount
import com.example.hicardimonitor.session.PatientEncoding
import com.example.hicardimonitor.session.PersonalizationQuality
import com.example.hicardimonitor.session.RecordingManager
import java.util.ArrayDeque
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.ExecutorService

/**
 * HiCardi BLE + 추론 + 기록 오케스트레이션 (단일 모드).
 */
class HiCardiCoordinator(private val activity: AppCompatActivity) {

  val bleSource = HiCardiBleSource(activity, ModelConfig.FS_HZ)
  val modelRunner = ModelRunner(activity)
  val cpcRunner = CpcRunner(activity)
  val arrhythmiaRunner = ArrhythmiaClassifierRunner(activity)
  val arrhythmiaEngine = ArrhythmiaInferenceEngine(arrhythmiaRunner, ModelConfig.FS_HZ)
  val recordRepo = RecordRepository(activity)
  val stripBuffer = StripRecordBuffer(fs = ModelConfig.FS_HZ)
  val cpcManager = CpcCalibrationManager()
  val recordingManager = RecordingManager()

  // 아래 필드들은 메인 스레드(UI 토글·연결 콜백)와 bleExecutor/inferenceExecutor 가
  // 동기화 없이 공유하므로 @Volatile 로 가시성을 보장한다.

  /** 마지막 개인화 품질 지표 + 저장된 레코드 id (측정 종료 화면에서 사용). */
  @Volatile var personalizationQuality: PersonalizationQuality? = null
    private set
  @Volatile var lastPersonalizationRecordId: String? = null
    private set

  @Volatile var measurementMode: MeasurementMode = MeasurementMode.IDLE
    private set

  @Volatile var enableInference: Boolean = true

  /** 실시간 6-class 추론에 개인화 patient_vec 적용 여부(설정에서 토글). */
  @Volatile var applyPersonalization: Boolean = true

  /** 최신 6-class 부정맥 추론 결과(모니터 카드 초기 표시용). */
  var lastArrhythmia: ArrhythmiaResult? = null
    private set

  @Volatile var patientEmbedding: FloatArray? = null
    private set

  var onPlotBatch: ((BleSampleBatch, IntArray) -> Unit)? = null

  /** 1분 개인화 인코딩 완료(저장된 recordId 전달, 메인 스레드). */
  var onCpcCalibrationComplete: ((recordId: String?) -> Unit)? = null

  // CopyOnWriteArrayList: 리스너가 콜백 안에서 자신을 제거해도(순회 중 remove) CME 없이 안전.
  private val connectionListeners = CopyOnWriteArrayList<(Boolean) -> Unit>()
  private val logListeners = CopyOnWriteArrayList<(String) -> Unit>()
  private val bpmListeners = CopyOnWriteArrayList<(Int) -> Unit>()
  private val predListeners = CopyOnWriteArrayList<(String) -> Unit>()
  private val arrhythmiaListeners = CopyOnWriteArrayList<(ArrhythmiaResult) -> Unit>()
  private val tempListeners = CopyOnWriteArrayList<(Float?) -> Unit>()
  private val stripListeners = CopyOnWriteArrayList<() -> Unit>()
  private val linkStateListeners = CopyOnWriteArrayList<(SensorLinkState) -> Unit>()

  fun addConnectionListener(listener: (Boolean) -> Unit) {
    connectionListeners.add(listener)
  }

  fun removeConnectionListener(listener: (Boolean) -> Unit) {
    connectionListeners.remove(listener)
  }

  fun addLogListener(listener: (String) -> Unit) {
    logListeners.add(listener)
  }

  fun removeLogListener(listener: (String) -> Unit) {
    logListeners.remove(listener)
  }

  fun addBpmListener(listener: (Int) -> Unit) {
    bpmListeners.add(listener)
  }

  fun removeBpmListener(listener: (Int) -> Unit) {
    bpmListeners.remove(listener)
  }

  fun addPredictionListener(listener: (String) -> Unit) {
    predListeners.add(listener)
  }

  fun removePredictionListener(listener: (String) -> Unit) {
    predListeners.remove(listener)
  }

  fun addArrhythmiaListener(listener: (ArrhythmiaResult) -> Unit) {
    arrhythmiaListeners.add(listener)
  }

  fun removeArrhythmiaListener(listener: (ArrhythmiaResult) -> Unit) {
    arrhythmiaListeners.remove(listener)
  }

  fun addTemperatureListener(listener: (Float?) -> Unit) {
    tempListeners.add(listener)
  }

  fun removeTemperatureListener(listener: (Float?) -> Unit) {
    tempListeners.remove(listener)
  }

  fun addStripListener(listener: () -> Unit) {
    stripListeners.add(listener)
  }

  fun removeStripListener(listener: () -> Unit) {
    stripListeners.remove(listener)
  }

  fun addLinkStateListener(listener: (SensorLinkState) -> Unit) {
    linkStateListeners.add(listener)
  }

  fun removeLinkStateListener(listener: (SensorLinkState) -> Unit) {
    linkStateListeners.remove(listener)
  }

  /** @deprecated 단일 리스너 대신 addConnectionListener 사용 */
  var onConnectionChanged: ((Boolean) -> Unit)?
    get() = null
    set(value) {
      if (value != null) connectionListeners.add(value)
    }
  var onNavigateToSensor: (() -> Unit)? = null
  var onSensorDisconnected: (() -> Unit)? = null

  private val mainHandler = Handler(Looper.getMainLooper())
  private val bleExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
    Thread(r, "HiCardiBle").apply { isDaemon = true }
  }

  /** 무거운 일회성 추론(CPC 인코딩)용 — bleExecutor(배치 파이프라인)를 막지 않는다. */
  private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
    Thread(r, "HiCardiInfer").apply { isDaemon = true }
  }

  @Volatile private var lastInferAtMs = 0L
  private val inferMinIntervalMs = 350L
  private val inferWindow = FloatArrayDeque(ModelConfig.WINDOW_SIZE)
  private val bpmWindow = FloatArrayDeque((ModelConfig.FS_HZ * 4).toInt())
  private val rpeakTracker = SensorRPeakTracker(ModelConfig.FS_HZ)
  @Volatile private var lastBpm = 0
  @Volatile private var lastPred = "-"
  private var lastTemp: Float? = null
  private var wasConnected = false

  private val continuousFiltered = ArrayList<Float>(8192)
  private val continuousRaw = ArrayList<Float>(8192)
  private val continuousFlags = ArrayList<FloatArray>(8192)
  private val continuousRPeaks = ArrayList<Int>()
  private var continuousStartMs = 0L
  private val reconnectEvents = ArrayList<Long>()

  /** 연속 모니터 자동 버퍼 상한(1시간 @ fs) — 무한 성장으로 인한 OOM 방지. */
  private val maxContinuousSamples = (ModelConfig.FS_HZ * 3600).toInt()
  private var continuousCapLogged = false

  fun attach() {
    stripBuffer.secondsPerRow = 30f
    stripBuffer.visibleWindowSec = 60f
    arrhythmiaEngine.patientVecProvider = { if (applyPersonalization) patientEmbedding else null }
    arrhythmiaEngine.enabledProvider = { enableInference }
    arrhythmiaEngine.bpmProvider = { rpeakTracker.bpm() ?: lastBpm }
    arrhythmiaEngine.onResult = { res, _ ->
      mainHandler.post {
        lastArrhythmia = res
        arrhythmiaListeners.forEach { it(res) }
      }
    }
    bleSource.onLog = { msg -> mainHandler.post { logListeners.forEach { it(msg) } } }
    bleSource.onLinkStateChanged = { state ->
      mainHandler.post { linkStateListeners.forEach { it(state) } }
    }
    bleSource.onBatch = { batch -> bleExecutor.execute { handleBleBatch(batch) } }
    bleSource.onConnectionChanged = { connected ->
      mainHandler.post {
        if (wasConnected && !connected) {
          onSensorDisconnected?.invoke()
          onNavigateToSensor?.invoke()
          measurementMode = MeasurementMode.IDLE
        }
        if (!wasConnected && connected) {
          reconnectEvents.add(System.currentTimeMillis())
        }
        wasConnected = connected
        if (connected && measurementMode == MeasurementMode.IDLE) {
          measurementMode = MeasurementMode.CONTINUOUS_MONITOR
          continuousStartMs = System.currentTimeMillis()
        }
        onConnectionChanged?.invoke(connected)
        connectionListeners.forEach { it(connected) }
      }
    }
    bleExecutor.execute {
      modelRunner.load()
      cpcRunner.load()
      arrhythmiaRunner.load()
      // 재시작 후 patient_vec 이 null 이면 저장된 최신 개인화 기록에서 복원.
      // (없으면 6-class 모델이 zero patient_vec 로 붕괴 → AF 오탐. #personalization)
      restorePatientEmbeddingIfNeeded()
    }
  }

  fun requestBlePermissions() {
    val perms = arrayOf(
      Manifest.permission.BLUETOOTH_SCAN,
      Manifest.permission.BLUETOOTH_CONNECT,
    )
    ActivityCompat.requestPermissions(activity, perms, BLE_PERM_REQUEST)
  }

  fun onBlePermissionsResult(granted: Boolean) {
    if (granted) bleSource.start() else log("BLE permission denied")
  }

  fun connectBle() {
    if (hasBlePermissions()) bleSource.start() else requestBlePermissions()
  }

  fun disconnectBle() {
    bleSource.stop()
    resetBeatPipeline()
    measurementMode = MeasurementMode.IDLE
  }

  fun setContinuousMonitor() {
    if (!bleSource.isConnected) return
    resetBeatPipeline()
    measurementMode = MeasurementMode.CONTINUOUS_MONITOR
  }

  fun startRecording() {
    if (!bleSource.isConnected) return
    resetBeatPipeline()
    recordingManager.start()
    measurementMode = MeasurementMode.RECORDING
  }

  fun startCpcCalibration() {
    if (!bleSource.isConnected) return
    resetBeatPipeline()
    cpcManager.start()
    measurementMode = MeasurementMode.CPC_CALIBRATION
  }

  /**
   * R-peak 트래커·비트 추론 엔진 리셋. 두 객체의 내부 상태는 bleExecutor 에서만 쓰므로
   * 메인에서 직접 부르지 않고 같은 executor 로 직렬화해 배치 처리와의 경합을 막는다.
   */
  private fun resetBeatPipeline() {
    bleExecutor.execute {
      rpeakTracker.reset()
      arrhythmiaEngine.reset()
    }
  }

  fun cancelActiveMeasurement() {
    when (measurementMode) {
      MeasurementMode.RECORDING -> recordingManager.cancel()
      MeasurementMode.CPC_CALIBRATION -> cpcManager.cancel()
      else -> {}
    }
    measurementMode = if (bleSource.isConnected) MeasurementMode.CONTINUOUS_MONITOR else MeasurementMode.IDLE
  }

  /**
   * 녹화 중지 → 그 시점까지의 데이터로 레코드 생성(저장 여부는 호출측 결정).
   * feed 와 같은 bleExecutor 에서 직렬화해 buildRecord 중 버퍼 변경을 차단한다.
   * 콜백은 메인 스레드. 샘플이 없으면 null.
   */
  fun stopRecording(onBuilt: (DiagnosisRecord?) -> Unit) {
    bleExecutor.execute {
      recordingManager.stop()
      val record = if (recordingManager.hasSamples()) {
        recordingManager.buildRecord(recordRepo.newId())
      } else {
        null
      }
      recordingManager.reset()
      mainHandler.post {
        measurementMode = if (bleSource.isConnected) MeasurementMode.CONTINUOUS_MONITOR else MeasurementMode.IDLE
        onBuilt(record)
      }
    }
  }

  fun saveRecording(record: DiagnosisRecord) {
    recordRepo.save(record)
  }

  fun finalizeCpcRecord(): DiagnosisRecord? {
    if (cpcManager.state != CpcCalibrationManager.State.COMPLETE) return null
    val record = cpcManager.buildRecord(recordRepo.newId())
    recordRepo.save(record)
    return record
  }

  /**
   * 1분 캘리브레이션 → CPC patient embedding + 품질 지표 산출, 레코드 저장.
   * CPC .pte 미로드 시 구형 classifier fallback(extractPatientEmbedding)을 사용.
   * inferenceExecutor 에서 실행 — measurementMode 가드가 이중 호출을 막고,
   * 종료 시 CONTINUOUS_MONITOR 로 전환하기 전까지 bleExecutor 는 추론을 돌리지 않는다.
   */
  fun applyCpcCalibration() {
    if (measurementMode != MeasurementMode.CPC_CALIBRATION) return // 이중 호출 방지
    val signal = cpcManager.recordedRawSignal()
    if (cpcRunner.isLoaded) {
      val encoding = cpcRunner.encodePatient(signal)
      if (encoding != null) {
        patientEmbedding = encoding.pooled
        personalizationQuality = PersonalizationQuality.from(encoding)
      }
    }
    if (patientEmbedding == null || patientEmbedding?.size != ModelConfig.PATIENT_EMBEDDING_DIM) {
      cpcManager.calibrationWindows?.let { patientEmbedding = modelRunner.extractPatientEmbedding(it) }
    }
    lastPersonalizationRecordId = finalizeCpcRecord()?.id
    measurementMode = MeasurementMode.CONTINUOUS_MONITOR
  }

  /** 저장된 개인화 신호를 다시 인코딩 (기록 상세에서 임베딩·품질·CAM 재계산용). */
  fun encodeSignal(signal: FloatArray): PatientEncoding? =
    if (cpcRunner.isLoaded) cpcRunner.encodePatient(signal) else null

  /**
   * 저장된 개인화 기록을 CPC 인코딩해 patient_vec 으로 적용.
   * background(bleExecutor/inferenceExecutor)에서 호출할 것 — 디스크 로드 + 무거운 인코딩.
   * @return 성공 여부.
   */
  fun applyPersonalizationFromRecord(recordId: String): Boolean {
    if (!cpcRunner.isLoaded) return false
    val record = recordRepo.load(recordId) ?: return false
    val signal = record.mergedRawAdc()
    if (signal.size < ModelConfig.CPC_WINDOW_SIZE) return false
    val encoding = cpcRunner.encodePatient(signal) ?: return false
    patientEmbedding = encoding.pooled
    personalizationQuality = PersonalizationQuality.from(encoding)
    lastPersonalizationRecordId = recordId
    mainHandler.post { log("개인화 복원: 기록 $recordId → patient_vec ${encoding.pooled.size}D") }
    return true
  }

  /** 앱 시작 시 patient_vec 이 비어 있으면 최신 개인화 기록에서 복원(라이브 캘리브 불필요). */
  private fun restorePatientEmbeddingIfNeeded() {
    if (patientEmbedding?.size == ModelConfig.PATIENT_EMBEDDING_DIM) return
    val latest = recordRepo.listRecords().firstOrNull { it.isPersonalization } ?: return
    applyPersonalizationFromRecord(latest.id)
  }

  fun saveContinuousRecord(): DiagnosisRecord? {
    if (continuousFiltered.isEmpty()) return null
    val record = DiagnosisRecord(
      id = recordRepo.newId(),
      startedAtMs = continuousStartMs,
      endedAtMs = System.currentTimeMillis(),
      fs = ModelConfig.FS_HZ.toInt(),
      bpm = lastBpm,
      summary = lastPred,
      blocks = listOf(
        RecordSampleBlock(
          filtered = continuousFiltered.toFloatArray(),
          rawAdc = continuousRaw.toFloatArray(),
          perSampleFlags = continuousFlags.toTypedArray(),
          rPeakIndices = continuousRPeaks.toIntArray(),
          dataLossIndices = intArrayOf(),
          leadOffIndices = intArrayOf(),
        ),
      ),
      isShortTerm = false,
      reconnectEvents = reconnectEvents.toList(),
    )
    recordRepo.save(record)
    return record
  }

  fun exportMat(recordId: String): java.io.File? {
    val record = recordRepo.load(recordId) ?: return null
    val out = recordRepo.matFileFor(recordId)
    MatlabExporter.export(record, out)
    return out
  }

  fun patientEmbeddingSize(): Int = patientEmbedding?.size ?: 0

  fun destroy() {
    bleSource.stop()
    bleExecutor.shutdownNow()
    inferenceExecutor.shutdownNow()
    modelRunner.close()
    cpcRunner.close()
    arrhythmiaRunner.close()
  }

  private data class UiPulse(
    val batch: BleSampleBatch,
    val beatsRel: IntArray = IntArray(0),
    val bpm: Int? = null,
    val pred: String? = null,
    val temp: Float? = null,
    val stripDirty: Boolean = false,
  )

  private fun handleBleBatch(batch: BleSampleBatch) {
    if (!bleSource.isConnected) return

    var bpmUpdate: Int? = null
    var predUpdate: String? = null
    val tempUpdate = batch.temperatureC
    var stripDirty = false
    var beatsRel = IntArray(0)
    // lead-off: 전극 미접촉 노이즈로 모델·R-peak·BPM 을 돌리지 않는다(CPU 절약 + 오탐 방지).
    // 파형 플롯과 기록(lead_off 플래그)은 유지.
    val leadOff = batch.leadOff

    when (measurementMode) {
      MeasurementMode.CPC_CALIBRATION -> {
        // lead-off 동안 캘리브레이션 버퍼 오염 방지 — 일시정지.
        val done = if (leadOff) false else cpcManager.feed(batch.filtered, batch.rawAdc)
        if (done) {
          // 인코딩(무거움)은 전용 스레드로 — bleExecutor 가 막히면 배치가 큐에 쌓여
          // 플롯/BPM 이 인코딩 시간만큼 정지하므로 분리한다. 완료 콜백만 메인으로.
          inferenceExecutor.execute {
            applyCpcCalibration()
            val rid = lastPersonalizationRecordId
            mainHandler.post { onCpcCalibrationComplete?.invoke(rid) }
          }
        }
      }
      MeasurementMode.RECORDING -> {
        if (!leadOff) {
          beatsRel = rpeakTracker.process(batch.filtered, batch.rPeakIndices)
          predUpdate = processInference(batch)
          bpmUpdate = currentBpm()
          feedArrhythmia(batch, beatsRel)
        }
        recordingManager.feed(
          filteredSamples = batch.filtered,
          rawSamples = batch.rawAdc,
          rPeakRelIndices = beatsRel,
          isDataLoss = false,
          isLeadOff = leadOff,
          bpm = lastBpm,
          prediction = lastPred,
        )
      }
      MeasurementMode.CONTINUOUS_MONITOR -> {
        if (!leadOff) {
          beatsRel = rpeakTracker.process(batch.filtered, batch.rPeakIndices)
          predUpdate = processInference(batch)
          bpmUpdate = currentBpm()
          feedArrhythmia(batch, beatsRel)
        }
        appendContinuous(batch, beatsRel)
        stripBuffer.appendSegment(batch.filtered, null)
        stripDirty = true
      }
      MeasurementMode.IDLE -> {}
    }

    postUiPulse(
      UiPulse(
        batch = batch,
        beatsRel = beatsRel,
        bpm = bpmUpdate,
        pred = predUpdate,
        temp = tempUpdate,
        stripDirty = stripDirty,
      ),
    )
  }

  /** 센서 R-peak RR 기반 BPM. 아직 비트가 없으면 파형 임계검출로 폴백. */
  private fun currentBpm(): Int? {
    rpeakTracker.bpm()?.let { return it }
    if (bpmWindow.size >= (ModelConfig.FS_HZ * 2).toInt()) {
      return BpmFromPeaks.estimate(bpmWindow.toFloatArray(), ModelConfig.FS_HZ).bpm
    }
    return null
  }

  // UI 펄스 병합: 배치(~22/s)마다 post 하지 않고 큐에 쌓아 메인 루퍼가 한가할 때
  // 한 번에 반영한다. 파형은 전 배치를 그리고(연속성), 텍스트류는 마지막 값만.
  private val pendingPulses = java.util.concurrent.ConcurrentLinkedQueue<UiPulse>()
  private val uiDrainScheduled = java.util.concurrent.atomic.AtomicBoolean(false)

  // 리스너 통지 게이트(메인 스레드 전용): 값이 실제로 바뀔 때만 setText 등이 돌게 해
  // 배치 주기(22/s)의 불필요한 레이아웃 패스를 없앤다.
  private var uiLastBpm = Int.MIN_VALUE
  private var uiLastPred: String? = null
  private var uiLastTemp: Float? = null

  private fun postUiPulse(pulse: UiPulse) {
    pendingPulses.add(pulse)
    if (uiDrainScheduled.compareAndSet(false, true)) {
      mainHandler.post { drainUiPulses() }
    }
  }

  private fun drainUiPulses() {
    uiDrainScheduled.set(false)
    if (!bleSource.isConnected && measurementMode == MeasurementMode.IDLE) {
      pendingPulses.clear()
      return
    }
    var bpm: Int? = null
    var pred: String? = null
    var temp: Float? = null
    var stripDirty = false
    while (true) {
      val p = pendingPulses.poll() ?: break
      // 샘플 유실로 인한 파형 불연속(수직 스파이크)을 막기 위해 모든 배치를 그린다.
      onPlotBatch?.invoke(p.batch, p.beatsRel)
      p.bpm?.let { bpm = it }
      p.pred?.let { pred = it }
      p.temp?.let { temp = it }
      stripDirty = stripDirty || p.stripDirty
    }
    temp?.let {
      lastTemp = it
      if (it != uiLastTemp) {
        uiLastTemp = it
        tempListeners.forEach { l -> l(it) }
      }
    }
    bpm?.let {
      lastBpm = it
      if (it != uiLastBpm) {
        uiLastBpm = it
        bpmListeners.forEach { l -> l(it) }
      }
    }
    pred?.let {
      lastPred = it
      if (it != uiLastPred) {
        uiLastPred = it
        predListeners.forEach { l -> l(it) }
      }
    }
    if (stripDirty) {
      stripListeners.forEach { it() }
    }
  }

  /** 파형 버퍼 갱신 + AI 추론. @return 갱신된 예측 라벨(없으면 null). BPM은 [currentBpm]에서 산출. */
  private fun processInference(batch: BleSampleBatch): String? {
    // 모델 입력은 학습과 동일하게 raw ECG(밴드패스 없이 z-score는 ModelRunner 에서 수행).
    // BPM 폴백/피크검출은 표시용 필터 신호를 사용해 표시 파형과 일치시킨다.
    for (sample in batch.rawCounts) inferWindow.addLast(sample)
    for (sample in batch.filtered) bpmWindow.addLast(sample)

    if (inferWindow.size >= ModelConfig.WINDOW_SIZE && enableInference) {
      val now = System.currentTimeMillis()
      if (now - lastInferAtMs >= inferMinIntervalMs) {
        lastInferAtMs = now
        val window = inferWindow.toFloatArray()
        val calib = cpcManager.calibrationWindows
        val result = modelRunner.infer(window, calib, patientEmbedding)
        if (result != null) {
          lastPred = result.summaryLabel()
          lastBeatFlags = result.multilabelFlags()
          return lastPred
        }
      }
    }
    return null
  }

  /** 6-class 비트 중심 추론 엔진에 이 배치의 raw mV + 확정 R-peak 를 투입. */
  private fun feedArrhythmia(batch: BleSampleBatch, beatsRel: IntArray) {
    val mv = FloatArray(batch.rawCounts.size) { batch.rawCounts[it] / ArrhythmiaConfig.ADC_SCALE_MV }
    arrhythmiaEngine.onBatch(mv, beatsRel)
  }

  private var lastBeatFlags: FloatArray? = null

  private fun appendContinuous(batch: BleSampleBatch, beatsRel: IntArray) {
    if (continuousFiltered.size >= maxContinuousSamples) {
      if (!continuousCapLogged) {
        continuousCapLogged = true
        mainHandler.post { log("연속 기록 버퍼 상한(1시간) 도달 — 이후 샘플은 저장 버퍼에 기록되지 않습니다") }
      }
      return
    }
    val base = continuousFiltered.size
    for (i in batch.filtered.indices) {
      continuousFiltered.add(batch.filtered[i])
      continuousRaw.add(batch.rawAdc.getOrElse(i) { batch.filtered[i] })
      continuousFlags.add(lastBeatFlags ?: FloatArray(HiCardiLabelCount.NUM_CLASSES))
    }
    // 센서 온칩 R-peak(0xFF, 스냅 완료)를 실제 기록 인덱스로 저장.
    // 음수 rel 은 직전 배치로 스냅된 비트 → base+rel 이 버퍼 범위 안이면 그대로 유효.
    // 범위 밖(버퍼 시작 이전)은 클램프하면 엉뚱한 위치에 기록되므로 스킵.
    val size = continuousFiltered.size
    for (rel in beatsRel) {
      val idx = base + rel
      if (idx in 0 until size) continuousRPeaks.add(idx)
    }
  }

  private fun hasBlePermissions(): Boolean {
    val connect = ContextCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED
    val scan = ContextCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
    return connect && scan
  }

  private fun log(msg: String) = logListeners.forEach { it(msg) }

  companion object {
    const val BLE_PERM_REQUEST = 701
  }
}

private class FloatArrayDeque(private val capacity: Int) {
  private val deque = ArrayDeque<Float>(capacity)
  val size: Int get() = deque.size
  fun addLast(v: Float) {
    deque.addLast(v)
    while (deque.size > capacity) deque.removeFirst()
  }
  fun removeFirst() = deque.removeFirst()
  fun toFloatArray(): FloatArray = deque.toFloatArray()
}

private fun InferenceResult.summaryLabel(): String {
  val active = ArrayList<String>()
  for (i in probabilities.indices) {
    if (probabilities[i] >= 0.5f) {
      active.add(ModelConfig.CLASS_NAMES.getOrElse(i) { "?" })
    }
  }
  return if (active.isEmpty()) className else active.joinToString(", ")
}

private fun InferenceResult.multilabelFlags(): FloatArray {
  val n = HiCardiLabelCount.NUM_CLASSES
  val out = FloatArray(n)
  for (i in probabilities.indices) {
    if (i < n && probabilities[i] >= 0.5f) out[i] = 1f
  }
  return out
}
