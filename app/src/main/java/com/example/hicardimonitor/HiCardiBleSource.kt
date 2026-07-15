package com.example.hicardimonitor

import android.Manifest
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context
import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import androidx.core.content.ContextCompat
import java.nio.charset.StandardCharsets
import java.util.UUID

/**
 * HiCardi BLE 실시간 수신 — PyQt BleMonitorThread 와 동일 파이프라인.
 * parse: ((hi<<8)|lo) - 0x4000 → OnlineFilter → /32 mV
 */
class HiCardiBleSource(
  private val context: Context,
  private val fs: Float = 250f,
) {
  private val targetAddr = "EB:B5:51:19:06:24"
  private val targetName = "HiCardi-07143"

  private val nusServiceUuid = UUID.fromString("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
  private val nusTxUuid = UUID.fromString("6E400002-B5A3-F393-E0A9-E50E24DCCA9E")
  private val nusRxUuid = UUID.fromString("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
  private val cccdUuid = UUID.fromString("00002902-0000-1000-8000-00805F9B34FB")

  private val frameLen = 42
  private val frameEnd = 0x8F
  private val start = 8
  private val nTriplets = 11
  private val step = 3

  private val mainHandler = Handler(Looper.getMainLooper())
  private val initCmds = listOf("START\n", "ECG_ON\n", "\u0001")

  var onLog: ((String) -> Unit)? = null

  /** BLE 바인더 스레드에서 직접 호출됨 — 소비자가 자기 executor 로 직렬화할 것(메인 경유 없음). */
  var onBatch: ((BleSampleBatch) -> Unit)? = null
  var onConnectionChanged: ((Boolean) -> Unit)? = null
  var onLinkStateChanged: ((SensorLinkState) -> Unit)? = null

  // GATT/스캔 콜백은 바인더 스레드, start/stop 은 메인 스레드에서 실행되므로
  // 공유 상태는 @Volatile 로 가시성을 보장한다.
  @Volatile var isConnected: Boolean = false
    private set

  @Volatile var linkState: SensorLinkState = SensorLinkState.IDLE
    private set

  val targetDisplayName: String get() = targetName
  val targetDisplayAddress: String get() = targetAddr

  private val onlineFilter = OnlineFilter(fs)
  @Volatile private var gatt: BluetoothGatt? = null
  @Volatile private var scanning = false
  @Volatile private var txChar: BluetoothGattCharacteristic? = null

  /** 마지막 ECG 프레임 도착 시각(ms) — 패킷 유실 감지용. */
  @Volatile private var lastEcgFrameMs = 0L

  // ── lead-off 휴리스틱 ──────────────────────────────────────────────────
  // 프레임에 전용 플래그가 없어 신호로 추정: 전극 미접촉 시 입력이 레일/비생리학적
  // 값으로 튀어 isPlausibleEcg 에서 폐기된다. ECG 태그 triplet 의 절반 이상이
  // 폐기되는 프레임이 연속되면 lead-off 로 판정(히스테리시스).
  @Volatile private var leadOffActive = false
  private var leadOffBadFrames = 0
  private var leadOffGoodFrames = 0

  /** 프레임 간격 임계(ms). 11샘플/프레임 ≈ 44ms 이므로 ~2.5프레임 이상 비면 유실로 간주. */
  private val GAP_THRESHOLD_MS = 100L

  /** lead-off 진입: 연속 bad 프레임 수 (~220ms). */
  private val LEAD_OFF_ENTER_FRAMES = 5

  /** lead-off 해제: 연속 정상 프레임 수 (~530ms) — 재접촉 후 안정화 여유. */
  private val LEAD_OFF_EXIT_FRAMES = 12

  private val initRunnable = Runnable { sendInitBurst() }

  private val scanCallback = object : ScanCallback() {
    override fun onScanResult(callbackType: Int, result: ScanResult) {
      val dev = result.device ?: return
      if (!matchesTarget(dev)) return
      stopScan()
      log("scan hit: ${dev.address}")
      connect(dev)
    }

    override fun onScanFailed(errorCode: Int) {
      scanning = false
      log("scan failed: $errorCode")
      setLinkState(SensorLinkState.FAILED)
      onConnectionChanged?.invoke(false)
    }
  }

  private val gattCallback = object : BluetoothGattCallback() {
    override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
      if (newState == BluetoothGatt.STATE_CONNECTED) {
        if (status != BluetoothGatt.GATT_SUCCESS) {
          log("connect error status=$status")
          setLinkState(SensorLinkState.FAILED)
          onConnectionChanged?.invoke(false)
          return
        }
        this@HiCardiBleSource.gatt = gatt
        isConnected = true
        setLinkState(SensorLinkState.CONNECTED)
        onConnectionChanged?.invoke(true)
        log("connected: ${gatt.device.address}")
        @Suppress("MissingPermission")
        gatt.discoverServices()
      } else if (newState == BluetoothGatt.STATE_DISCONNECTED) {
        isConnected = false
        setLinkState(SensorLinkState.IDLE)
        onConnectionChanged?.invoke(false)
        log("disconnected (status=$status)")
      }
    }

    override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
      if (status != BluetoothGatt.GATT_SUCCESS) {
        log("discoverServices fail: $status")
        return
      }
      val service = gatt.getService(nusServiceUuid)
      val rx = service?.getCharacteristic(nusRxUuid)
      val tx = service?.getCharacteristic(nusTxUuid)
      if (rx == null || tx == null) {
        log("NUS chars missing")
        return
      }
      txChar = tx
      @Suppress("MissingPermission")
      gatt.setCharacteristicNotification(rx, true)
      val cccd = rx.getDescriptor(cccdUuid)
      if (cccd == null) {
        log("CCCD missing")
        return
      }
      cccd.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
      @Suppress("MissingPermission")
      gatt.writeDescriptor(cccd)
    }

    override fun onDescriptorWrite(
      gatt: BluetoothGatt,
      descriptor: BluetoothGattDescriptor,
      status: Int,
    ) {
      if (descriptor.uuid != cccdUuid) return
      if (status != BluetoothGatt.GATT_SUCCESS) {
        log("CCCD write fail: $status")
        return
      }
      log("notify enabled")
      mainHandler.post(initRunnable)
    }

    @Deprecated("Deprecated in Java")
    override fun onCharacteristicWrite(
      gatt: BluetoothGatt,
      characteristic: BluetoothGattCharacteristic,
      status: Int,
    ) {
    }

    @Deprecated("Deprecated in Java")
    override fun onCharacteristicChanged(
      gatt: BluetoothGatt,
      characteristic: BluetoothGattCharacteristic,
    ) {
      dispatchNotify(characteristic.value)
    }

    override fun onCharacteristicChanged(
      gatt: BluetoothGatt,
      characteristic: BluetoothGattCharacteristic,
      value: ByteArray,
    ) {
      dispatchNotify(value)
    }
  }

  fun start() {
    if (!hasBlePermissions()) {
      log("BLE permission missing")
      setLinkState(SensorLinkState.FAILED)
      onConnectionChanged?.invoke(false)
      return
    }
    stop()
    onlineFilter.reset()
    val adapter = (context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter
    if (adapter == null || !adapter.isEnabled) {
      log("Bluetooth off")
      setLinkState(SensorLinkState.FAILED)
      onConnectionChanged?.invoke(false)
      return
    }

    adapter.bondedDevices?.firstOrNull { matchesTarget(it) }?.let { bonded ->
      log("connect bonded: ${bonded.address}")
      setLinkState(SensorLinkState.CONNECTING)
      connect(bonded)
      return
    }

    try {
      @Suppress("MissingPermission")
      val direct = adapter.getRemoteDevice(targetAddr)
      log("connect direct: $targetAddr")
      setLinkState(SensorLinkState.CONNECTING)
      connect(direct)
      return
    } catch (e: Exception) {
      log("direct connect failed: ${e.message}")
    }

    val scanner = adapter.bluetoothLeScanner
    if (scanner == null) {
      log("BLE scanner unavailable")
      setLinkState(SensorLinkState.FAILED)
      onConnectionChanged?.invoke(false)
      return
    }
    scanning = true
    setLinkState(SensorLinkState.SCANNING)
    log("scanning...")
    @Suppress("MissingPermission")
    scanner.startScan(scanCallback)
  }

  fun stop() {
    stopScan()
    try {
      @Suppress("MissingPermission")
      gatt?.disconnect()
      gatt?.close()
    } catch (_: Throwable) {
    }
    gatt = null
    txChar = null
    mainHandler.removeCallbacks(initRunnable)
    onlineFilter.reset()
    lastEcgFrameMs = 0L
    leadOffActive = false
    leadOffBadFrames = 0
    leadOffGoodFrames = 0
    isConnected = false
    setLinkState(SensorLinkState.IDLE)
  }

  private fun stopScan() {
    if (!scanning) return
    val adapter = (context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter
    @Suppress("MissingPermission")
    adapter?.bluetoothLeScanner?.stopScan(scanCallback)
    scanning = false
  }

  private fun connect(device: BluetoothDevice) {
    stopScan()
    setLinkState(SensorLinkState.CONNECTING)
    @Suppress("MissingPermission")
    gatt?.close()
    @Suppress("MissingPermission")
    gatt = device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
  }

  private fun sendInitBurst() {
    // 연결이 이미 끊겼으면 지연 전송 3건을 스케줄하지 않는다(실제 사용은 람다 내부 재조회).
    if (gatt == null || txChar == null) return
    for ((i, cmd) in initCmds.withIndex()) {
      mainHandler.postDelayed({
        val gg = gatt ?: return@postDelayed
        val t = txChar ?: return@postDelayed
        t.writeType = BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE
        t.value = cmd.toByteArray(StandardCharsets.UTF_8)
        @Suppress("MissingPermission")
        gg.writeCharacteristic(t)
        log("TX: ${cmd.replace("\n", "\\n")}")
      }, i * 80L)
    }
  }

  private fun dispatchNotify(data: ByteArray?) {
    if (data == null) return
    val parsed = parseFrame(data)
    updateLeadOff(parsed)
    if (parsed.ecgRaw.isEmpty() && parsed.temperatureC == null) return
    // 프레임 도착 간격으로 패킷 유실 추정(11샘플/프레임 ≈ 44ms). 임계 초과 시 플롯 선을 끊는다.
    val now = System.currentTimeMillis()
    val gapBefore = parsed.ecgRaw.isNotEmpty() &&
      lastEcgFrameMs != 0L &&
      (now - lastEcgFrameMs) > GAP_THRESHOLD_MS
    if (parsed.ecgRaw.isNotEmpty()) lastEcgFrameMs = now
    val filtered = FloatArray(parsed.ecgRaw.size) { i ->
      val counts = onlineFilter.processOne(parsed.ecgRaw[i].toDouble()).toFloat()
      HiCardiSignal.countsToMv(counts)
    }
    val batch = BleSampleBatch(
      rawCounts = parsed.ecgRaw,
      filtered = filtered,
      rawAdc = parsed.ecgAdc,
      temperatureC = parsed.temperatureC,
      rPeakIndices = parsed.rPeakIndices,
      gapBefore = gapBefore,
      leadOff = leadOffActive,
    )
    // 메인 스레드 경유 없이 바인더 스레드에서 직접 전달 — 초당 ~22회의 불필요한
    // 메인 루퍼 메시지를 없앤다. 소비자(coordinator)는 자기 executor 로 넘긴다.
    onBatch?.invoke(batch)
  }

  /** 프레임 단위 lead-off 판정 갱신. ECG 태그가 없는(텔레메트리-only) 프레임은 판정 제외. */
  private fun updateLeadOff(parsed: ParsedFrame) {
    val ecgTags = parsed.ecgAdc.size + parsed.discardedEcg
    if (ecgTags == 0) return
    val bad = parsed.discardedEcg * 2 >= ecgTags
    if (bad) {
      leadOffBadFrames++
      leadOffGoodFrames = 0
    } else {
      leadOffGoodFrames++
      leadOffBadFrames = 0
    }
    if (!leadOffActive && leadOffBadFrames >= LEAD_OFF_ENTER_FRAMES) {
      leadOffActive = true
      log("lead-off 감지 — 전극 접촉을 확인하세요 (추론 일시중지)")
    } else if (leadOffActive && leadOffGoodFrames >= LEAD_OFF_EXIT_FRAMES) {
      leadOffActive = false
      log("lead-off 해제 — 신호 정상, 추론 재개")
    }
  }

  private data class ParsedFrame(
    val ecgRaw: FloatArray,
    val ecgAdc: FloatArray,
    val temperatureC: Float?,
    val rPeakIndices: IntArray = IntArray(0),
    /** ECG 태그(0xF0/0xFF)였지만 비생리학적 값이라 폐기된 triplet 수. */
    val discardedEcg: Int = 0,
  )

  private fun parseFrame(data: ByteArray): ParsedFrame {
    if (data.size != frameLen) return ParsedFrame(FloatArray(0), FloatArray(0), null)
    if ((data.last().toInt() and 0xFF) != frameEnd) return ParsedFrame(FloatArray(0), FloatArray(0), null)
    val ecgRaw = ArrayList<Float>(nTriplets)
    val ecgAdc = ArrayList<Float>(nTriplets)
    val rPeaks = ArrayList<Int>(2)
    var temperature: Float? = null
    var discarded = 0
    for (i in 0 until nTriplets) {
      val off = start + i * step
      val tag = data[off].toInt() and 0xFF
      // ECG 샘플 태그(0xF0 정상, 0xFF R-peak)만 파형으로 처리.
      // 바이트 5~7의 0xF1(별도 채널)과 38~40 텔레메트리(0x92~0xD1 회전)는 제외.
      if (tag != HiCardiSignal.ECG_TAG_F0 && tag != HiCardiSignal.ECG_TAG_FF) continue
      val hi = data[off + 1].toInt() and 0xFF
      val lo = data[off + 2].toInt() and 0xFF
      val adc = HiCardiSignal.adcFromBytes(hi, lo)
      // 손상/간섭으로 tag 만 ECG 이고 값이 비생리학적이면 폐기(세로 스파이크 근절, 모델·기록 공통).
      if (!HiCardiSignal.isPlausibleEcg(adc)) {
        discarded++
        continue
      }
      ecgAdc.add(adc.toFloat())
      ecgRaw.add(HiCardiSignal.offsetCounts(adc))
      // tag 0xFF = 센서가 온칩에서 검출한 R-peak 표시 샘플
      if (tag == HiCardiSignal.ECG_TAG_FF) rPeaks.add(ecgAdc.size - 1)
    }
    return ParsedFrame(
      ecgRaw = ecgRaw.toFloatArray(),
      ecgAdc = ecgAdc.toFloatArray(),
      temperatureC = temperature,
      rPeakIndices = rPeaks.toIntArray(),
      discardedEcg = discarded,
    )
  }

  private fun matchesTarget(device: BluetoothDevice): Boolean {
    val addr = device.address?.uppercase() ?: ""
    @Suppress("MissingPermission")
    val name = device.name?.uppercase() ?: ""
    return addr == targetAddr.uppercase() || name.contains(targetName.uppercase())
  }

  private fun hasBlePermissions(): Boolean {
    val connect = ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED
    val scan = ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
    return connect && scan
  }

  private fun log(msg: String) {
    onLog?.invoke("[HiCardi BLE] $msg")
  }

  private fun setLinkState(state: SensorLinkState) {
    linkState = state
    onLinkStateChanged?.invoke(state)
  }
}
