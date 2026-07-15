package com.example.hicardimonitor.data

import android.content.Context
import android.os.Handler
import android.os.Looper
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors

class RecordRepository(private val context: Context) {

  private val dir: File
    get() = File(context.filesDir, "records").also { it.mkdirs() }

  /**
   * 목록 메타 — 레코드 본문(.json, 수 MB 샘플 배열 포함)은 파싱하지 않고
   * 사이드카(.meta.json, 수백 바이트)만 읽는다. 사이드카가 없는 구버전 저장분은
   * 1회 full 파싱 후 사이드카를 만들어 자가 치유한다.
   *
   * 마이그레이션이 낄 수 있으므로 메인 스레드에서는 [listRecordsAsync] 를 쓸 것.
   */
  fun listRecords(): List<DiagnosisRecordMeta> {
    return recordFiles()
      .mapNotNull { metaFor(it) }
      .sortedByDescending { it.startedAtMs }
  }

  /** IO 스레드에서 목록 생성 → 메인 스레드 콜백. */
  fun listRecordsAsync(onResult: (List<DiagnosisRecordMeta>) -> Unit) {
    io.execute {
      val items = listRecords()
      mainHandler.post { onResult(items) }
    }
  }

  /** 레코드 로드 + .mat 생성까지 IO 스레드에서 수행 → 메인 콜백(실패 시 null). */
  fun prepareMatAsync(id: String, onResult: (DiagnosisRecord?, File?) -> Unit) {
    io.execute {
      val record = load(id)
      val file = record?.let {
        try {
          val f = matFileFor(id)
          RecordMatActions.ensureMatFile(it, f)
          f
        } catch (_: Exception) {
          null
        }
      }
      mainHandler.post { onResult(record, file) }
    }
  }

  fun save(record: DiagnosisRecord) {
    val file = File(dir, "${record.id}.json")
    val obj = JSONObject()
    obj.put("id", record.id)
    obj.put("startedAtMs", record.startedAtMs)
    obj.put("endedAtMs", record.endedAtMs)
    obj.put("fs", record.fs)
    obj.put("bpm", record.bpm)
    obj.put("summary", record.summary)
    obj.put("isShortTerm", record.isShortTerm)
    obj.put("isPersonalization", record.isPersonalization)
    val reconnect = JSONArray()
    record.reconnectEvents.forEach { reconnect.put(it) }
    obj.put("reconnectEvents", reconnect)
    val blocks = JSONArray()
    for (b in record.blocks) {
      val bo = JSONObject()
      bo.put("filtered", JSONArray(b.filtered.toList()))
      bo.put("rawAdc", JSONArray(b.rawAdc.toList()))
      val flags = JSONArray()
      b.perSampleFlags.forEach { flags.put(JSONArray(it.toList())) }
      bo.put("perSampleFlags", flags)
      bo.put("rPeakIndices", JSONArray(b.rPeakIndices.toList()))
      bo.put("dataLossIndices", JSONArray(b.dataLossIndices.toList()))
      bo.put("leadOffIndices", JSONArray(b.leadOffIndices.toList()))
      blocks.put(bo)
    }
    obj.put("blocks", blocks)
    file.writeText(obj.toString())
    writeMeta(
      DiagnosisRecordMeta(
        id = record.id,
        startedAtMs = record.startedAtMs,
        bpm = record.bpm,
        summary = record.summary,
        isShortTerm = record.isShortTerm,
        isPersonalization = record.isPersonalization,
      ),
    )
  }

  fun load(id: String): DiagnosisRecord? {
    val file = File(dir, "$id.json")
    if (!file.exists()) return null
    return parseRecord(JSONObject(file.readText()))
  }

  fun delete(id: String) {
    File(dir, "$id.json").delete()
    File(dir, "$id.mat").delete()
    metaFile(id).delete()
  }

  fun matFileFor(id: String): File = File(dir, "$id.mat")

  fun newId(): String = UUID.randomUUID().toString().take(8)

  private fun recordFiles(): List<File> =
    dir.listFiles { f -> f.name.endsWith(".json") && !f.name.endsWith(META_SUFFIX) }
      ?.toList()
      ?: emptyList()

  private fun metaFile(id: String): File = File(dir, "$id$META_SUFFIX")

  private fun metaFor(recordFile: File): DiagnosisRecordMeta? {
    val id = recordFile.name.removeSuffix(".json")
    val mf = metaFile(id)
    if (mf.exists()) {
      try {
        return parseMetaObj(JSONObject(mf.readText()))
      } catch (_: Exception) {
        // 손상된 사이드카 → 아래에서 본문으로 재생성
      }
    }
    // 사이드카 없음(구버전 저장분): 1회 full 파싱 후 사이드카 생성.
    // 메타 필드들은 레코드 본문 JSON 의 최상위 필드와 동일하다.
    return try {
      val meta = parseMetaObj(JSONObject(recordFile.readText()))
      writeMeta(meta)
      meta
    } catch (_: Exception) {
      null
    }
  }

  private fun writeMeta(meta: DiagnosisRecordMeta) {
    val obj = JSONObject()
      .put("id", meta.id)
      .put("startedAtMs", meta.startedAtMs)
      .put("bpm", meta.bpm)
      .put("summary", meta.summary)
      .put("isShortTerm", meta.isShortTerm)
      .put("isPersonalization", meta.isPersonalization)
    metaFile(meta.id).writeText(obj.toString())
  }

  private fun parseMetaObj(obj: JSONObject): DiagnosisRecordMeta = DiagnosisRecordMeta(
    id = obj.getString("id"),
    startedAtMs = obj.getLong("startedAtMs"),
    bpm = obj.optInt("bpm", 0),
    summary = obj.optString("summary", ""),
    isShortTerm = obj.optBoolean("isShortTerm", false),
    isPersonalization = obj.optBoolean("isPersonalization", false),
  )

  private fun parseRecord(obj: JSONObject): DiagnosisRecord {
    val blocksArr = obj.getJSONArray("blocks")
    val blocks = ArrayList<RecordSampleBlock>(blocksArr.length())
    for (i in 0 until blocksArr.length()) {
      val bo = blocksArr.getJSONObject(i)
      val filtered = jsonToFloatArray(bo.getJSONArray("filtered"))
      val raw = jsonToFloatArray(bo.getJSONArray("rawAdc"))
      val flagsArr = bo.getJSONArray("perSampleFlags")
      val flags = Array(flagsArr.length()) { j ->
        jsonToFloatArray(flagsArr.getJSONArray(j))
      }
      blocks.add(
        RecordSampleBlock(
          filtered = filtered,
          rawAdc = raw,
          perSampleFlags = flags,
          rPeakIndices = jsonToIntArray(bo.getJSONArray("rPeakIndices")),
          dataLossIndices = jsonToIntArray(bo.getJSONArray("dataLossIndices")),
          leadOffIndices = jsonToIntArray(bo.getJSONArray("leadOffIndices")),
        ),
      )
    }
    val reconnect = mutableListOf<Long>()
    val reconnArr = obj.optJSONArray("reconnectEvents")
    if (reconnArr != null) {
      for (i in 0 until reconnArr.length()) reconnect.add(reconnArr.getLong(i))
    }
    return DiagnosisRecord(
      id = obj.getString("id"),
      startedAtMs = obj.getLong("startedAtMs"),
      endedAtMs = obj.getLong("endedAtMs"),
      fs = obj.getInt("fs"),
      bpm = obj.optInt("bpm", 0),
      summary = obj.optString("summary", ""),
      blocks = blocks,
      isShortTerm = obj.optBoolean("isShortTerm", false),
      isPersonalization = obj.optBoolean("isPersonalization", false),
      reconnectEvents = reconnect,
    )
  }

  private fun jsonToFloatArray(arr: JSONArray): FloatArray {
    return FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
  }

  private fun jsonToIntArray(arr: JSONArray): IntArray {
    return IntArray(arr.length()) { i -> arr.getInt(i) }
  }

  companion object {
    private const val META_SUFFIX = ".meta.json"

    /** 목록/로드/내보내기용 공용 IO 스레드 — 메인 스레드에서 디스크·JSON 파싱 금지. */
    private val io = Executors.newSingleThreadExecutor { r ->
      Thread(r, "RecordRepoIO").apply { isDaemon = true }
    }
    private val mainHandler = Handler(Looper.getMainLooper())
  }
}

data class DiagnosisRecordMeta(
  val id: String,
  val startedAtMs: Long,
  val bpm: Int,
  val summary: String,
  val isShortTerm: Boolean,
  val isPersonalization: Boolean = false,
)
