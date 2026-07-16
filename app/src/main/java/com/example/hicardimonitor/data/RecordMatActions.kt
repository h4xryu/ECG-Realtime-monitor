package com.example.hicardimonitor.data

import android.app.Activity
import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import com.example.hicardimonitor.R
import java.io.File

object RecordMatActions {

  fun ensureMatFile(record: DiagnosisRecord, outFile: File) {
    if (!outFile.exists() || outFile.length() == 0L) {
      MatlabExporter.export(record, outFile)
    }
  }

  fun matUri(context: Context, file: File): Uri =
    FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)

  fun shareMat(activity: Activity, file: File, displayName: String) {
    val uri = matUri(activity, file)
    val intent = Intent(Intent.ACTION_SEND).apply {
      type = "*/*"
      putExtra(Intent.EXTRA_STREAM, uri)
      putExtra(Intent.EXTRA_SUBJECT, displayName)
      putExtra(Intent.EXTRA_TITLE, displayName)
      addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
      clipData = ClipData.newUri(activity.contentResolver, displayName, uri)
    }
    activity.startActivity(
      Intent.createChooser(intent, activity.getString(R.string.share_chooser_title)),
    )
  }

  fun createSaveIntent(fileName: String, mime: String = "application/octet-stream"): Intent =
    Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
      addCategory(Intent.CATEGORY_OPENABLE)
      type = mime
      putExtra(Intent.EXTRA_TITLE, fileName)
    }

  /** 임의 파일을 FileProvider 로 공유(ACTION_SEND). PDF strip-view 등. */
  fun shareFile(activity: Activity, file: File, displayName: String, mime: String) {
    val uri = matUri(activity, file)
    val intent = Intent(Intent.ACTION_SEND).apply {
      type = mime
      putExtra(Intent.EXTRA_STREAM, uri)
      putExtra(Intent.EXTRA_SUBJECT, displayName)
      putExtra(Intent.EXTRA_TITLE, displayName)
      addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
      clipData = ClipData.newUri(activity.contentResolver, displayName, uri)
    }
    activity.startActivity(
      Intent.createChooser(intent, activity.getString(R.string.share_chooser_title)),
    )
  }

  fun writeToUri(context: Context, uri: Uri, file: File) {
    context.contentResolver.openOutputStream(uri)?.use { out ->
      file.inputStream().use { input -> input.copyTo(out) }
    } ?: error("저장 위치를 열 수 없습니다")
  }

  fun displayFileName(record: DiagnosisRecord): String {
    val prefix = if (record.isPersonalization) "hicardi_personalization" else "hicardi_record"
    return "${prefix}_${record.id}.mat"
  }
}
