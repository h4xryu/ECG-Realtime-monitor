package com.example.hicardimonitor.ui.home

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.hicardimonitor.R
import com.example.hicardimonitor.data.DiagnosisRecordMeta
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object RecordListAdapter {

  sealed class Row {
    data class MonthHeader(val label: String) : Row()
    data class Entry(val meta: DiagnosisRecordMeta) : Row()
  }

  fun groupByMonth(items: List<DiagnosisRecordMeta>): List<Row> {
    if (items.isEmpty()) return emptyList()
    val monthFmt = SimpleDateFormat("yyyy.MM", Locale.KOREA)
    val out = ArrayList<Row>()
    var lastMonth: String? = null
    for (item in items.sortedByDescending { it.startedAtMs }) {
      val month = monthFmt.format(Date(item.startedAtMs))
      if (month != lastMonth) {
        out.add(Row.MonthHeader(month))
        lastMonth = month
      }
      out.add(Row.Entry(item))
    }
    return out
  }

  class Adapter(
    private val onClick: (DiagnosisRecordMeta) -> Unit,
    private val showSummary: Boolean = false,
  ) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val rows = ArrayList<Row>()
    private val timeFmt = SimpleDateFormat("M월 d일 a hh:mm", Locale.KOREA)

    fun submit(items: List<DiagnosisRecordMeta>) {
      rows.clear()
      rows.addAll(groupByMonth(items))
      notifyDataSetChanged()
    }

    override fun getItemViewType(position: Int): Int = when (rows[position]) {
      is Row.MonthHeader -> 0
      is Row.Entry -> 1
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
      val inf = LayoutInflater.from(parent.context)
      return if (viewType == 0) {
        MonthVH(inf.inflate(R.layout.item_record_month, parent, false))
      } else {
        EntryVH(inf.inflate(R.layout.item_record, parent, false))
      }
    }

    override fun getItemCount(): Int = rows.size

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
      when (val row = rows[position]) {
        is Row.MonthHeader -> (holder as MonthVH).month.text = row.label
        is Row.Entry -> {
          val h = holder as EntryVH
          val ctx = h.itemView.context
          h.time.text = timeFmt.format(Date(row.meta.startedAtMs))
          h.bpm.text = when {
            row.meta.isPersonalization -> ctx.getString(R.string.record_type_personalization)
            showSummary && row.meta.summary.isNotBlank() -> row.meta.summary
            row.meta.bpm > 0 -> "${row.meta.bpm} BPM"
            else -> row.meta.summary.ifBlank { "-" }
          }
          h.detail.text = when {
            row.meta.isPersonalization -> ctx.getString(R.string.record_type_personalization_detail)
            else -> ctx.getString(R.string.view_diagnosis_detail)
          }
          h.itemView.setOnClickListener { onClick(row.meta) }
        }
      }
    }

    private class MonthVH(v: View) : RecyclerView.ViewHolder(v) {
      val month: TextView = v.findViewById(R.id.monthHeader)
    }

    private class EntryVH(v: View) : RecyclerView.ViewHolder(v) {
      val time: TextView = v.findViewById(R.id.recordTime)
      val detail: TextView = v.findViewById(R.id.recordDetail)
      val bpm: TextView = v.findViewById(R.id.recordBpm)
    }
  }
}
