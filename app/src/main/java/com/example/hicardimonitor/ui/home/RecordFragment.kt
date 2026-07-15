package com.example.hicardimonitor.ui.home

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.hicardimonitor.R
import com.example.hicardimonitor.data.RecordMatActions
import com.example.hicardimonitor.ui.MainActivity
import com.example.hicardimonitor.ui.record.RecordDetailActivity

class RecordFragment : Fragment() {

  private lateinit var adapter: RecordListAdapter.Adapter

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?,
  ): View? = inflater.inflate(R.layout.fragment_record, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    adapter = RecordListAdapter.Adapter(onClick = { meta ->
      startActivity(
        Intent(requireContext(), RecordDetailActivity::class.java)
          .putExtra(RecordDetailActivity.EXTRA_ID, meta.id),
      )
    })
    view.findViewById<RecyclerView>(R.id.recordList).apply {
      layoutManager = LinearLayoutManager(requireContext())
      adapter = this@RecordFragment.adapter
    }
    view.findViewById<Button>(R.id.uploadFab).setOnClickListener {
      shareLatestRecord()
    }
    // 목록은 onResume 에서 비동기 로드 — ViewPager 프리로드 시 불필요한 작업 방지.
  }

  override fun onResume() {
    super.onResume()
    refresh()
  }

  private fun refresh() {
    val coord = (requireActivity() as MainActivity).coordinator
    coord.recordRepo.listRecordsAsync { items ->
      if (!isAdded) return@listRecordsAsync
      adapter.submit(items.filter { !it.isShortTerm })
    }
  }

  private fun shareLatestRecord() {
    val coord = (requireActivity() as MainActivity).coordinator
    coord.recordRepo.listRecordsAsync { items ->
      if (!isAdded) return@listRecordsAsync
      val latest = items.firstOrNull { !it.isShortTerm }
      if (latest == null) {
        Toast.makeText(requireContext(), R.string.share_no_records, Toast.LENGTH_SHORT).show()
        return@listRecordsAsync
      }
      // full 레코드 로드 + .mat 생성(수 MB)은 IO 스레드에서.
      coord.recordRepo.prepareMatAsync(latest.id) { record, matFile ->
        if (!isAdded) return@prepareMatAsync
        if (record == null || matFile == null) {
          Toast.makeText(requireContext(), "기록을 불러올 수 없습니다", Toast.LENGTH_SHORT).show()
          return@prepareMatAsync
        }
        RecordMatActions.shareMat(requireActivity(), matFile, RecordMatActions.displayFileName(record))
      }
    }
  }
}
