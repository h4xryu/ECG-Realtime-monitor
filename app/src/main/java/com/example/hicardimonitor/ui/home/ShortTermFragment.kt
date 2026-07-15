package com.example.hicardimonitor.ui.home

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.hicardimonitor.R
import com.example.hicardimonitor.ui.MainActivity
import com.example.hicardimonitor.ui.measurement.EcgMeasurementActivity
import com.example.hicardimonitor.ui.record.RecordDetailActivity

class ShortTermFragment : Fragment() {

  private lateinit var adapter: RecordListAdapter.Adapter
  private lateinit var emptyView: TextView

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?,
  ): View? = inflater.inflate(R.layout.fragment_short_term, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val coord = (requireActivity() as MainActivity).coordinator
    emptyView = view.findViewById(R.id.shortTermEmpty)
    adapter = RecordListAdapter.Adapter(onClick = { meta ->
      startActivity(
        Intent(requireContext(), RecordDetailActivity::class.java)
          .putExtra(RecordDetailActivity.EXTRA_ID, meta.id),
      )
    })
    view.findViewById<RecyclerView>(R.id.shortTermList).apply {
      layoutManager = LinearLayoutManager(requireContext())
      adapter = this@ShortTermFragment.adapter
    }
    view.findViewById<Button>(R.id.startMeasureFab).setOnClickListener {
      if (!coord.bleSource.isConnected) {
        (requireActivity() as MainActivity).selectNav(R.id.nav_sensor)
        return@setOnClickListener
      }
      startActivity(
        Intent(requireContext(), EcgMeasurementActivity::class.java)
          .putExtra(EcgMeasurementActivity.EXTRA_MODE, EcgMeasurementActivity.MODE_RECORDING),
      )
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
      val filtered = items.filter { it.isShortTerm }
      adapter.submit(filtered)
      emptyView.visibility = if (filtered.isEmpty()) View.VISIBLE else View.GONE
    }
  }
}
