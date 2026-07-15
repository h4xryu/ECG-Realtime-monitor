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

class PersonalizationFragment : Fragment() {

  private lateinit var adapter: RecordListAdapter.Adapter
  private lateinit var emptyView: TextView

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?,
  ): View? = inflater.inflate(R.layout.fragment_personalization, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val coord = (requireActivity() as MainActivity).coordinator
    val accuracyText = view.findViewById<TextView>(R.id.cpcAccuracyText)
    val statusText = view.findViewById<TextView>(R.id.cpcStatusText)
    emptyView = view.findViewById(R.id.cpcHistoryEmpty)

    adapter = RecordListAdapter.Adapter(
      onClick = { meta ->
        startActivity(
          Intent(requireContext(), RecordDetailActivity::class.java)
            .putExtra(RecordDetailActivity.EXTRA_ID, meta.id),
        )
      },
      showSummary = true,
    )
    view.findViewById<RecyclerView>(R.id.cpcHistoryList).apply {
      layoutManager = LinearLayoutManager(requireContext())
      adapter = this@PersonalizationFragment.adapter
    }

    val dim = coord.patientEmbeddingSize()
    if (dim > 0) {
      val q = coord.personalizationQuality
      accuracyText.text = if (q != null) "적합도 ${q.fitScore} / 100" else "학습 완료"
      statusText.text = "당신의 리듬(${dim}차원)이 이후 분석에 적용되고 있습니다."
    }

    view.findViewById<Button>(R.id.startCpcButton).setOnClickListener {
      if (!coord.bleSource.isConnected) {
        statusText.text = getString(R.string.sensor_disconnected)
        (requireActivity() as MainActivity).selectNav(R.id.nav_sensor)
        return@setOnClickListener
      }
      startActivity(
        Intent(requireContext(), EcgMeasurementActivity::class.java)
          .putExtra(EcgMeasurementActivity.EXTRA_MODE, EcgMeasurementActivity.MODE_CPC)
          .putExtra(EcgMeasurementActivity.EXTRA_SECONDS, 60),
      )
    }
    // 목록은 onResume 에서 비동기 로드 — ViewPager 프리로드 시 불필요한 작업 방지.
  }

  override fun onResume() {
    super.onResume()
    val coord = (requireActivity() as MainActivity).coordinator
    view?.findViewById<TextView>(R.id.cpcAccuracyText)?.let { tv ->
      val dim = coord.patientEmbeddingSize()
      if (dim > 0) {
        val q = coord.personalizationQuality
        tv.text = if (q != null) "적합도 ${q.fitScore} / 100" else "학습 완료"
      }
    }
    refresh()
  }

  private fun refresh() {
    val coord = (requireActivity() as MainActivity).coordinator
    coord.recordRepo.listRecordsAsync { items ->
      if (!isAdded) return@listRecordsAsync
      val filtered = items.filter { it.isPersonalization }
      adapter.submit(filtered)
      emptyView.visibility = if (filtered.isEmpty()) View.VISIBLE else View.GONE
    }
  }
}
