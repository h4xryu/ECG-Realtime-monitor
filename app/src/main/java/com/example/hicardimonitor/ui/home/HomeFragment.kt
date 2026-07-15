package com.example.hicardimonitor.ui.home

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.viewpager2.adapter.FragmentStateAdapter
import com.example.hicardimonitor.R
import com.google.android.material.tabs.TabLayoutMediator

class HomeFragment : Fragment() {

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?,
  ): View? = inflater.inflate(R.layout.fragment_home, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val pager = view.findViewById<androidx.viewpager2.widget.ViewPager2>(R.id.home_pager)
    val tabs = view.findViewById<com.google.android.material.tabs.TabLayout>(R.id.home_tabs)
    pager.adapter = HomePagerAdapter(this)
    TabLayoutMediator(tabs, pager) { tab, pos ->
      tab.text = when (pos) {
        0 -> getString(R.string.tab_monitor)
        1 -> getString(R.string.tab_short_term)
        2 -> getString(R.string.tab_record)
        else -> getString(R.string.tab_personalization)
      }
    }.attach()
  }

  private class HomePagerAdapter(host: Fragment) : FragmentStateAdapter(host) {
    override fun getItemCount(): Int = 4
    override fun createFragment(position: Int): Fragment = when (position) {
      0 -> MonitorFragment()
      1 -> ShortTermFragment()
      2 -> RecordFragment()
      else -> PersonalizationFragment()
    }
  }
}
