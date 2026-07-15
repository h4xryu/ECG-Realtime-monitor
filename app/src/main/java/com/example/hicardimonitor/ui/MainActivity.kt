package com.example.hicardimonitor.ui

import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.widget.TextViewCompat
import androidx.fragment.app.Fragment
import com.example.hicardimonitor.HiCardiCoordinator
import com.example.hicardimonitor.R
import com.example.hicardimonitor.ui.home.HomeFragment
import com.example.hicardimonitor.ui.sensor.SensorFragment
import com.example.hicardimonitor.ui.settings.SettingsFragment

class MainActivity : AppCompatActivity() {

  lateinit var coordinator: HiCardiCoordinator
    private set

  private var activeTag: String = TAG_SENSOR

  private lateinit var navHome: TextView
  private lateinit var navSensor: TextView
  private lateinit var navSettings: TextView

  companion object {
    private const val TAG_HOME = "home"
    private const val TAG_SENSOR = "sensor"
    private const val TAG_SETTINGS = "settings"
    private const val STATE_TAG = "active_nav_tag"

    @Volatile
    private var holder: MainActivity? = null

    fun requireCoordinator() = peekCoordinator()
      ?: error("MainActivity is not running")

    fun peekCoordinator() = holder?.coordinator
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    holder = this
    setContentView(R.layout.activity_main)

    val root = findViewById<View>(R.id.main)
    ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
      v.setPadding(v.paddingLeft, bars.top, v.paddingRight, bars.bottom)
      insets
    }

    coordinator = HiCardiCoordinator(this)
    coordinator.attach()
    coordinator.onNavigateToSensor = { selectNav(R.id.nav_sensor) }

    navHome = findViewById(R.id.nav_home)
    navSensor = findViewById(R.id.nav_sensor)
    navSettings = findViewById(R.id.nav_settings)

    navHome.setOnClickListener { selectNav(R.id.nav_home) }
    navSensor.setOnClickListener { selectNav(R.id.nav_sensor) }
    navSettings.setOnClickListener { selectNav(R.id.nav_settings) }

    if (savedInstanceState == null) {
      selectNav(R.id.nav_sensor)
    } else {
      activeTag = savedInstanceState.getString(STATE_TAG, TAG_SENSOR)
      val navId = when (activeTag) {
        TAG_HOME -> R.id.nav_home
        TAG_SETTINGS -> R.id.nav_settings
        else -> R.id.nav_sensor
      }
      highlightBottomNav(navId)
      if (supportFragmentManager.findFragmentById(R.id.nav_host) == null) {
        selectNav(navId)
      }
    }
  }

  fun selectNav(itemId: Int) {
    highlightBottomNav(itemId)
    when (itemId) {
      R.id.nav_home -> showRoot(HomeFragment(), TAG_HOME)
      R.id.nav_sensor -> showRoot(SensorFragment(), TAG_SENSOR)
      R.id.nav_settings -> showRoot(SettingsFragment(), TAG_SETTINGS)
    }
  }

  private fun highlightBottomNav(selectedId: Int) {
    val active = R.style.HiCardiBottomNav_Active
    val inactive = R.style.HiCardiBottomNav_Inactive
    listOf(
      navHome to R.id.nav_home,
      navSensor to R.id.nav_sensor,
      navSettings to R.id.nav_settings,
    ).forEach { (view, id) ->
      val style = if (id == selectedId) active else inactive
      TextViewCompat.setTextAppearance(view, style)
    }
  }

  private fun showRoot(fragment: Fragment, tag: String) {
    val current = supportFragmentManager.findFragmentById(R.id.nav_host)
    if (tag == activeTag && current != null) return
    activeTag = tag
    if (supportFragmentManager.isStateSaved) {
      supportFragmentManager.beginTransaction()
        .replace(R.id.nav_host, fragment, tag)
        .commitAllowingStateLoss()
    } else {
      supportFragmentManager.beginTransaction()
        .replace(R.id.nav_host, fragment, tag)
        .commitNow()
    }
  }

  override fun onSaveInstanceState(outState: Bundle) {
    super.onSaveInstanceState(outState)
    outState.putString(STATE_TAG, activeTag)
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<out String>,
    grantResults: IntArray,
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == HiCardiCoordinator.BLE_PERM_REQUEST) {
      val granted = grantResults.isNotEmpty() &&
        grantResults.all { it == android.content.pm.PackageManager.PERMISSION_GRANTED }
      coordinator.onBlePermissionsResult(granted)
    }
  }

  override fun onDestroy() {
    if (holder === this) holder = null
    coordinator.destroy()
    super.onDestroy()
  }
}
