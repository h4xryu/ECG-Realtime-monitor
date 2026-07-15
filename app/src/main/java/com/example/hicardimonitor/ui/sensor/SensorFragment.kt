package com.example.hicardimonitor.ui.sensor

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.hicardimonitor.R
import com.example.hicardimonitor.SensorLinkState
import com.example.hicardimonitor.ui.MainActivity

class SensorFragment : Fragment() {

  private lateinit var statusBanner: LinearLayout
  private lateinit var statusIcon: ImageView
  private lateinit var statusProgress: ProgressBar
  private lateinit var statusText: TextView
  private lateinit var deviceCard: LinearLayout
  private lateinit var deviceName: TextView
  private lateinit var deviceAddr: TextView
  private lateinit var deviceConnectedIcon: ImageView
  private lateinit var deviceAction: TextView
  private lateinit var helpText: TextView

  private val connListener: (Boolean) -> Unit = conn@{ _ ->
    if (isAdded) renderState()
  }

  private val linkListener: (SensorLinkState) -> Unit = link@{ _ ->
    if (isAdded) renderState()
  }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?,
  ): View? = inflater.inflate(R.layout.fragment_sensor, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val coord = (requireActivity() as MainActivity).coordinator
    statusBanner = view.findViewById(R.id.sensorStatusBanner)
    statusIcon = view.findViewById(R.id.sensorStatusIcon)
    statusProgress = view.findViewById(R.id.sensorProgress)
    statusText = view.findViewById(R.id.sensorStatusText)
    deviceCard = view.findViewById(R.id.deviceCard)
    deviceName = view.findViewById(R.id.deviceNameText)
    deviceAddr = view.findViewById(R.id.deviceAddrText)
    deviceConnectedIcon = view.findViewById(R.id.deviceConnectedIcon)
    deviceAction = view.findViewById(R.id.deviceActionText)
    helpText = view.findViewById(R.id.sensorHelpText)

    deviceName.text = coord.bleSource.targetDisplayName
    deviceAddr.text = coord.bleSource.targetDisplayAddress

    deviceCard.setOnClickListener {
      if (coord.bleSource.isConnected) {
        coord.disconnectBle()
      } else {
        coord.connectBle()
      }
    }

    coord.addConnectionListener(connListener)
    coord.addLinkStateListener(linkListener)
    renderState()
  }

  private fun renderState() {
    val coord = (activity as? MainActivity)?.coordinator ?: return
    val ble = coord.bleSource
    val ctx = requireContext()

    when (ble.linkState) {
      SensorLinkState.CONNECTED -> {
        statusProgress.visibility = View.GONE
        statusIcon.visibility = View.VISIBLE
        statusIcon.setImageResource(R.drawable.ic_connected_check)
        statusText.text = getString(R.string.sensor_state_connected)
        statusText.setTextColor(ContextCompat.getColor(ctx, R.color.status_ok))
        deviceConnectedIcon.visibility = View.VISIBLE
        deviceAction.text = getString(R.string.sensor_tap_disconnect)
        deviceAction.setTextColor(ContextCompat.getColor(ctx, R.color.text_secondary))
        helpText.text = getString(R.string.sensor_help_connected)
      }
      SensorLinkState.SCANNING -> {
        statusProgress.visibility = View.VISIBLE
        statusIcon.visibility = View.GONE
        statusText.text = getString(R.string.sensor_state_scanning)
        statusText.setTextColor(ContextCompat.getColor(ctx, R.color.tab_selected))
        deviceConnectedIcon.visibility = View.GONE
        deviceAction.text = getString(R.string.sensor_connecting)
        helpText.text = getString(R.string.sensor_help_scanning)
      }
      SensorLinkState.CONNECTING -> {
        statusProgress.visibility = View.VISIBLE
        statusIcon.visibility = View.GONE
        statusText.text = getString(R.string.sensor_state_connecting)
        statusText.setTextColor(ContextCompat.getColor(ctx, R.color.tab_selected))
        deviceConnectedIcon.visibility = View.GONE
        deviceAction.text = getString(R.string.sensor_connecting)
        helpText.text = getString(R.string.sensor_help_connecting)
      }
      SensorLinkState.FAILED -> {
        statusProgress.visibility = View.GONE
        statusIcon.visibility = View.VISIBLE
        statusIcon.setImageResource(R.drawable.ic_sensor_ble)
        statusText.text = getString(R.string.sensor_state_failed)
        statusText.setTextColor(ContextCompat.getColor(ctx, R.color.text_error))
        deviceConnectedIcon.visibility = View.GONE
        deviceAction.text = getString(R.string.sensor_tap_connect)
        helpText.text = getString(R.string.sensor_help_failed)
      }
      else -> {
        statusProgress.visibility = View.GONE
        statusIcon.visibility = View.VISIBLE
        statusIcon.setImageResource(R.drawable.ic_sensor_ble)
        statusText.text = getString(R.string.sensor_state_idle)
        statusText.setTextColor(ContextCompat.getColor(ctx, R.color.text_secondary))
        deviceConnectedIcon.visibility = View.GONE
        deviceAction.text = getString(R.string.sensor_tap_connect)
        deviceAction.setTextColor(ContextCompat.getColor(ctx, R.color.tab_selected))
        helpText.text = getString(R.string.sensor_help_disconnected)
      }
    }
  }

  override fun onDestroyView() {
    val coord = (activity as? MainActivity)?.coordinator
    coord?.removeConnectionListener(connListener)
    coord?.removeLinkStateListener(linkListener)
    super.onDestroyView()
  }
}
