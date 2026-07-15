package com.example.hicardimonitor

/**
 * 27-class 멀티라벨 체계 (슬라이드 Rthm_label 대응).
 * 각 행: [labelId, classIndex, reserved]
 */
object HiCardiLabels {

  val BEAT_CLASSES = listOf(
    "Normal",
    "Pause",
    "Ventricular_Fib_Tach",
    "Ventricular_Trigeminy",
    "Ventricular_Bigeminy",
    "Ventricular_Couplet",
    "Ventricular_PAC",
    "Ventricular_PNC",
    "Ventricular_Rhythm",
    "Supraventricular_Tachy",
    "Supraventricular_Bradycardia",
    "Atrial_Fibrillation_Flutter",
    "Atrial_Tachy",
    "Junctional",
    "Idioventricular",
    "Paced",
    "Fusion",
    "PVC",
    "APC",
    "Aberrant",
    "Sinus_Tachy",
    "Sinus_Bradycardia",
    "Sinus_Arrhythmia",
    "ST_Change",
    "T_Wave_Abnormal",
    "Noise",
    "Unknown",
  )

  const val LABEL_DATA_LOSS = 96
  const val LABEL_LEAD_OFF = 98
  const val LABEL_UNKNOWN = 100

  fun rhythmLabelMatrix(): Array<DoubleArray> {
    return Array(BEAT_CLASSES.size) { i ->
      doubleArrayOf(i.toDouble(), i.toDouble(), 0.0)
    }
  }

  fun classIndex(name: String): Int = BEAT_CLASSES.indexOfFirst { it.equals(name, ignoreCase = true) }
}
