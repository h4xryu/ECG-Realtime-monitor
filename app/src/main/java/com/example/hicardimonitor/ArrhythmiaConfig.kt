package com.example.hicardimonitor

/**
 * ResUMamba 부정맥 분류기 온디바이스 계약(DEPLOYMENT_ONDEVICE_SPEC.md 기준).
 *
 * classifier.pte forward(x, patient_vec, clinical_feats) → logits[6]
 *   x            : float32 [1, 1, 501]  beat crop, raw mV (ADC→mV, z-score 없음)
 *   patient_vec  : float32 [1, 256]     CPC pooled patient embedding
 *   clinical_feats: float32 [1, 32]     ClinicalFeatures.compute
 *   logits       : float32 [1, 6] → sigmoid → 멀티라벨 (threshold 0.5)
 *
 * 모델은 6 로짓(서맥 포함)을 내지만 앱은 서맥 로짓([BRADY_LOGIT_INDEX])을 폐기하고
 * 5클래스(CLASS_NAMES)만 표시한다. 서맥은 BPM rule 로 별도 판정(ArrhythmiaResult.bradycardia).
 *
 * ADC 스케일: HiCardi 센서 offset = 16384(0x4000)이며 BleSampleBatch.rawCounts 는 이미
 * (adc - 16384) offset counts 이므로 모델 mV = rawCounts / ADC_SCALE_MV.
 */
object ArrhythmiaConfig {

  /** assets 에 배치할 분류기 파일명 (Linux resumamba 환경에서 export). */
  const val CLASSIFIER_PTE_FILE = "classifier.pte"

  const val BEAT_PRE = 250
  const val BEAT_POST = 250

  /** BEAT_PRE + 1 + BEAT_POST */
  const val BEAT_LEN = 501

  const val CLINICAL_DIM = 32

  /** 모델(.pte) 출력 로짓 수 — 서맥 포함 6 (deployment 계약, 변경 불가). */
  const val MODEL_NUM_LOGITS = 6

  /**
   * 앱 표시 클래스 수 = CLASS_NAMES.size. 서맥은 모델 로짓을 폐기하고 BPM rule 로
   * 별도 판정하므로 표시 목록에서 제외된다(모델 로짓 [4] 드롭, VPC 5→4 로 시프트).
   */
  const val NUM_CLASSES = 5

  const val PATIENT_DIM = 256

  const val THRESHOLD = 0.5f

  /** mV = offsetCounts / 1000 (spec ADC_SCALE=1000). offset 16384 은 rawCounts 에서 이미 제거됨. */
  const val ADC_SCALE_MV = 1000f

  const val EPS = 1e-8f

  const val NORMAL_INDEX = 0

  /**
   * 모델 로짓에서의 서맥 인덱스(폐기 대상). 서맥은 정의 자체가 심박수 기준(HR < 60)이라
   * 모델 출력을 쓰지 않고 BPM rule 로 판정하며, 표시 클래스(CLASS_NAMES)에서도 제외된다.
   * 표시 확률 벡터는 이 로짓을 드롭하고 VPC 를 5→4 로 시프트한 5-vector.
   */
  const val BRADY_LOGIT_INDEX = 4

  /** 서맥 rule 표시용 라벨(표시 클래스 목록 밖). */
  const val BRADY_LABEL_KO = "서맥"

  /** 서맥 진입 기준 — BPM 이 이 값 미만이면 서맥(rule-based). */
  const val BRADYCARDIA_BPM_MAX = 60

  /** 서맥 해제 기준 — BPM 이 이 값 이상이면 해제. 경계(60 부근) 플리커 방지. */
  const val BRADYCARDIA_CLEAR_BPM = 62


  val CLASS_NAMES = listOf(
    "Normal",
    "Sinus_Tachy",
    "APC",
    "AF_AFL",
    "VPC",
  )

  val CLASS_LABELS_KO = listOf(
    "정상",
    "동성빈맥",
    "심방조기수축",
    "심방조/세동",
    "심실조기수축",
  )

  fun labelKo(index: Int): String = CLASS_LABELS_KO.getOrElse(index) { "?" }

  fun name(index: Int): String = CLASS_NAMES.getOrElse(index) { "?" }
}
