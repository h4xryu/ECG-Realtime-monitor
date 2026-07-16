package com.example.hicardimonitor

/**
 * 표시 전용 실시간 스무더 — sweep 파형을 공식 앱 수준으로 매끄럽게.
 *
 * 파이프라인(모두 causal/스트리밍): trailing median baseline(0.6s) 로 DC·QRS-후 오버슈트를
 * 강건하게 제거(중앙값은 QRS 스파이크 무시) → EMA LP(α=0.4) → 이동평균(3) 로 고주파 fuzz 제거.
 *
 * OnlineFilter 출력(표시 mV) 위에 얹는다. R-peak 트래커·기록·모델 입력은 건드리지 않는다
 * (그쪽은 원본 filtered/raw 를 계속 사용). 실측(live rec 528b06ec): 2차차분 ripple 19→6.8,
 * QRS 진폭 94% 보존, baseline 0 정렬.
 */
class EcgDisplaySmoother(fs: Float = ModelConfig.FS_HZ) {

  private val medWin = (fs * 0.6f).toInt().coerceAtLeast(3)
  private val order = FloatArray(medWin)   // 삽입 순서 ring (가장 오래된 값 제거용)
  private var orderHead = 0
  private var filled = 0
  private val sorted = FloatArray(medWin)  // 현재 창 값들을 정렬 보관
  private var sortedLen = 0

  private var lpY = 0f
  private val lpAlpha = 0.4f

  private val maN = 3
  private val maBuf = FloatArray(maN)
  private var maHead = 0
  private var maCount = 0
  private var maSum = 0f

  fun reset() {
    orderHead = 0
    filled = 0
    sortedLen = 0
    lpY = 0f
    maHead = 0
    maCount = 0
    maSum = 0f
  }

  /** 한 샘플 처리 → 매끄러운 표시값. */
  fun process(x: Float): Float {
    val base = pushMedian(x)
    val y = x - base
    lpY = lpAlpha * y + (1f - lpAlpha) * lpY
    return pushMovingAvg(lpY)
  }

  /** trailing median: 창이 차면 가장 오래된 값을 정렬배열에서 제거하고 새 값을 삽입. O(win). */
  private fun pushMedian(x: Float): Float {
    if (filled == medWin) {
      val old = order[orderHead]
      val idx = lowerBound(old)
      // idx 위치의 old 제거(좌측 시프트)
      var i = idx
      while (i < sortedLen - 1) { sorted[i] = sorted[i + 1]; i++ }
      sortedLen--
    } else {
      filled++
    }
    order[orderHead] = x
    orderHead = (orderHead + 1) % medWin

    val pos = lowerBound(x)
    var j = sortedLen
    while (j > pos) { sorted[j] = sorted[j - 1]; j-- }
    sorted[pos] = x
    sortedLen++

    return if (sortedLen and 1 == 1) {
      sorted[sortedLen / 2]
    } else {
      0.5f * (sorted[sortedLen / 2 - 1] + sorted[sortedLen / 2])
    }
  }

  /** sorted[0..sortedLen) 에서 v 이상이 처음 나오는 위치(이진 탐색). */
  private fun lowerBound(v: Float): Int {
    var lo = 0
    var hi = sortedLen
    while (lo < hi) {
      val mid = (lo + hi) ushr 1
      if (sorted[mid] < v) lo = mid + 1 else hi = mid
    }
    return lo
  }

  private fun pushMovingAvg(v: Float): Float {
    if (maCount == maN) {
      maSum -= maBuf[maHead]
    } else {
      maCount++
    }
    maBuf[maHead] = v
    maSum += v
    maHead = (maHead + 1) % maN
    return maSum / maCount
  }
}
