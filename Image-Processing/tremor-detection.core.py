"""레이저 점 떨림 분석 핵심 함수 (Streamlit 화면 없이도 동작)."""
import numpy as np
import cv2
from scipy.signal import butter, filtfilt, find_peaks

BAND = (2.0, 14.0)  # 분석할 떨림 주파수 대역 (Hz)


# ---------------------------------------------------------------- 영상 읽기
def read_frames_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise ValueError("영상을 읽을 수 없습니다.")
    return dict(fps=fps, n=n, w=w, h=h, first=cv2.cvtColor(first, cv2.COLOR_BGR2RGB))


def resolution_label(w, h):
    s = min(w, h)
    if s >= 2160:
        return "4K (UHD)"
    if s >= 1440:
        return "QHD"
    if s >= 1080:
        return "FHD"
    if s >= 720:
        return "HD"
    return "SD (낮음)"


def frame_timing(path):
    """프레임을 디코딩하지 않고 시간만 읽어 실제 프레임 간격을 잰다."""
    cap = cv2.VideoCapture(path)
    t = []
    while cap.grab():
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    t = np.asarray(t)
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        return dict(ok=False, n=len(t))
    dt = np.diff(t)
    med = float(np.median(dt))
    return dict(ok=True, n=len(t),
                fps_real=1.0 / med,                       # 실제 프레임 수 (중앙값 기준)
                jitter=float(np.std(dt) / med),           # 간격의 흔들림 (0이면 일정)
                long_gaps=int(np.sum(dt > 1.5 * med)),    # 프레임이 빠진 곳의 수
                duration=float(t[-1] - t[0]))


def detect_dot(frame_bgr, roi=None, min_score=60):
    """빨간 레이저 점의 중심(px)을 찾는다. 못 찾으면 None.

    score = R - (G+B)/2 : 빨간 정도. 점 가운데가 하얗게 포화돼도
    주변의 붉은 테두리로 중심을 잡도록, 밝기 가중 중심을 쓴다.
    """
    img = frame_bgr.astype(np.int16)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    score = r - (g + b) // 2
    bright = (r > 200) & (g > 200) & (b > 200)          # 과노출된 점 중심
    if roi is not None:
        x0, y0, x1, y1 = roi
        mask = np.zeros(score.shape, bool)
        mask[y0:y1, x0:x1] = True
        score = np.where(mask, score, -255)
        bright &= mask
    smax = score.max()
    if smax < min_score:
        return None
    sel = score > max(min_score, 0.5 * smax)
    # 붉은 테두리 바로 안쪽의 하얀 중심도 포함
    sel = cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (sel | bright)
    n_lab, lab, stats, cent = cv2.connectedComponentsWithStats(sel.astype(np.uint8), 8)
    if n_lab <= 1:
        return None
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])     # 가장 큰 덩어리 = 레이저 점
    ys, xs = np.nonzero(lab == k)
    w = np.clip(score[ys, xs], 1, None).astype(float)
    return float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())


def track_video(path, roi=None, min_score=60, progress=None):
    """프레임마다 (시간 s, x px, y px). 점을 못 찾은 프레임은 NaN."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    t, xs, ys = [], [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t.append(ms / 1000.0 if ms > 0 or i == 0 else i / fps)
        p = detect_dot(fr, roi, min_score)
        xs.append(np.nan if p is None else p[0])
        ys.append(np.nan if p is None else p[1])
        i += 1
        if progress and n and i % 10 == 0:
            progress(min(i / n, 1.0))
    cap.release()
    t = np.asarray(t, float)
    if len(t) > 2 and (np.any(np.diff(t) <= 0)):      # 시간 정보가 이상하면 fps로 대체
        t = np.arange(len(t)) / fps
    return t, np.asarray(xs), np.asarray(ys), fps


# ---------------------------------------------------------------- 보정
def homography(corners_px, width_cm, height_cm):
    """모눈 사각형 네 모서리(왼위, 오위, 오아래, 왼아래 순서, px) → 벽 좌표(cm).
    벽 좌표는 왼아래 모서리가 (0, 0), 오른쪽이 +x, 위쪽이 +y."""
    src = np.float32(corners_px)
    dst = np.float32([[0, height_cm], [width_cm, height_cm], [width_cm, 0], [0, 0]])
    return cv2.getPerspectiveTransform(src, dst)


def to_wall(H, x, y):
    pts = np.stack([x, y], 1).astype(np.float32)
    ok = ~np.isnan(pts).any(1)
    out = np.full(pts.shape, np.nan)
    if ok.any():
        out[ok] = cv2.perspectiveTransform(pts[ok][None], H)[0]
    return out[:, 0], out[:, 1]


# ---------------------------------------------------------------- 분석
def analyze(t, xw, yw, skip_s=3.0, win_s=10.0, fs_out=None,
            D_cm=None, hL_cm=None, hB_cm=None, target=None):
    """벽 좌표(cm) 시계열로 떨림을 분석한다."""
    res = {}
    fs = fs_out or round(1 / np.median(np.diff(t)))
    t0 = t[0] + skip_s
    t1 = min(t0 + win_s, t[-1])
    if t1 - t0 < 2:
        raise ValueError("분석할 구간이 2초보다 짧습니다. 영상 길이와 제외 시간을 확인하세요.")
    tu = np.arange(t0, t1, 1 / fs)                       # 일정한 시간 간격으로 다시 맞춤
    sel = (t >= t0 - 0.5) & (t <= t1 + 0.5)
    good = sel & ~np.isnan(xw)
    res["missing_ratio"] = float(1 - good.sum() / max(sel.sum(), 1))
    if good.sum() < 10:
        raise ValueError("레이저 점을 거의 찾지 못했습니다. 분석 영역과 검출 감도를 확인하세요.")
    x = np.interp(tu, t[good], xw[good])
    y = np.interp(tu, t[good], yw[good])
    res.update(t=tu, x_raw=x, y_raw=y, fs=fs)

    # 표적 이탈 (필터 전, 느린 움직임 포함)
    mx, my = x.mean(), y.mean()
    res["mean_pos"] = (mx, my)
    if target is not None:
        res["target_dev_cm"] = float(np.mean(np.hypot(x - target[0], y - target[1])))

    # 2–14 Hz 대역 통과
    hi = min(BAND[1], 0.45 * fs)
    b, a = butter(4, [BAND[0], hi], btype="bandpass", fs=fs)
    xf, yf = filtfilt(b, a, x - mx), filtfilt(b, a, y - my)
    # 가장 많이 움직인 방향으로 합치기
    v = np.linalg.svd(np.stack([xf, yf], 1), full_matrices=False)[2][0]
    s = xf * v[0] + yf * v[1]
    res.update(xf=xf, yf=yf, s=s, direction_deg=float(np.degrees(np.arctan2(v[1], v[0])) % 180))

    # FFT
    w = np.hanning(len(s))
    F = np.fft.rfft(s * w)
    fr = np.fft.rfftfreq(len(s), 1 / fs)
    P = np.abs(F) ** 2
    band = (fr >= BAND[0]) & (fr <= hi)
    k = np.where(band)[0][np.argmax(P[band])]
    peak = float(fr[k])
    near = band & (np.abs(fr - peak) <= 0.5)
    res.update(freqs=fr, power=P, peak_hz=peak,
               rel_power=float(P[near].sum() / P[band].sum()))

    # 피크 간격으로 교차 확인
    pk, _ = find_peaks(s, distance=max(1, int(fs / hi)), prominence=0.5 * np.std(s))
    res["interval_hz"] = float(1 / np.median(np.diff(tu[pk]))) if len(pk) > 2 else float("nan")

    # 진폭 (벽에서, cm). 사인파라면 RMS×√2 = 진폭
    amp_cm = float(np.sqrt(2) * np.std(s))
    res["amp_mm"] = amp_cm * 10

    # 빛의 기울기 보정
    if D_cm:
        theta = 0.0
        if hL_cm is not None and hB_cm is not None:
            r = hL_cm - (hB_cm + my)                     # 레이저 높이 - 점 높이
            theta = float(np.arctan(abs(r) / D_cm))
        res["theta_deg"] = float(np.degrees(theta))
        res["ang_amp_deg"] = float(np.degrees(amp_cm * np.cos(theta) ** 2 / D_cm))
    return res


def judge(res, cond, rel_thr=0.3):
    """측정값을 쉬운 말로 정리한다. 진단이 아니다."""
    has = res["rel_power"] >= rel_thr
    msgs = []
    if has:
        msgs.append(f"떨림이 감지되었습니다. 초당 약 {res['peak_hz']:.1f}회(={res['peak_hz']:.1f} Hz) 떨립니다.")
        if cond == "안정시":
            msgs.append("힘을 빼고 쉬고 있을 때 떨림이 감지되었습니다. 신경과 진료를 받아 보시길 권합니다.")
        else:
            msgs.append("팔을 뻗고 있을 때 떨림이 감지되었습니다. 떨림이 계속되거나 생활에 불편하면 진료를 받아 보세요.")
    else:
        msgs.append("뚜렷한 떨림이 감지되지 않았습니다. 표시된 주파수는 참고값입니다.")
    return has, msgs
