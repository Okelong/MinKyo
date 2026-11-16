import hashlib
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ====================== 계산 부분 (예전 core.py) ======================
import cv2
import types
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


def preview(rgb, max_w=1280):
    """화면 표시용 축소본과 배율 (4K 그림을 그대로 그리면 메모리를 많이 씀)."""
    s = min(1.0, max_w / rgb.shape[1])
    if s >= 1.0:
        return rgb, 1.0
    return cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA), s


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
    ox, oy = 0, 0
    if roi is not None:                                  # 관심 영역만 잘라서 계산 (메모리·속도 절약)
        x0, y0, x1, y1 = roi
        frame_bgr = frame_bgr[y0:y1, x0:x1]
        ox, oy = x0, y0
    img = frame_bgr.astype(np.int16)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    score = r - (g + b) // 2
    bright = (r > 200) & (g > 200) & (b > 200)          # 과노출된 점 중심
    del img, b, g
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
    return float((xs * w).sum() / w.sum()) + ox, float((ys * w).sum() / w.sum()) + oy


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
def auto_window(t, xw, yw, win_s=10.0, min_s=3.0, margin_s=0.5, v_thr=None):
    """준비 동작(팔 뻗기, 자리 잡기, 점이 화면 밖으로 나감)을 빼고
    레이저 점이 한자리에 안정적으로 머무는 구간을 자동으로 찾는다.

    1) 점을 찾은 프레임만 쓰고, 일정한 시간 간격으로 맞춘다.
    2) 1 Hz 이하의 느린 움직임(자세 이동)만 남겨 그 속도(cm/s)를 구한다.
       떨림(2 Hz 이상)은 이 속도에 거의 들어가지 않는다.
    3) 느린 속도가 기준보다 작고 점이 보이는 시간이 가장 길게 이어진 구간을 고른다.
    4) 구간 앞뒤로 margin_s만큼 여유를 두고, 그 안에서 win_s초를 쓴다.
    """
    fs = round(1 / np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], 1 / fs)
    seen = ~np.isnan(xw)
    if seen.sum() < 10:
        raise ValueError("레이저 점을 거의 찾지 못했습니다. 분석 영역과 검출 감도를 확인하세요.")
    vis = np.interp(tu, t, seen.astype(float)) > 0.99            # 점이 보이는 시점
    x = np.interp(tu, t[seen], xw[seen])
    y = np.interp(tu, t[seen], yw[seen])
    b, a = butter(2, 1.0, btype="low", fs=fs)
    xs, ys = filtfilt(b, a, x), filtfilt(b, a, y)
    speed = np.hypot(np.gradient(xs), np.gradient(ys)) * fs         # 느린 움직임 속도 (cm/s)
    if v_thr is None:
        v_thr = max(1.0, 3 * np.median(speed[vis]))                 # 자세를 잡은 뒤의 흔들림보다 충분히 큰 값
    ok = vis & (speed < v_thr)
    # 가장 길게 이어진 안정 구간 찾기
    best, cur, start, best_start = 0, 0, 0, 0
    for i, v in enumerate(ok):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    t_s = tu[best_start] + margin_s
    t_e = tu[best_start + best - 1] - margin_s if best else t_s
    info = dict(speed_t=tu, speed=speed, v_thr=float(v_thr), stable=ok,
                stable_start=float(tu[best_start]) if best else np.nan,
                stable_end=float(tu[best_start + best - 1]) if best else np.nan)
    if t_e - t_s < min_s:
        raise ValueError(f"점이 안정적으로 머문 구간이 {max(t_e - t_s, 0):.1f}초뿐입니다. "
                         f"자세를 잡은 뒤 {min_s:.0f}초 이상 움직이지 않도록 다시 찍어 주세요.")
    t1 = min(t_s + win_s, t_e)
    return float(t_s), float(t1), info


def analyze(t, xw, yw, skip_s=3.0, win_s=10.0, fs_out=None,
            D_cm=None, hL_cm=None, hB_cm=None, target=None, window=None, h_dot_cm=None):
    """벽 좌표(cm) 시계열로 떨림을 분석한다.
    window=(시작, 끝)을 주면 그 구간을, 없으면 앞 skip_s초를 뺀 구간을 쓴다."""
    res = {}
    fs = fs_out or round(1 / np.median(np.diff(t)))
    if window is not None:
        t0, t1 = window
    else:
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
        if hL_cm is not None and h_dot_cm is not None:
            r = hL_cm - h_dot_cm                         # 레이저 높이 - 벽의 점 높이 (줄자로 잰 값)
            theta = float(np.arctan(abs(r) / D_cm))
        elif hL_cm is not None and hB_cm is not None:
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


# ---------------------------------------------------------------- 모눈 자동 인식
def paper_roi(rgb):
    """첫 화면에서 가장 큰 밝은 영역(종이)의 사각형 (x0, y0, x1, y1). 못 찾으면 None."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    sc = 800.0 / max(h, w)
    sm = cv2.resize(g, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (5, 5), 0)
    _, bw = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # 모눈 선 메우기
    n, lab, st_, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return None
    k = 1 + np.argmax(st_[1:, cv2.CC_STAT_AREA])
    if st_[k, cv2.CC_STAT_AREA] < 0.05 * sm.size:
        return None
    x, y, ww, hh = st_[k, :4]
    return (int(x / sc), int(y / sc), int((x + ww) / sc), int((y + hh) / sc))


def _pitch_1d(p):
    """밝기 단면에서 반복 간격(px)을 자기상관으로 찾는다."""
    p = np.asarray(p, float)
    n = len(p)
    if n < 40:
        return np.nan, 0.0
    win = max(5, n // 8) | 1
    p = p - np.convolve(p, np.ones(win) / win, mode="same")
    p = p[win:-win]
    p = p - p.mean()
    m = len(p)
    f = np.fft.rfft(p, 2 * m)
    ac = np.fft.irfft(f * np.conj(f))[:m]
    if ac[0] <= 0:
        return np.nan, 0.0
    ac = ac / ac[0]
    lo, hi = 4, m // 3
    pk, pr = find_peaks(ac[lo:hi], prominence=0.05)
    if len(pk) == 0:
        return np.nan, 0.0
    pk = pk + lo
    hts = ac[pk]
    first = pk[np.argmax(hts >= 0.3 * hts.max())]     # 충분히 뚜렷한 첫 번째 반복
    # 여러 배수 위치의 봉우리로 간격을 정밀하게
    lags, mult = [], []
    for j in range(1, 30):
        c = j * first
        if c + 3 >= hi:
            break
        a, b = int(c - 0.25 * first), int(c + 0.25 * first) + 1
        q = a + int(np.argmax(ac[a:b]))
        if 0 < q < m - 1:                                   # 포물선 보간
            d = ac[q - 1] - 2 * ac[q] + ac[q + 1]
            q = q + (0.5 * (ac[q - 1] - ac[q + 1]) / d if d != 0 else 0)
        lags.append(q)
        mult.append(j)
    pitch = float(np.polyfit(mult, lags, 1)[0]) if len(lags) >= 2 else float(first)
    return pitch, float(ac[first])


def grid_pitch(rgb, roi=None):
    """모눈 한 칸이 영상에서 몇 px인지 찾는다. 반환: dict(px, px_x, px_y, quality, ok)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    if roi is not None:
        x0, y0, x1, y1 = roi
        dx, dy = int(0.12 * (x1 - x0)), int(0.12 * (y1 - y0))
        g = g[y0 + dy:y1 - dy, x0 + dx:x1 - dx]
    d = 255.0 - g                                          # 선이 어두우므로 어두운 정도
    px_x, qx = _pitch_1d(d.mean(axis=0))                   # 세로 선 → 가로 간격
    px_y, qy = _pitch_1d(d.mean(axis=1))                   # 가로 선 → 세로 간격
    vals = [v for v in (px_x, px_y) if np.isfinite(v)]
    if not vals:
        return dict(ok=False, px=np.nan, px_x=px_x, px_y=px_y, quality=0.0)
    px = float(np.mean(vals))
    same = len(vals) == 2 and abs(px_x - px_y) / px < 0.15
    q = float(min(qx, qy)) if len(vals) == 2 else float(max(qx, qy))
    return dict(ok=bool(same and q > 0.15), px=px, px_x=px_x, px_y=px_y, quality=q)


core = types.SimpleNamespace(preview=preview, paper_roi=paper_roi, grid_pitch=grid_pitch, auto_window=auto_window, BAND=BAND, read_frames_info=read_frames_info, resolution_label=resolution_label, frame_timing=frame_timing, detect_dot=detect_dot, track_video=track_video, homography=homography, to_wall=to_wall, analyze=analyze, judge=judge)
# ====================== 화면 부분 ======================
import gc
# 화면에 보여야 할 것 (이용자 입력)
# 1. 측정 정보: 몇 차시인지, 사용한 손, 사용한 모눈 크기, 벽까지 거리
# 2. 측정 당시 상태
# 3. 카메라 설정 상태
# 4. 영상 2개 업로드 칸: 이완(안정시), 긴장(자세 유지)
#    + 이완 상태만 레이저 높이와 벽의 점 높이 (각도 보정용)
# 이후: 모눈 간격 자동 인식 → 분석 → 결과 → 두 조건 비교와 기록
# 진폭은 진단 기준으로 쓰지 않고, 파킨슨병·본태성 떨림 판정은 하지 않는다.

st.set_page_config(page_title="레이저 떨림 측정", layout="wide")
st.title("레이저 포인터 떨림 측정")
st.caption("손등에 붙인 레이저 점의 움직임을 촬영해 떨림의 주파수와 크기를 계산합니다. "
           "측정 보조용이며 진단 결과가 아닙니다.")

with st.expander("촬영 방법 (처음이라면 꼭 읽어 주세요)"):
    st.markdown(
        "1. 벽에 모눈종이를 평평하게 붙이고, 사람과 벽 사이를 **1~1.5 m**로 고정합니다.\n"
        "2. 스마트폰을 삼각대에 고정해 모눈종이를 **정면**으로 찍습니다.\n"
        "3. 카메라는 **FHD 60 fps**, 자동 FPS·손떨림 보정·HDR은 끄고, 일반 동영상으로 찍습니다.\n"
        "4. 레이저 포인터는 **손등 가운데**에 테이프로 붙이고 켜짐 상태로 고정합니다.\n"
        "5. **이완:** 팔꿈치부터 손목까지 받침대에 고정하고 손목은 힘을 빼 늘어뜨립니다.\n"
        "6. **긴장:** 팔을 앞으로 뻗어 모눈종이의 표적 점을 가리킵니다.\n"
        "7. 카메라는 모눈종이와 **정면으로 마주 보게** 두고, 모눈이 화면에 크게 나오게 찍습니다. 모눈 간격은 프로그램이 자동으로 찾습니다.\n"
        "8. 각 조건을 **13초 정도** 찍습니다. 앞 3초는 자세를 잡는 시간이라 분석에서 뺍니다.\n"
        "9. 레이저를 사람의 눈에 비추지 마세요.")

with st.sidebar.expander("고급 설정"):
    win_mode = st.radio("분석 구간", ["자동으로 찾기", "앞부분 직접 제외"],
                        help="자동: 점이 한자리에 안정적으로 머문 구간만 골라 씁니다.")
    skip_s = st.number_input("앞부분 제외 (초, 직접 제외일 때)", 0.0, 10.0, 3.0, 0.5)
    win_s = st.number_input("최대 분석 시간 (초)", 2.0, 60.0, 10.0, 1.0)
    min_score = st.slider("레이저 검출 감도 (낮을수록 예민)", 20, 150, 60)
    rel_thr = st.slider("떨림 판단 기준 (상대 세기)", 0.1, 0.8, 0.3, 0.05)

# ------------------------------------------------------------------ 1. 측정 정보
st.header("1. 측정 정보")
c = st.columns(4)
trial = c[0].number_input("몇 차시 측정인가요?", 1, 20, 1)
hand = c[1].radio("사용한 손", ["오른손", "왼손"], horizontal=True)
cell_cm = c[2].number_input("사용한 모눈 한 칸 크기 (cm)", 0.1, 10.0, 1.0, 0.1)
D_cm = c[3].number_input("레이저에서 벽까지 거리 (cm)", 10.0, 500.0, 100.0, 1.0)
if not 100 <= D_cm <= 150:
    st.warning("정한 거리(1~1.5 m)를 벗어났습니다.")

# ------------------------------------------------------------------ 2. 측정 당시 상태
st.header("2. 측정 당시 상태")
c = st.columns(4)
med = c[0].selectbox("떨림 관련 약 복용", ["해당 없음", "복용 전", "복용 후", "모름"])
med_h = None
if med == "복용 후":
    med_h = c[0].number_input("복용 후 몇 시간이 지났나요?", 0.0, 48.0, 1.0, 0.5)
caffeine = c[1].selectbox("측정 전 3시간 안에 카페인 섭취", ["아니요", "예", "모름"])
fatigue = c[2].selectbox("피로·수면 부족", ["괜찮음", "조금 피곤함", "많이 피곤함"])
stress = c[3].selectbox("긴장·불안", ["편안함", "조금 긴장됨", "많이 긴장됨"])
memo = st.text_input("그 밖의 메모 (예: 운동 직후, 추운 곳에서 측정)", "")
st.caption("약, 카페인, 피로, 긴장은 떨림에 영향을 줄 수 있어 결과와 함께 기록합니다.")

# ------------------------------------------------------------------ 3. 카메라 설정 상태
st.header("3. 카메라 설정 상태")
c = st.columns(2)
sel_res = c[0].selectbox("촬영한 해상도", ["FHD (1920×1080)", "4K (3840×2160)", "HD (1280×720)", "모름"])
sel_fps = c[1].selectbox("촬영한 프레임", ["60 fps", "30 fps", "120 fps", "모름"])
kk = st.columns(5)
cam_chk = {"삼각대 고정": kk[0].checkbox("삼각대에 고정함"),
           "손떨림 보정 끔": kk[1].checkbox("손떨림 보정 끔"),
           "자동 FPS 끔": kk[2].checkbox("자동 FPS 끔"),
           "HDR 끔": kk[3].checkbox("HDR 끔"),
           "일반 동영상 모드": kk[4].checkbox("슬로우 모션 아님")}
cam_missing = [a for a, v in cam_chk.items() if not v]
if sel_fps != "60 fps" or not sel_res.startswith(("FHD", "4K")):
    st.warning("권장 설정은 FHD 이상, 60 fps입니다.")
if cam_missing:
    st.info("체크하지 않은 설정: " + ", ".join(cam_missing) + ". 결과가 부정확할 수 있어 기록에 함께 남깁니다.")
st.caption("영상을 올리면 실제 해상도와 프레임을 파일에서 자동으로 읽어 선택한 값과 비교합니다.")

# ------------------------------------------------------------------ 4. 영상 올리기
st.header("4. 영상 올리기")
left, right = st.columns(2)
cases = {}
with left:
    st.subheader("이완 상태 (안정시)")
    up = st.file_uploader("이완 상태 영상", type=["mp4", "mov", "m4v", "avi"], key="rest_up")
    st.caption("이완 상태는 손목이 늘어져 빛이 비스듬히 내려가므로, 벽에 찍힌 점이 길게 늘어난 만큼 각도를 보정합니다.")
    hL = st.number_input("바닥 → 레이저(손등) 높이 (cm)", 0.0, 300.0, 90.0, 0.5, key="rest_hL")
    hd_unknown = st.checkbox("벽에 찍힌 레이저 점 높이를 모름", key="rest_hdno",
                             help="모르면 빛의 기울기 보정을 하지 않습니다. 주파수에는 영향이 없습니다.")
    h_dot = None if hd_unknown else st.number_input(
        "바닥 → 벽에 찍힌 레이저 점 높이 (cm)", 0.0, 300.0, 70.0, 0.5, key="rest_hd",
        help="자세를 잡은 뒤 점이 머무는 위치를 줄자로 바닥부터 잽니다.")
    cases["rest_"] = dict(cond="이완", up=up, hL=hL, h_dot=h_dot, posture=False)
with right:
    st.subheader("긴장 상태 (자세 유지)")
    up = st.file_uploader("긴장 상태 영상", type=["mp4", "mov", "m4v", "avi"], key="post_up")
    st.caption("긴장 상태는 팔을 뻗어 표적을 정면으로 가리키므로 빛이 거의 수평이라 각도 보정을 하지 않습니다.")
    cases["post_"] = dict(cond="긴장", up=up, hL=None, h_dot=None, posture=True)


@st.cache_data(show_spinner="레이저 점을 찾는 중입니다. 영상 길이에 따라 수십 초 걸릴 수 있습니다.")
def track_cached(p, roi_, ms_):
    return core.track_video(p, roi_, ms_)


@st.cache_data(show_spinner=False)
def timing_cached(p):
    return core.frame_timing(p)


def save_upload(up):
    data = up.getbuffer()                                # 복사 없이 읽기 (메모리 절약)
    key = hashlib.md5(data).hexdigest()
    path = os.path.join(tempfile.gettempdir(), f"laser_{key}{os.path.splitext(up.name)[1] or '.mp4'}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path, key


def camera_auto(path, info):
    """파일에서 읽은 실제 해상도·프레임을 보여 주고, 사용자가 고른 값과 비교한다."""
    tm = timing_cached(path)
    dur = tm["duration"] if tm["ok"] else (info["n"] / info["fps"] if info["fps"] else 0)
    fps_real = tm["fps_real"] if tm["ok"] else info["fps"]
    lab = core.resolution_label(info["w"], info["h"])
    c = st.columns(4)
    c[0].metric("실제 해상도", lab, f"{info['w']}×{info['h']}", delta_color="off")
    c[1].metric("파일에 적힌 프레임", f"{info['fps']:.1f} fps")
    c[2].metric("실제 측정 프레임", f"{fps_real:.1f} fps",
                f"간격 흔들림 {tm['jitter'] * 100:.1f}%" if tm["ok"] else "측정 불가", delta_color="off")
    c[3].metric("영상 길이", f"{dur:.1f} 초")
    warns = []
    if sel_res != "모름" and not sel_res.startswith(lab.split()[0]):
        warns.append(f"선택한 해상도({sel_res})와 실제 해상도({lab})가 다릅니다.")
    if sel_fps != "모름" and abs(fps_real - float(sel_fps.split()[0])) > 3:
        warns.append(f"선택한 프레임({sel_fps})과 실제 프레임({fps_real:.0f} fps)이 다릅니다.")
    if min(info["w"], info["h"]) < 1080:
        warns.append("FHD보다 낮은 해상도입니다. 작은 떨림을 놓칠 수 있습니다.")
    if fps_real < 55:
        warns.append("60 fps보다 낮습니다. 12 Hz 근처의 빠른 떨림은 정확하지 않을 수 있습니다.")
    if abs(fps_real - info["fps"]) > 0.1 * max(info["fps"], 1):
        warns.append("파일에 적힌 프레임과 실제 프레임이 다릅니다. 슬로우 모션이나 자동 FPS를 확인하세요.")
    if tm["ok"] and (tm["jitter"] > 0.1 or tm["long_gaps"] > 0.02 * tm["n"]):
        warns.append(f"프레임 간격이 일정하지 않습니다(빠진 프레임 {tm['long_gaps']}곳).")
    for w_ in warns:
        st.warning(w_)
    return dict(dur=dur, fps_real=fps_real, lab=lab)


def run_case(k, cs):
    cond, up = cs["cond"], cs["up"]
    posture = cs["posture"]
    if up is None:
        st.info(f"4번에서 {cond} 상태 영상을 올리면 여기서 분석할 수 있습니다.")
        return None
    path, vkey = save_upload(up)
    info = core.read_frames_info(path)
    st.markdown("**카메라 설정 (파일에서 자동으로 읽은 값)**")
    cam = camera_auto(path, info)
    if cam["dur"] < (4 if win_mode == "자동으로 찾기" else skip_s + 2):
        st.error("영상이 너무 짧습니다. 앞부분 제외 시간보다 2초 이상 길게 찍어 주세요.")
        return None

    # 모눈 간격 자동 인식 (정면 촬영 가정)
    W, Hh = info["w"], info["h"]
    proi = core.paper_roi(info["first"])
    gp = core.grid_pitch(info["first"], proi)
    with st.expander("자동 인식이 틀렸을 때 직접 고치기"):
        px_manual = st.number_input("영상에서 모눈 한 칸의 픽셀 수 (0 = 자동)", 0.0, 1000.0, 0.0, 0.5, key=k + "pxm")
        use_full = st.checkbox("종이 영역을 쓰지 않고 화면 전체에서 레이저 찾기", key=k + "full")
    px = px_manual if px_manual > 0 else gp["px"]
    if not np.isfinite(px) or px <= 0:
        st.error("모눈 간격을 찾지 못했습니다. 모눈종이가 화면에 크게, 정면으로 나오게 다시 찍거나 "
                 "위의 '직접 고치기'에서 한 칸의 픽셀 수를 넣어 주세요.")
        return None
    kcm = cell_cm / px                                   # 1 px = kcm cm
    roi = None if (use_full or proi is None) else proi
    st.markdown(f"**모눈 자동 인식:** 한 칸 = **{px:.1f} px** "
                f"(가로 {gp['px_x']:.1f}, 세로 {gp['px_y']:.1f}) → 1 px = {kcm * 10:.3f} mm")
    if roi is not None:
        st.caption(f"인식한 종이 크기 약 {(roi[2] - roi[0]) * kcm:.1f} × {(roi[3] - roi[1]) * kcm:.1f} cm "
                   "(A4는 21 × 29.7 cm). 크게 다르면 모눈 간격을 잘못 잡은 것입니다.")
    if px_manual <= 0 and not gp["ok"]:
        st.warning("모눈 간격 인식이 불확실합니다(가로·세로 간격이 다르거나 선이 흐림). "
                   "카메라가 정면이 아니거나 초점이 맞지 않았을 수 있습니다. 아래 그림의 노란 눈금이 모눈 선과 맞는지 확인하세요.")

    first_bgr = info["first"][..., ::-1].copy()
    dot0 = core.detect_dot(first_bgr, roi, min_score)
    pv, ps = core.preview(info["first"])                 # 화면 표시용 축소본 (ps = 축소 배율)
    fig, ax = plt.subplots(figsize=(9, 9 * Hh / W), dpi=80)
    ax.imshow(pv)
    if roi is not None:
        ax.add_patch(plt.Rectangle((roi[0] * ps, roi[1] * ps), (roi[2] - roi[0]) * ps, (roi[3] - roi[1]) * ps,
                                   fill=False, ls="--", color="#eb6834", lw=1.5))
    cx = ((roi[0] + roi[2]) / 2 if roi else W / 2) * ps
    cy = ((roi[1] + roi[3]) / 2 if roi else Hh / 2) * ps
    pp = px * ps
    for i in range(-5, 6):                               # 인식한 간격으로 그린 확인용 눈금
        ax.plot([cx + i * pp] * 2, [cy - 5 * pp, cy + 5 * pp], color="yellow", lw=0.8)
        ax.plot([cx - 5 * pp, cx + 5 * pp], [cy + i * pp] * 2, color="yellow", lw=0.8)
    if dot0:
        ax.plot(dot0[0] * ps, dot0[1] * ps, "+", color="red", ms=18, mew=2)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("주황 점선: 인식한 종이(레이저를 찾는 영역) · 노란 눈금: 인식한 모눈 간격(10칸) · 빨간 +: 첫 화면의 레이저 점. "
               "노란 눈금 간격이 모눈 한 칸과 같으면 정상입니다(위치는 어긋나도 됩니다).")
    if dot0 is None:
        st.warning("첫 화면에서 레이저 점을 찾지 못했습니다. 준비 중이라 점이 없었다면 괜찮습니다. "
                   "분석이 실패하면 검출 감도를 낮추거나 '화면 전체에서 찾기'를 켜 보세요.")

    run_key = (vkey, roi, min_score, round(float(px), 3))
    if st.button(f"{cond} 상태 분석 시작", type="primary", key=k + "go"):
        st.session_state[k + "ran"] = run_key
    if st.session_state.get(k + "ran") != run_key:
        return None
    t, xp, yp, fps = track_cached(path, roi, min_score)
    gc.collect()
    xw, yw = xp * kcm, (Hh - yp) * kcm                  # 화면 픽셀 → 벽 cm (위쪽이 +y)
    try:
        aw = None
        if win_mode == "자동으로 찾기":
            t_s, t_e, aw = core.auto_window(t, xw, yw, win_s)
            window = (t_s, t_e)
        else:
            window = None
        r = core.analyze(t, xw, yw, skip_s, win_s, D_cm=D_cm, hL_cm=cs["hL"], h_dot_cm=cs["h_dot"],
                         window=window)
    except ValueError as e:
        st.error(str(e))
        return None
    has = r["rel_power"] >= rel_thr

    st.markdown(f"### {cond} 상태 결과")
    used0, used1 = r["t"][0] - t[0], r["t"][-1] - t[0]
    if aw is not None:
        st.info(f"준비 동작을 빼고 점이 안정적으로 머문 **{used0:.1f}초 ~ {used1:.1f}초** "
                f"({used1 - used0:.1f}초)를 자동으로 골라 분석했습니다.")
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.plot(aw["speed_t"] - t[0], aw["speed"], color="#2a78d6", lw=1)
        ax.axhline(aw["v_thr"], color="#eb6834", ls="--", lw=1)
        ax.axvspan(used0, used1, color="#e5f4e4")
        ax.set_ylim(0, max(3 * aw["v_thr"], 1))
        ax.set_xlabel("time in video (s)")
        ax.set_ylabel("slow speed (cm/s)")
        ax.set_title("Green: segment used for analysis  |  dashed: stability threshold", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(f"영상의 {used0:.1f}초 ~ {used1:.1f}초를 분석했습니다.")
    if has:
        st.warning(f"떨림이 감지되었습니다. 초당 약 {r['peak_hz']:.1f}회({r['peak_hz']:.1f} Hz) 떨립니다.")
    else:
        st.success("뚜렷한 떨림이 감지되지 않았습니다. 표시된 주파수는 참고값입니다.")
    m = st.columns(4)
    m[0].metric("떨림", "감지됨" if has else "감지 안 됨", f"상대 세기 {r['rel_power']:.2f}", delta_color="off")
    m[1].metric("주파수", f"{r['peak_hz']:.2f} Hz",
                f"피크 간격 {r['interval_hz']:.2f} Hz" if np.isfinite(r["interval_hz"]) else None, delta_color="off")
    m[2].metric("벽에서 진폭", f"{r['amp_mm']:.2f} mm")
    no_corr = cs["h_dot"] is None
    m[3].metric("겉보기 각진폭", f"{r['ang_amp_deg']:.3f}°",
                ("긴장 상태는 보정 안 함" if posture else "각도 보정 안 됨") if no_corr else f"빛의 기울기 θ {r['theta_deg']:.1f}°", delta_color="off")
    if posture:
        dev = float(np.mean(np.hypot(r["x_raw"] - r["mean_pos"][0], r["y_raw"] - r["mean_pos"][1])))
        st.metric("점이 평균 위치에서 벗어난 거리 (흔들림 + 느린 흔들림)", f"{dev:.2f} cm")

    warns = []
    if r["missing_ratio"] > 0.1:
        warns.append(f"분석 구간의 {r['missing_ratio'] * 100:.0f}%에서 레이저 점을 찾지 못했습니다.")
    if no_corr and not posture:
        warns.append("벽에 찍힌 점 높이를 몰라 빛의 기울기 보정을 하지 않았습니다. "
                     "손이 많이 처졌다면 각진폭이 실제보다 크게 나올 수 있습니다. 주파수에는 영향이 없습니다.")
    if r["theta_deg"] > 25:
        warns.append("빛의 기울기가 25°보다 큽니다. 진폭 보정 오차가 커질 수 있습니다.")
    for w_ in warns:
        st.warning(w_)

    g = st.columns(3)
    with g[0]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(r["x_raw"], r["y_raw"], lw=0.8, color="#2a78d6")
        ax.set_aspect("equal")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title("Dot path on wall", fontsize=10)
        st.pyplot(fig)
        plt.close(fig)
    with g[1]:
        fig, ax = plt.subplots(figsize=(4, 4))
        tt = r["t"] - r["t"][0]
        ax.plot(tt, r["s"] * 10, lw=0.9, color="#2a78d6")
        ax.set_xlim(0, min(3, tt[-1]))
        ax.set_xlabel("time (s, first 3 s)")
        ax.set_ylabel("filtered position (mm)")
        ax.set_title("Tremor signal (2-14 Hz)", fontsize=10)
        st.pyplot(fig)
        plt.close(fig)
    with g[2]:
        fig, ax = plt.subplots(figsize=(4, 4))
        band = (r["freqs"] >= 1) & (r["freqs"] <= 15)
        ax.plot(r["freqs"][band], r["power"][band] / r["power"].max(), color="#2a78d6")
        ax.axvline(r["peak_hz"], color="#eb6834", ls="--", lw=1)
        ax.axvspan(1, core.BAND[0], color="0.9")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("relative power")
        ax.set_title(f"Spectrum (peak {r['peak_hz']:.2f} Hz)", fontsize=10)
        st.pyplot(fig)
        plt.close(fig)

    coords = pd.DataFrame({"t_s": t, "x_px": xp, "y_px": yp, "x_cm": xw, "y_cm": yw})
    st.download_button(f"{cond} 상태 좌표 CSV 저장", coords.to_csv(index=False).encode("utf-8-sig"),
                       f"coords_{k}{hand}_{int(trial)}차.csv", "text/csv", key=k + "dl")

    return {"차시": int(trial), "조건": cond, "손": hand, "떨림": "있음" if has else "없음",
            "주파수(Hz)": round(r["peak_hz"], 2), "피크간격(Hz)": round(r["interval_hz"], 2),
            "상대세기": round(r["rel_power"], 2), "진폭(mm)": round(r["amp_mm"], 2),
            "각도 보정": "안 함" if no_corr else "함", "θ(°)": np.nan if no_corr else round(r["theta_deg"], 1), "각진폭(°)": round(r["ang_amp_deg"], 3),
            "평균위치 이탈(cm)": round(dev, 2) if posture else np.nan,
            "분석 구간(s)": f"{r['t'][0] - t[0]:.1f}~{r['t'][-1] - t[0]:.1f}",
            "레이저 높이(cm)": np.nan if cs["hL"] is None else cs["hL"],
            "점 높이(cm)": np.nan if no_corr else cs["h_dot"], "거리(cm)": D_cm,
            "모눈 크기(cm)": cell_cm, "모눈 한 칸(px)": round(float(px), 1),
            "약 복용": med + (f" ({med_h}시간 후)" if med_h is not None else ""),
            "카페인": caffeine, "피로": fatigue, "긴장·불안": stress, "메모": memo,
            "선택 해상도": sel_res, "선택 프레임": sel_fps,
            "실제 해상도": f"{cam['lab']} {info['w']}×{info['h']}", "실제 fps": round(cam["fps_real"], 1),
            "미확인 설정": ", ".join(cam_missing) if cam_missing else "없음"}


# ------------------------------------------------------------------ 5. 분석
st.header("5. 분석")
tab1, tab2 = st.tabs(["이완 상태 (안정시)", "긴장 상태 (자세 유지)"])
with tab1:
    row_rest = run_case("rest_", cases["rest_"])
with tab2:
    row_post = run_case("post_", cases["post_"])

# ------------------------------------------------------------------ 6. 비교와 기록
st.header("6. 이완·긴장 비교와 기록")
rows = [x for x in (row_rest, row_post) if x]
if len(rows) < 2:
    st.info("두 영상을 모두 분석하면 여기서 비교할 수 있습니다.")
else:
    cmp_cols = ["떨림", "주파수(Hz)", "상대세기", "진폭(mm)", "각도 보정", "θ(°)", "각진폭(°)", "평균위치 이탈(cm)"]
    st.dataframe(pd.DataFrame(rows).set_index("조건")[cmp_cols].T.astype(str), use_container_width=True)
    rest_has = row_rest["떨림"] == "있음"
    post_has = row_post["떨림"] == "있음"
    if rest_has:
        st.warning("힘을 빼고 쉬고 있을 때 떨림이 감지되었습니다. 신경과 진료를 받아 보시길 권합니다.")
    if post_has and not rest_has:
        st.info("팔을 뻗고 있을 때만 떨림이 감지되었습니다. 떨림이 계속되거나 생활에 불편하면 진료를 받아 보세요.")
    if not rest_has and not post_has:
        st.success("두 조건 모두 뚜렷한 떨림이 감지되지 않았습니다.")
    if rest_has and post_has:
        st.write(f"두 조건의 주파수 차이: {abs(row_rest['주파수(Hz)'] - row_post['주파수(Hz)']):.2f} Hz")
    st.caption("겉보기 각진폭은 같은 사람의 두 조건을 비교하는 용도입니다. "
               "주파수나 진폭만으로 파킨슨병과 본태성 떨림을 구별할 수 없습니다.")

if "history" not in st.session_state:
    st.session_state["history"] = []
if rows and st.button(f"{int(trial)}차시 결과를 기록에 추가"):
    st.session_state["history"].extend(rows)
hist = pd.DataFrame(st.session_state["history"])
if len(hist):
    st.dataframe(hist, use_container_width=True)
    summ = hist.groupby(["조건", "손"])[["주파수(Hz)", "진폭(mm)", "각진폭(°)"]].agg(["mean", "std", "count"])
    st.write("조건별 평균과 표준편차")
    st.dataframe(summ.round(3), use_container_width=True)
    st.download_button("기록 CSV 저장", hist.to_csv(index=False).encode("utf-8-sig"),
                       "tremor_results.csv", "text/csv")
    if st.button("기록 지우기"):
        st.session_state["history"] = []
        st.rerun()
else:
    st.caption("차시를 바꿔 가며 3회 정도 측정해 기록하면 평균과 편차를 볼 수 있습니다.")

st.divider()
st.caption("이 결과는 떨림을 수치로 보여 주는 측정 보조 자료이며, 파킨슨병이나 본태성 떨림을 진단하지 않습니다. "
           "진단은 다른 소견과 함께 의사가 합니다.")
