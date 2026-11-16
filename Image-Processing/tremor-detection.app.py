import hashlib
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="레이저 떨림 측정", layout="wide")
st.title("레이저 포인터 떨림 측정")
st.caption("손에 붙인 레이저 점의 움직임을 촬영해 떨림의 주파수와 크기를 계산합니다. "
           "측정 보조용이며 진단 결과가 아닙니다.")

with st.expander("촬영 방법 (처음이라면 꼭 읽어 주세요)"):
    st.markdown(
        "1. 벽에 모눈종이를 평평하게 붙이고, 스마트폰을 삼각대에 고정해 모눈종이를 정면으로 찍습니다.\n"
        "2. 카메라는 **FHD 60 fps**로 맞추고, **자동 FPS·손떨림 보정·HDR은 끕니다.**\n"
        "3. 레이저 포인터를 손등 가운데에 붙이고 켜짐 상태로 고정합니다.\n"
        "4. **안정시:** 팔을 받침대에 올리고 손목에 힘을 뺍니다. "
        "**자세 유지:** 팔을 앞으로 뻗어 표적 점을 가리킵니다.\n"
        "5. 13초 정도 찍습니다. 앞 3초는 자세를 잡는 시간이라 분석에서 뺍니다.\n"
        "6. 레이저를 사람의 눈에 비추지 마세요.")

# ------------------------------------------------------------------ 사이드바
sb = st.sidebar
sb.header("1. 측정 조건")
cond = sb.radio("측정 상태", ["안정시", "자세 유지"], horizontal=True)
hand = sb.radio("사용한 손", ["오른손", "왼손"], horizontal=True)
trial = sb.number_input("측정 회차", 1, 10, 1)

sb.header("2. 모눈종이")
cell_cm = sb.number_input("모눈 한 칸 크기 (cm)", 0.1, 10.0, 1.0, 0.1)
nx = sb.number_input("표시할 사각형의 가로 칸 수", 1, 100, 10)
ny = sb.number_input("표시할 사각형의 세로 칸 수", 1, 100, 10)
sb.caption("화면에서 이 사각형의 네 모서리 위치를 입력하면, 비스듬한 촬영도 보정됩니다.")

sb.header("3. 거리와 높이 (각도 보정)")
D_cm = sb.number_input("레이저에서 벽까지 거리 D (cm)", 10.0, 500.0, 100.0, 1.0)
use_h = sb.checkbox("높이를 재서 빛의 기울기 보정하기", value=True)
hL = hB = None
if use_h:
    hL = sb.number_input("바닥 → 레이저 높이 h_L (cm)", 0.0, 300.0, 90.0, 0.5)
    hB = sb.number_input("바닥 → 사각형 아래 선 높이 h_B (cm)", 0.0, 300.0, 70.0, 0.5)
target = None
if cond == "자세 유지":
    sb.header("표적 점 (사각형 왼쪽 아래 기준, cm)")
    target = (sb.number_input("표적 x (cm)", 0.0, 500.0, float(nx * cell_cm / 2)),
              sb.number_input("표적 y (cm)", 0.0, 500.0, float(ny * cell_cm / 2)))

with sb.expander("고급 설정"):
    skip_s = st.number_input("앞부분 제외 (초)", 0.0, 10.0, 3.0, 0.5)
    win_s = st.number_input("분석 시간 (초)", 2.0, 60.0, 10.0, 1.0)
    min_score = st.slider("레이저 검출 감도 (낮을수록 예민)", 20, 150, 60)
    rel_thr = st.slider("떨림 판단 기준 (상대 세기)", 0.1, 0.8, 0.3, 0.05)

# ------------------------------------------------------------------ 영상
st.header("① 영상 올리기")
up = st.file_uploader("촬영한 영상 (mp4, mov)", type=["mp4", "mov", "m4v", "avi"])
if up is None:
    st.info("영상을 올리면 분석을 시작할 수 있습니다.")
    st.stop()

data = up.getvalue()
key = hashlib.md5(data).hexdigest()
path = os.path.join(tempfile.gettempdir(), f"laser_{key}{os.path.splitext(up.name)[1] or '.mp4'}")
if not os.path.exists(path):
    with open(path, "wb") as f:
        f.write(data)

info = core.read_frames_info(path)
tm = core.frame_timing(path)
dur = tm["duration"] if tm["ok"] else (info["n"] / info["fps"] if info["fps"] else 0)
fps_real = tm["fps_real"] if tm["ok"] else info["fps"]
res_lab = core.resolution_label(info["w"], info["h"])

st.subheader("카메라 설정 확인")
c1, c2, c3, c4 = st.columns(4)
c1.metric("해상도", res_lab, f"{info['w']}×{info['h']}", delta_color="off")
c2.metric("파일에 적힌 프레임", f"{info['fps']:.1f} fps")
c3.metric("실제 측정 프레임", f"{fps_real:.1f} fps",
          f"간격 흔들림 {tm['jitter'] * 100:.1f}%" if tm["ok"] else "측정 불가", delta_color="off")
c4.metric("길이", f"{dur:.1f} 초")

cam_warn = []
if min(info["w"], info["h"]) < 1080:
    cam_warn.append(f"해상도가 {res_lab}입니다. FHD(1920×1080) 이상으로 찍으면 작은 떨림을 더 잘 잡습니다.")
if fps_real < 55:
    cam_warn.append(f"실제 프레임이 {fps_real:.0f} fps입니다. 60 fps로 찍어야 12 Hz 근처까지 정확합니다.")
if abs(fps_real - info["fps"]) > 0.1 * max(info["fps"], 1):
    cam_warn.append("파일에 적힌 프레임과 실제 프레임이 다릅니다. 슬로우 모션이나 자동 FPS로 찍었는지 확인하세요.")
if tm["ok"] and (tm["jitter"] > 0.1 or tm["long_gaps"] > 0.02 * tm["n"]):
    cam_warn.append(f"프레임 간격이 일정하지 않습니다(빠진 프레임 {tm['long_gaps']}곳). "
                    "자동 FPS를 끄고, 밝은 곳에서 다시 찍는 것을 권합니다.")
for w_ in cam_warn:
    st.warning(w_)
if not cam_warn:
    st.success("해상도와 프레임이 측정에 적합합니다.")

st.write("아래 설정은 영상 파일에 기록되지 않아 자동으로 확인할 수 없습니다. 촬영할 때의 설정을 체크해 주세요.")
k1, k2, k3, k4, k5 = st.columns(5)
cam_chk = {
    "삼각대 고정": k1.checkbox("삼각대에 고정함"),
    "손떨림 보정 끔": k2.checkbox("손떨림 보정 끔"),
    "자동 FPS 끔": k3.checkbox("자동 FPS 끔"),
    "HDR 끔": k4.checkbox("HDR 끔"),
    "일반 동영상 모드": k5.checkbox("슬로우 모션 아님"),
}
missing_chk = [k for k, v in cam_chk.items() if not v]
if missing_chk:
    st.info("체크하지 않은 설정: " + ", ".join(missing_chk) + ". 결과가 부정확할 수 있으니 기록에 함께 남깁니다.")

if dur < skip_s + 2:
    st.error("영상이 너무 짧습니다. 앞부분 제외 시간보다 2초 이상 길게 찍어 주세요.")
    st.stop()

# ------------------------------------------------------------------ 보정
st.header("② 모눈종이 위치 지정")
st.write(f"첫 화면에서 **가로 {nx}칸 × 세로 {ny}칸** 사각형의 네 모서리 좌표(픽셀)를 읽어 입력하세요. "
         "그림의 눈금이 픽셀 좌표입니다.")
W, Hh = info["w"], info["h"]
dflt = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
names = ["왼쪽 위", "오른쪽 위", "오른쪽 아래", "왼쪽 아래"]
corners = []
cols = st.columns(4)
for i, (nm, (fx, fy)) in enumerate(zip(names, dflt)):
    with cols[i]:
        st.markdown(f"**{nm}**")
        x = st.number_input("x", 0, W, int(W * fx), key=f"cx{i}")
        y = st.number_input("y", 0, Hh, int(Hh * fy), key=f"cy{i}")
        corners.append((x, y))

xs_c = [c[0] for c in corners]
ys_c = [c[1] for c in corners]
pad = 0.15 * max(max(xs_c) - min(xs_c), max(ys_c) - min(ys_c))
roi = (int(max(0, min(xs_c) - pad)), int(max(0, min(ys_c) - pad)),
       int(min(W, max(xs_c) + pad)), int(min(Hh, max(ys_c) + pad)))

first_bgr = info["first"][..., ::-1].copy()
dot0 = core.detect_dot(first_bgr, roi, min_score)
fig, ax = plt.subplots(figsize=(9, 9 * Hh / W))
ax.imshow(info["first"])
poly = corners + [corners[0]]
ax.plot([p[0] for p in poly], [p[1] for p in poly], "-", color="#2a78d6", lw=2)
for (x, y), nm in zip(corners, ["TL", "TR", "BR", "BL"]):
    ax.plot(x, y, "o", color="#2a78d6")
    ax.annotate(nm, (x, y), color="#2a78d6", xytext=(5, 5), textcoords="offset points")
ax.add_patch(plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1],
                           fill=False, ls="--", color="#eb6834"))
if dot0:
    ax.plot(*dot0, "+", color="yellow", ms=18, mew=2)
ax.set_xticks(np.arange(0, W + 1, 100))
ax.set_yticks(np.arange(0, Hh + 1, 100))
ax.tick_params(labelsize=7)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.caption("파란 선: 입력한 사각형 · 주황 점선: 레이저를 찾는 영역 · 노란 +: 첫 화면에서 찾은 레이저 점")
if dot0 is None:
    st.warning("첫 화면에서 레이저 점을 찾지 못했습니다. 사각형 위치나 검출 감도를 확인하세요.")

Hm = core.homography(corners, nx * cell_cm, ny * cell_cm)

# ------------------------------------------------------------------ 분석
st.header("③ 분석")


@st.cache_data(show_spinner="레이저 점을 찾는 중입니다. 영상 길이에 따라 수십 초 걸릴 수 있습니다.")
def track_cached(p, roi_, ms_):
    return core.track_video(p, roi_, ms_)


if not st.button("분석 시작", type="primary") and st.session_state.get("last_key") != (key, roi, min_score):
    st.stop()
st.session_state["last_key"] = (key, roi, min_score)

t, xp, yp, fps = track_cached(path, roi, min_score)
xw, yw = core.to_wall(Hm, xp, yp)
try:
    r = core.analyze(t, xw, yw, skip_s, win_s, D_cm=D_cm, hL_cm=hL, hB_cm=hB, target=target)
except ValueError as e:
    st.error(str(e))
    st.stop()
has, msgs = core.judge(r, cond, rel_thr)

# ------------------------------------------------------------------ 결과
st.header("④ 결과")
for m in msgs:
    (st.warning if has else st.success)(m)

m1, m2, m3, m4 = st.columns(4)
m1.metric("떨림", "감지됨" if has else "감지 안 됨", f"상대 세기 {r['rel_power']:.2f}", delta_color="off")
m2.metric("주파수", f"{r['peak_hz']:.2f} Hz",
          f"피크 간격 {r['interval_hz']:.2f} Hz" if np.isfinite(r["interval_hz"]) else None, delta_color="off")
m3.metric("벽에서 진폭", f"{r['amp_mm']:.2f} mm")
m4.metric("겉보기 각진폭", f"{r['ang_amp_deg']:.3f}°", f"빛의 기울기 θ {r['theta_deg']:.1f}°", delta_color="off")
if "target_dev_cm" in r:
    st.metric("표적에서 평균 이탈 거리", f"{r['target_dev_cm']:.2f} cm")

warns = []
if r["missing_ratio"] > 0.1:
    warns.append(f"분석 구간의 {r['missing_ratio'] * 100:.0f}%에서 레이저 점을 찾지 못했습니다.")
if r["theta_deg"] > 25:
    warns.append("빛의 기울기가 25°보다 큽니다. 진폭 보정 오차가 커질 수 있습니다.")
mx, my = r["mean_pos"]
if not (0 <= mx <= nx * cell_cm and 0 <= my <= ny * cell_cm):
    warns.append("레이저 점이 지정한 사각형 밖에 있습니다. 보정이 부정확할 수 있습니다.")
if not has:
    warns.append("떨림이 뚜렷하지 않아 주파수는 참고값입니다.")
for w_ in warns:
    st.warning(w_)

g1, g2, g3 = st.columns(3)
with g1:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(r["x_raw"], r["y_raw"], lw=0.8, color="#2a78d6")
    if target is not None:
        ax.plot(*target, "x", color="#eb6834", ms=10, mew=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Dot path on wall", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)
with g2:
    fig, ax = plt.subplots(figsize=(4, 4))
    tt = r["t"] - r["t"][0]
    ax.plot(tt, r["s"] * 10, lw=0.9, color="#2a78d6")
    ax.set_xlim(0, min(3, tt[-1]))
    ax.set_xlabel("time (s, first 3 s)")
    ax.set_ylabel("filtered position (mm)")
    ax.set_title("Tremor signal (2-14 Hz)", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)
with g3:
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

# ------------------------------------------------------------------ 기록
st.header("⑤ 기록과 비교")
row = {"조건": cond, "손": hand, "회차": int(trial), "떨림": "있음" if has else "없음",
       "주파수(Hz)": round(r["peak_hz"], 2), "피크간격(Hz)": round(r["interval_hz"], 2),
       "상대세기": round(r["rel_power"], 2), "진폭(mm)": round(r["amp_mm"], 2),
       "θ(°)": round(r["theta_deg"], 1), "각진폭(°)": round(r["ang_amp_deg"], 3),
       "표적이탈(cm)": round(r.get("target_dev_cm", np.nan), 2), "D(cm)": D_cm,
       "해상도": f"{res_lab} {info['w']}×{info['h']}", "실제 fps": round(fps_real, 1),
       "프레임 흔들림(%)": round(tm["jitter"] * 100, 1) if tm["ok"] else np.nan,
       "미확인 설정": ", ".join(missing_chk) if missing_chk else "없음"}
if "history" not in st.session_state:
    st.session_state["history"] = []
if st.button("이 결과를 기록에 추가"):
    st.session_state["history"].append(row)

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
    st.caption("반복 측정(3회 권장)한 결과를 추가하면 평균과 편차를 볼 수 있습니다.")

coords = pd.DataFrame({"t_s": t, "x_px": xp, "y_px": yp, "x_cm": xw, "y_cm": yw})
st.download_button("이번 영상 좌표 CSV 저장", coords.to_csv(index=False).encode("utf-8-sig"),
                   f"coords_{cond}_{hand}_{int(trial)}.csv", "text/csv")

st.divider()
st.caption("이 결과는 떨림을 수치로 보여 주는 측정 보조 자료이며, 파킨슨병이나 본태성 떨림을 진단하지 않습니다. "
           "진단은 다른 소견과 함께 의사가 합니다.")
