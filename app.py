import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os
import tempfile

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成＆高度編集ツール", layout="wide")

st.title("🎬 LINEアニメーションスタンプ自動生成 ＆ 高度編集ツール")
st.caption("動画解析後、中央プレビューを見ながら両サイドの操作バーで「切り出し」「マスク」「透過」を自在に調整できます。")

if 'raw_stamps' not in st.session_state:
    st.session_state['raw_stamps'] = None
if 'fps' not in st.session_state:
    st.session_state['fps'] = 10
if 'first_frame' not in st.session_state:
    st.session_state['first_frame'] = None
if 'raw_video_frames' not in st.session_state:
    st.session_state['raw_video_frames'] = None

ROWS = 3
COLS = 4

def draw_grid_preview(frame_bgr, rows=3, cols=4, offset_x=0, offset_y=0, cell_expand=0, selected_idx=0, custom_offsets=None):
    """動画の第1フレーム上に3x4グリッド枠（赤・黄）を描画して視覚的に確認する関数"""
    h, w, _ = frame_bgr.shape
    preview = frame_bgr.copy()
    
    cell_h = h // rows
    cell_w = w // cols
    
    if custom_offsets is None:
        custom_offsets = {}
        
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            
            x1 = c * cell_w + offset_x
            y1 = r * cell_h + offset_y
            x2 = (c + 1) * cell_w + offset_x
            y2 = (r + 1) * cell_h + offset_y
            
            c_ox, c_oy, c_exp = custom_offsets.get(idx, (0, 0, 0))
            x1 += c_ox - (cell_expand + c_exp)
            y1 += c_oy - (cell_expand + c_exp)
            x2 += c_ox + (cell_expand + c_exp)
            y2 += c_oy + (cell_expand + c_exp)
            
            x1_c = max(0, min(w - 1, int(x1)))
            y1_c = max(0, min(h - 1, int(y1)))
            x2_c = max(0, min(w - 1, int(x2)))
            y2_c = max(0, min(h - 1, int(y2)))
            
            if selected_idx is not None and idx == selected_idx:
                color = (0, 255, 255) # 黄色（選択中のスタンプ）
                thickness = 3
            else:
                color = (0, 0, 255) # 赤色
                thickness = 2
                
            cv2.rectangle(preview, (x1_c, y1_c), (x2_c, y2_c), color, thickness)
            cv2.putText(preview, f"#{idx+1}", (x1_c + 8, y1_c + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(preview, f"#{idx+1}", (x1_c + 8, y1_c + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

def extract_stamp_cell(frame_bgr, r, c, rows=3, cols=4, offset_x=0, offset_y=0, cell_expand=0, custom_offsets=None):
    """調整されたグリッド座標に従ってセル領域をクロップ抽出する関数"""
    h, w, _ = frame_bgr.shape
    cell_h = h // rows
    cell_w = w // cols
    idx = r * cols + c
    
    x1 = c * cell_w + offset_x
    y1 = r * cell_h + offset_y
    x2 = (c + 1) * cell_w + offset_x
    y2 = (r + 1) * cell_h + offset_y
    
    if custom_offsets is None:
        custom_offsets = {}
        
    c_ox, c_oy, c_exp = custom_offsets.get(idx, (0, 0, 0))
    x1 += c_ox - (cell_expand + c_exp)
    y1 += c_oy - (cell_expand + c_exp)
    x2 += c_ox + (cell_expand + c_exp)
    y2 += c_oy + (cell_expand + c_exp)
    
    x1_c = max(0, min(w - 1, int(x1)))
    y1_c = max(0, min(h - 1, int(y1)))
    x2_c = max(x1_c + 1, min(w, int(x2)))
    y2_c = max(y1_c + 1, min(h, int(y2)))
    
    return frame_bgr[y1_c:y2_c, x1_c:x2_c]

def crop_cell_margins(cell_bgr, crop_left_pct=0, crop_right_pct=0, crop_top_pct=0, crop_bottom_pct=0):
    """画面端の不要な見切れ領域をカットする関数"""
    h, w, _ = cell_bgr.shape
    top = int(h * crop_top_pct / 100.0)
    bottom = h - int(h * crop_bottom_pct / 100.0)
    left = int(w * crop_left_pct / 100.0)
    right = w - int(w * crop_right_pct / 100.0)
    
    if bottom <= top + 10:
        bottom = top + 10
    if right <= left + 10:
        right = left + 10
        
    return cell_bgr[top:bottom, left:right]

def remove_isolated_noise_alpha(alpha_channel, min_size_pct=0.015):
    """メインキャラ以外の小さな見切れゴミ（Zzzマーク等）を自動消去"""
    h, w = alpha_channel.shape
    total_area = h * w
    min_area = total_area * min_size_pct
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha_channel > 127).astype(np.uint8))
    if num_labels <= 2:
        return alpha_channel
        
    new_alpha = np.zeros_like(alpha_channel)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    largest_idx = areas[0][0]
    new_alpha[labels == largest_idx] = 255
    
    for idx, area in areas[1:]:
        if area >= min_area:
            new_alpha[labels == idx] = 255
            
    return new_alpha

def remove_background_floodfill_outer(cell_bgr, tolerance=70, filter_noise=True):
    """外枠からの塗りつぶしで背景や足元の影を自動透過"""
    h, w, _ = cell_bgr.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    img_work = cell_bgr.copy()
    bg_color = cell_bgr[0, 0].astype(np.float32)
    
    seeds = []
    step = 5
    for x in range(0, w, step):
        seeds.append((x, 0))
        seeds.append((x, h - 1))
    for y in range(0, h, step):
        seeds.append((0, y))
        seeds.append((w - 1, y))
        
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    
    for seed_x, seed_y in seeds:
        if mask[seed_y + 1, seed_x + 1] == 0:
            pixel_color = cell_bgr[seed_y, seed_x].astype(np.float32)
            color_dist = np.linalg.norm(pixel_color - bg_color)
            if color_dist <= tolerance * 2.0:
                cv2.floodFill(img_work, mask, seedPoint=(seed_x, seed_y), newVal=(0, 0, 0),
                              loDiff=lo_diff, upDiff=up_diff, flags=flags)
            
    bg_mask = mask[1:h+1, 1:w+1]
    alpha = np.where(bg_mask == 255, 0, 255).astype(np.uint8)
    
    if filter_noise:
        alpha = remove_isolated_noise_alpha(alpha)
        
    alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0)
    _, alpha_smoothed = cv2.threshold(alpha_blurred, 127, 255, cv2.THRESH_BINARY)
    
    b, g, r = cv2.split(cell_bgr)
    cell_bgra = cv2.merge([b, g, r, alpha_smoothed])
    return Image.fromarray(cv2.cvtColor(cell_bgra, cv2.COLOR_BGRA2RGBA))

def center_and_fit_stamp(frame_list, target_w=320, target_h=270, padding=12):
    """キャラを認識して中央寄せ配置"""
    if not frame_list:
        return frame_list
        
    alphas = [np.array(img)[:, :, 3] for img in frame_list]
    stacked_alpha = np.maximum.reduce(alphas)
    non_zeros = np.argwhere(stacked_alpha > 10)
    
    if non_zeros.size == 0:
        return [img.resize((target_w, target_h), Image.Resampling.LANCZOS) for img in frame_list]
        
    min_y, min_x = non_zeros.min(axis=0)
    max_y, max_x = non_zeros.max(axis=0)
    
    h_orig, w_orig = stacked_alpha.shape
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(w_orig - 1, max_x + padding)
    max_y = min(h_orig - 1, max_y + padding)
    
    crop_w = max_x - min_x + 1
    crop_h = max_y - min_y + 1
    
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    
    centered_frames = []
    for img in frame_list:
        cropped = img.crop((min_x, min_y, max_x + 1, max_y + 1))
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        canvas.paste(resized, (offset_x, offset_y), mask=resized)
        centered_frames.append(canvas)
        
    return centered_frames

def process_frame_sequence_strict(frames, start_frame, end_frame, target_frame_count=15, ping_pong=False, trim_end=False):
    """コマの選択・トリミング・往復再生・重複削除を行った上で、LINE規格の【5〜20コマ以内】に厳密に収める関数"""
    sub = frames[start_frame - 1 : end_frame]
    if not sub:
        sub = frames
        
    if trim_end and len(sub) > 2:
        sub = sub[:-1]
        
    if ping_pong and len(sub) > 2:
        reverse_part = sub[-2:0:-1]
        sub = sub + reverse_part
        
    target_count = max(5, min(20, target_frame_count))
    if len(sub) != target_count:
        indices = np.linspace(0, len(sub) - 1, target_count, dtype=int)
        sub = [sub[i] for i in indices]
        
    return sub

def create_preview_gif(frame_list, duration_ms):
    """プレビュー用Web GIF作成"""
    if not frame_list:
        return b""
    buf = io.BytesIO()
    frame_list[0].save(
        buf, format="GIF", save_all=True,
        append_images=frame_list[1:], duration=duration_ms, loop=0, disposal=2
    )
    return buf.getvalue()

def optimize_apng_bytes(img_list, durations, loop_count):
    """LINE 1MB以下適合APNG出力"""
    buf = io.BytesIO()
    img_list[0].save(
        buf, format="PNG", save_all=True,
        append_images=img_list[1:], duration=durations, loop=loop_count
    )
    data = buf.getvalue()
    if len(data) < 990000:
        return data
        
    for colors in [128, 64, 32]:
        quantized_imgs = [img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE).convert("RGBA") for img in img_list]
        buf = io.BytesIO()
        quantized_imgs[0].save(
            buf, format="PNG", save_all=True,
            append_images=quantized_imgs[1:], duration=durations, loop=loop_count
        )
        data = buf.getvalue()
        if len(data) < 990000:
            return data
    return data

# --- Step 1: 動画のアップロード ＆ フレームロード ---
uploaded_file = st.file_uploader("1. 動画ファイル (MP4 / MOV) をアップロードしてください", type=["mp4", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        uploaded_file.seek(0)
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
        
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    st.session_state['fps'] = fps
    
    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    if raw_frames:
        st.session_state['raw_video_frames'] = raw_frames
        st.session_state['first_frame'] = raw_frames[0]
    else:
        st.error("動画フレームの読み込みに失敗しました。正しい動画ファイルかご確認ください。")

# --- Step 2: 視覚的グリッド調整エリア (切り出し枠のリアルタイム調整) ---
if st.session_state.get('raw_video_frames') is not None:
    st.divider()
    st.header("📐 12カット自動切り出し枠（赤枠グリッド）の視覚調整")
    st.caption("動画全体の12コマ切り出し範囲を画面上で視覚的に調整できます。キャラクターが見切れないよう赤枠の幅や位置をあわせてください。")
    
    col_grid_img, col_grid_ctrl = st.columns([1.5, 1])
    
    with col_grid_ctrl:
        st.subheader("⚙️ グリッド位置・枠サイズの調整")
        
        st.markdown("**🌐 全体の枠ずらし ＆ 拡大**")
        grid_offset_x = st.slider("全体を左右に移動 (px)", min_value=-100, max_value=100, value=0, step=1)
        grid_offset_y = st.slider("全体を上下に移動 (px)", min_value=-100, max_value=100, value=0, step=1)
        grid_expand = st.slider("切り出し枠を外側に広げる (px)", min_value=0, max_value=100, value=10, step=2, help="枠を広げることで、動きが大きいキャラの見切れをまとめて防止できます")
        
        st.markdown("---")
        st.markdown("**🎯 特定スタンプ個別の位置調整 (微調整)**")
        custom_target_idx = st.number_input("位置を微調整するスタンプ番号 (1〜12)", min_value=1, max_value=12, value=3) - 1
        
        ind_ox = st.slider(f"スタンプ #{custom_target_idx+1} の左右移動", min_value=-50, max_value=50, value=0, step=1)
        ind_oy = st.slider(f"スタンプ #{custom_target_idx+1} の上下移動", min_value=-50, max_value=50, value=0, step=1)
        ind_exp = st.slider(f"スタンプ #{custom_target_idx+1} の枠個別拡大", min_value=-20, max_value=50, value=0, step=1)
        
        custom_offsets = {custom_target_idx: (ind_ox, ind_oy, ind_exp)}
        
        if st.button("🚀 この切り出し枠でスタンプ編集画面へ進む", type="primary"):
            raw_video_frames = st.session_state['raw_video_frames']
            raw_stamps = {i: [] for i in range(12)}
            
            for frame in raw_video_frames:
                for r in range(ROWS):
                    for c in range(COLS):
                        idx = r * COLS + c
                        cell = extract_stamp_cell(
                            frame, r, c, ROWS, COLS,
                            offset_x=grid_offset_x, offset_y=grid_offset_y,
                            cell_expand=grid_expand, custom_offsets=custom_offsets
                        )
                        raw_stamps[idx].append(cell)
                        
            st.session_state['raw_stamps'] = raw_stamps
            st.success("切り出し枠を反映しました！下部のプレビューエリアで編集を行ってください。")

    with col_grid_img:
        st.subheader("🖼️ 切り出し枠の確認（黄色枠＝調整中のスタンプ）")
        first_frame = st.session_state['first_frame']
        grid_preview_img = draw_grid_preview(
            first_frame, ROWS, COLS,
            offset_x=grid_offset_x, offset_y=grid_offset_y,
            cell_expand=grid_expand, selected_idx=custom_target_idx,
            custom_offsets=custom_offsets
        )
        st.image(grid_preview_img, caption="赤枠・黄枠が自動切り出しの範囲です", use_container_width=True)

# --- Step 3: 3カラム編集ワークスペース（プレビュー ＆ 高度編集） ---
if st.session_state.get('raw_stamps') is not None:
    st.divider()
    st.header("🎛️ 3画面 リアルタイムスタンプ編集ワークスペース")
    
    raw_stamps_data = st.session_state['raw_stamps']
    max_raw_frames = max(1, len(raw_stamps_data.get(0, [])))
    
    col_left, col_center, col_right = st.columns([1, 1.4, 1])
    
    # ------------------- 左カラム: アニメーション＆タイムライン -------------------
    with col_left:
        st.subheader("🎞️ アニメ設定")
        selected_stamp_idx = st.number_input("確認・編集するスタンプ番号 (1〜12)", min_value=1, max_value=12, value=custom_target_idx + 1) - 1
        
        st.markdown("---")
        if max_raw_frames > 1:
            frame_range = st.slider(
                "元動画の使用範囲",
                min_value=1, max_value=max_raw_frames,
                value=(1, max_raw_frames),
                help="切り出すコマの開始〜終了位置を指定します"
            )
        else:
            frame_range = (1, 1)

        default_target_count = min(20, max(5, max_raw_frames))
        target_frame_count = st.slider(
            "出力コマ数 (5〜20コマ)",
            min_value=5, max_value=20, value=default_target_count, step=1
        )
        
        ping_pong = st.checkbox("🔄 往復再生（ピンポン）", value=False)
        trim_end = st.checkbox("✂️ ループ末尾カット", value=True)
        
        target_sec = st.selectbox("総再生時間 (秒)", [1, 2, 3, 4], index=1)
        loop_count = st.selectbox("ループ回数", [1, 2, 3, 4], index=1)

    # ------------------- 右カラム: マスク・トリミング ＆ 透過度 -------------------
    with col_right:
        st.subheader("✂️ マスク ＆ 透過")
        filter_noise = st.checkbox("🧹 見切れゴミ（Zzz等）自動除去", value=True)
        
        st.markdown("**端のカット (見切れ削除)**")
        crop_right_pct = st.slider("右端削り (%)", min_value=0, max_value=50, value=0, step=1)
        crop_left_pct = st.slider("左端削り (%)", min_value=0, max_value=50, value=0, step=1)
        crop_top_pct = st.slider("上端削り (%)", min_value=0, max_value=50, value=0, step=1)
        crop_bottom_pct = st.slider("下端削り (%)", min_value=0, max_value=50, value=0, step=1)
        
        st.markdown("---")
        st.markdown("**🎨 透過感度**")
        tolerance = st.slider(
            "透過の強さ (しきい値)",
            min_value=10, max_value=150, value=75, step=5,
            help="足元の影や薄いゴミを消す強さです"
        )

    # ------------------- データ処理計算 -------------------
    current_raw_cells = raw_stamps_data.get(selected_stamp_idx, raw_stamps_data.get(0, []))
    cropped_cells = [crop_cell_margins(c, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct) for c in current_raw_cells]
    current_transparent_frames = [remove_background_floodfill_outer(c, tolerance=tolerance, filter_noise=filter_noise) for c in cropped_cells]
    current_centered_frames = center_and_fit_stamp(current_transparent_frames)
    
    edited_frames = process_frame_sequence_strict(
        current_centered_frames, frame_range[0], frame_range[1],
        target_frame_count=target_frame_count, ping_pong=ping_pong, trim_end=trim_end
    )
    
    total_ms = target_sec * 1000
    loop_ms = total_ms // loop_count
    frame_cnt = len(edited_frames)
    frame_duration_ms = max(50, loop_ms // frame_cnt)
    
    base_ms = loop_ms // frame_cnt
    remainder = loop_ms % frame_cnt
    durations_list = [base_ms] * frame_cnt
    for i in range(remainder):
        durations_list[i] += 1

    # ------------------- 中央カラム: 大画面アニメーションプレビュー -------------------
    with col_center:
        st.subheader("👁️ プレビュー確認 (大画面)")
        preview_gif = create_preview_gif(edited_frames, frame_duration_ms)
        st.image(preview_gif, caption=f"スタンプ #{selected_stamp_idx + 1} | {frame_cnt}コマ / {target_sec}秒 ({loop_count}回再生)", use_container_width=True)
        
        single_apng_data = optimize_apng_bytes(edited_frames, durations_list, loop_count)
        st.download_button(
            label=f"💾 スタンプ #{selected_stamp_idx + 1} を個別ダウンロード (APNG)",
            data=single_apng_data,
            file_name=f"stamp_{selected_stamp_idx+1:02d}.png",
            mime="image/png"
        )
        st.success(f"✅ LINE適合: **{frame_cnt}コマ** / 1コマ **{frame_duration_ms}ms** / 計 **{target_sec}秒**")
        
    st.divider()
    
    # --- Step 4: 全スタンプ一括出力 ---
    st.subheader("📦 編集した設定で全12個のスタンプを一括書き出し")
    
    if st.button("🚀 LINE審査適合APNGを一括ダウンロード (ZIP)"):
        with st.spinner("12個のスタンプを設定に従って一括書き出し中..."):
            try:
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for idx in range(12):
                        raw_cells = raw_stamps_data.get(idx, [])
                        c_cells = [crop_cell_margins(c, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct) for c in raw_cells]
                        trans_frames = [remove_background_floodfill_outer(c, tolerance=tolerance, filter_noise=filter_noise) for c in c_cells]
                        cent_frames = center_and_fit_stamp(trans_frames)
                        proc_f = process_frame_sequence_strict(
                            cent_frames, frame_range[0], frame_range[1],
                            target_frame_count=target_frame_count, ping_pong=ping_pong, trim_end=trim_end
                        )
                        
                        apng_data = optimize_apng_bytes(proc_f, durations_list, loop_count)
                        zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                        
                st.success("🎉 全12個のアニメーションスタンプの出力が完了しました！")
                
                st.download_button(
                    label="📦 LINE審査適合スタンプ一括ダウンロード (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="line_animation_stamps.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"書き出し中にエラーが発生しました: {e}")
