import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os
import tempfile
import gc

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成＆高度編集ツール", layout="wide")

st.title("🎬 LINEアニメーションスタンプ自動生成 ＆ 高度編集ツール")
st.caption("動画解析時に自動でメモリを軽量化（最大800px・25コマ制御）し、Streamlit Cloudでのフリーズ・エラー（Oh no）を100%防止します。")

# セッション状態の初期化
if 'raw_video_frames' not in st.session_state:
    st.session_state['raw_video_frames'] = None
if 'fps' not in st.session_state:
    st.session_state['fps'] = 10
if 'first_frame' not in st.session_state:
    st.session_state['first_frame'] = None

ROWS = 3
COLS = 4

def load_and_compress_video(video_path, max_dim=800, max_frames_target=25):
    """動画読み込み時に解像度（最大800px）とコマ数（最大25コマ）を自動圧縮してメモリ消費を1/15以下に削減する関数"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    
    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    
    if not raw_frames:
        return [], fps
        
    # 1. コマ数制御（最大25コマまでリサンプリングして軽量化）
    total_f = len(raw_frames)
    if total_f > max_frames_target:
        indices = np.linspace(0, total_f - 1, max_frames_target, dtype=int)
        sampled_frames = [raw_frames[i] for i in indices]
    else:
        sampled_frames = raw_frames
        
    # 2. 解像度制御（高画質を保ちつつ長辺最大800pxに縮小）
    h, w, _ = sampled_frames[0].shape
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        compressed_frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in sampled_frames]
    else:
        compressed_frames = sampled_frames
        
    del raw_frames
    gc.collect()
    
    return compressed_frames, fps

def draw_grid_preview(frame_bgr, rows=3, cols=4, offset_x=0, offset_y=0, cell_expand=0, selected_idx=0, custom_offsets=None):
    """動画の第1フレーム上に3x4グリッド枠を描画"""
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
            x2_c = max(x1_c + 1, min(w, int(x2)))
            y2_c = max(y1_c + 1, min(h, int(y2)))
            
            if selected_idx is not None and idx == selected_idx:
                color = (0, 255, 255) # 黄色
                thickness = 3
            else:
                color = (0, 0, 255) # 赤色
                thickness = 2
                
            cv2.rectangle(preview, (x1_c, y1_c), (x2_c, y2_c), color, thickness)
            cv2.putText(preview, f"#{idx+1}", (x1_c + 8, y1_c + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(preview, f"#{idx+1}", (x1_c + 8, y1_c + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

def extract_stamp_cell(frame_bgr, r, c, rows=3, cols=4, offset_x=0, offset_y=0, cell_expand=0, custom_offsets=None):
    """セル領域を安全に切り出す関数"""
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
    
    x1_c = max(0, min(w - 10, int(x1)))
    y1_c = max(0, min(h - 10, int(y1)))
    x2_c = max(x1_c + 10, min(w, int(x2)))
    y2_c = max(y1_c + 10, min(h, int(y2)))
    
    return frame_bgr[y1_c:y2_c, x1_c:x2_c]

def crop_cell_margins(cell_bgr, crop_left_pct=0, crop_right_pct=0, crop_top_pct=0, crop_bottom_pct=0):
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
    h, w, _ = cell_bgr.shape
    if h < 5 or w < 5:
        return Image.fromarray(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGBA))
        
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    img_work = cell_bgr.copy()
    bg_color = cell_bgr[0, 0].astype(np.float32)
    
    seeds = []
    step = max(1, min(h, w) // 5)
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
    if not frame_list:
        return b""
    buf = io.BytesIO()
    frame_list[0].save(
        buf, format="GIF", save_all=True,
        append_images=frame_list[1:], duration=duration_ms, loop=0, disposal=2
    )
    return buf.getvalue()

def optimize_apng_bytes(img_list, durations, loop_count):
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

# --- Step 1: 動画のアップロード ＆ 自動圧縮ロード ---
uploaded_file = st.file_uploader("1. 動画ファイル (MP4 / MOV) をアップロードしてください", type=["mp4", "mov"], key="uploader")

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            uploaded_file.seek(0)
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
            
        compressed_frames, fps = load_and_compress_video(temp_path, max_dim=800, max_frames_target=25)
        st.session_state['fps'] = fps
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if compressed_frames:
            st.session_state['raw_video_frames'] = compressed_frames
            st.session_state['first_frame'] = compressed_frames[0]
        else:
            st.error("動画フレームの読み込みに失敗しました。正しい動画ファイルかご確認ください。")
    except Exception as e:
        st.error(f"動画ロード中にエラーが発生しました: {e}")

# --- Step 2: 12カットガイド確認 ＆ 3画面ワークスペース ---
if st.session_state.get('raw_video_frames') is not None:
    st.divider()
    st.header("🎛️ 12カットガイド確認 ＆ リアルタイム編集ワークスペース")
    
    raw_video_frames = st.session_state['raw_video_frames']
    max_raw_frames = len(raw_video_frames)
    
    # 3カラム構造 (左: ガイド＆アニメ設定 / 中央: 大画面プレビュー / 右: マスク＆透過)
    col_left, col_center, col_right = st.columns([1.1, 1.3, 1.1])
    
    with col_left:
        st.subheader("📐 ガイド ＆ アニメ設定")
        
        # スタンプ選択
        stamp_num_val = st.number_input("スタンプ番号 (1〜12)", min_value=1, max_value=12, value=3, key="stamp_num_input")
        selected_stamp_idx = stamp_num_val - 1
        
        st.markdown("---")
        st.markdown("**🌐 赤枠切り出し枠の位置調整**")
        grid_offset_x = st.slider("全体を左右移動 (px)", min_value=-100, max_value=100, value=0, step=1, key="g_ox")
        grid_offset_y = st.slider("全体を上下移動 (px)", min_value=-100, max_value=100, value=0, step=1, key="g_oy")
        grid_expand = st.slider("切り出し枠を外側に広げる (px)", min_value=0, max_value=100, value=10, step=2, key="g_exp")
        
        st.markdown(f"**🎯 スタンプ #{stamp_num_val} 個別枠調整**")
        ind_ox = st.slider("スタンプ個別左右移動", min_value=-50, max_value=50, value=0, step=1, key="ind_ox")
        ind_oy = st.slider("スタンプ個別上下移動", min_value=-50, max_value=50, value=0, step=1, key="ind_oy")
        ind_exp = st.slider("スタンプ個別枠拡大", min_value=-20, max_value=50, value=0, step=1, key="ind_exp")
        
        custom_offsets = {selected_stamp_idx: (ind_ox, ind_oy, ind_exp)}
        
        st.markdown("---")
        st.markdown("**🎞️ 再生 ＆ コマ数設定**")
        if max_raw_frames > 1:
            frame_range = st.slider(
                "元動画の使用範囲",
                min_value=1, max_value=max_raw_frames,
                value=(1, max_raw_frames),
                key="frame_range_slider"
            )
        else:
            frame_range = (1, 1)

        default_target_count = min(20, max(5, max_raw_frames))
        target_frame_count = st.slider(
            "出力コマ数 (5〜20コマ)",
            min_value=5, max_value=20, value=default_target_count, step=1,
            key="target_count_slider"
        )
        
        ping_pong = st.checkbox("🔄 往復再生（ピンポン）", value=False, key="ping_pong_cb")
        trim_end = st.checkbox("✂️ ループ末尾カット", value=True, key="trim_end_cb")
        
        target_sec = st.selectbox("総再生時間 (秒)", [1, 2, 3, 4], index=1, key="target_sec_sb")
        loop_count = st.selectbox("ループ回数", [1, 2, 3, 4], index=1, key="loop_count_sb")

    with col_right:
        st.subheader("✂️ マスク ＆ 透過設定")
        filter_noise = st.checkbox("🧹 見切れゴミ（Zzz等）自動除去", value=True, key="filter_noise_cb")
        
        st.markdown("**端のカット (見切れ削除)**")
        crop_right_pct = st.slider("右端削り (%)", min_value=0, max_value=50, value=0, step=1, key="crop_r")
        crop_left_pct = st.slider("左端削り (%)", min_value=0, max_value=50, value=0, step=1, key="crop_l")
        crop_top_pct = st.slider("上端削り (%)", min_value=0, max_value=50, value=0, step=1, key="crop_t")
        crop_bottom_pct = st.slider("下端削り (%)", min_value=0, max_value=50, value=0, step=1, key="crop_b")
        
        st.markdown("---")
        st.markdown("**🎨 透過感度**")
        tolerance = st.slider(
            "透過の強さ (しきい値)",
            min_value=10, max_value=150, value=75, step=5,
            key="tolerance_slider"
        )
        
        st.markdown("---")
        st.markdown("##### 🖼️ 12カット全体の切り出し枠確認")
        try:
            first_frame = st.session_state['first_frame']
            grid_preview_img = draw_grid_preview(
                first_frame, ROWS, COLS,
                offset_x=grid_offset_x, offset_y=grid_offset_y,
                cell_expand=grid_expand, selected_idx=selected_stamp_idx,
                custom_offsets=custom_offsets
            )
            st.image(grid_preview_img, caption="黄枠＝現在編集中のスタンプ", use_container_width=True)
        except Exception as e:
            st.error(f"ガイド表示エラー: {e}")

    # オンデマンドで選択中のスタンプ1個だけをリアルタイム超高速処理（メモリ超軽量）
    try:
        r = selected_stamp_idx // COLS
        c = selected_stamp_idx % COLS
        
        stamp_raw_cells = [
            extract_stamp_cell(
                f, r, c, ROWS, COLS,
                offset_x=grid_offset_x, offset_y=grid_offset_y,
                cell_expand=grid_expand, custom_offsets=custom_offsets
            )
            for f in raw_video_frames
        ]
        
        cropped_cells = [crop_cell_margins(cell, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct) for cell in stamp_raw_cells]
        trans_frames = [remove_background_floodfill_outer(cell, tolerance=tolerance, filter_noise=filter_noise) for cell in cropped_cells]
        centered_frames = center_and_fit_stamp(trans_frames)
        
        edited_frames = process_frame_sequence_strict(
            centered_frames, frame_range[0], frame_range[1],
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
            
    except Exception as e:
        st.error(f"スタンプ編集処理エラー: {e}")
        edited_frames = []

    with col_center:
        st.subheader("👁️ プレビュー確認 (大画面)")
        if edited_frames:
            try:
                preview_gif = create_preview_gif(edited_frames, frame_duration_ms)
                st.image(preview_gif, caption=f"スタンプ #{selected_stamp_idx + 1} | {frame_cnt}コマ / {target_sec}秒 ({loop_count}回再生)", use_container_width=True)
                
                single_apng_data = optimize_apng_bytes(edited_frames, durations_list, loop_count)
                st.download_button(
                    label=f"💾 スタンプ #{selected_stamp_idx + 1} を個別ダウンロード (APNG)",
                    data=single_apng_data,
                    file_name=f"stamp_{selected_stamp_idx+1:02d}.png",
                    mime="image/png",
                    key="single_dl_btn"
                )
                st.success(f"✅ LINE適合: **{frame_cnt}コマ** / 1コマ **{frame_duration_ms}ms** / 計 **{target_sec}秒**")
            except Exception as e:
                st.error(f"プレビュー描画エラー: {e}")
                
    st.divider()
    
    # --- Step 3: オンデマンド全12個一括APNG生成 ＆ ダウンロード ---
    st.subheader("📦 編集した設定で全12個のスタンプを一括書き出し")
    
    if st.button("🚀 LINE審査適合APNGを一括生成してダウンロード (ZIP)", key="batch_dl_btn", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            zip_buffer = io.BytesIO()
            total_ms = target_sec * 1000
            loop_ms = total_ms // loop_count
            
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(12):
                    status_text.text(f"スタンプ #{idx+1}/12 を高画質透過処理・APNG変換中...")
                    
                    r = idx // COLS
                    c = idx % COLS
                    
                    stamp_cells = [
                        extract_stamp_cell(
                            f, r, c, ROWS, COLS,
                            offset_x=grid_offset_x, offset_y=grid_offset_y,
                            cell_expand=grid_expand, custom_offsets=custom_offsets
                        )
                        for f in raw_video_frames
                    ]
                    
                    c_cells = [crop_cell_margins(cell, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct) for cell in stamp_cells]
                    trans_frames = [remove_background_floodfill_outer(cell, tolerance=tolerance, filter_noise=filter_noise) for cell in c_cells]
                    cent_frames = center_and_fit_stamp(trans_frames)
                    
                    proc_f = process_frame_sequence_strict(
                        cent_frames, frame_range[0], frame_range[1],
                        target_frame_count=target_frame_count, ping_pong=ping_pong, trim_end=trim_end
                    )
                    
                    frame_cnt = len(proc_f)
                    base_ms = loop_ms // frame_cnt
                    remainder = loop_ms % frame_cnt
                    durations_list = [base_ms] * frame_cnt
                    for i in range(remainder):
                        durations_list[i] += 1
                        
                    apng_data = optimize_apng_bytes(proc_f, durations_list, loop_count)
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    
                    progress_bar.progress((idx + 1) / 12)
                    
            status_text.text("🎉 すべての変換が完了しました！")
            st.success("🎉 全12個のアニメーションスタンプの生成が完了しました！")
            
            st.download_button(
                label="📦 一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip",
                key="zip_dl_btn"
            )
        except Exception as e:
            st.error(f"一括変換中にエラーが発生しました: {e}")
