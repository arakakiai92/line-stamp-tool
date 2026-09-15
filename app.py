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
st.caption("均等グリッド分割に加え、不均等配置・余白・見切れ動画に対応した【自動領域検出モード】搭載版です。")

if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None
if 'fps' not in st.session_state:
    st.session_state['fps'] = 10
if 'total_frames' not in st.session_state:
    st.session_state['total_frames'] = 0
if 'video_w' not in st.session_state:
    st.session_state['video_w'] = 800
if 'video_h' not in st.session_state:
    st.session_state['video_h'] = 600
if 'is_vertical' not in st.session_state:
    st.session_state['is_vertical'] = False

# --- 領域自動検出関数 ---
def detect_content_boxes(frame_bgr, min_area_pct=0.03, filter_cut_edges=False):
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # 白背景（>235）以外のコンテンツ領域を抽出
    non_bg = (gray < 235).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(non_bg, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    boxes = []
    min_area = h * w * min_area_pct
    
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area >= min_area and bw > 30 and bh > 30:
            if filter_cut_edges:
                # 画面の左右端（見切れ）に接触している要素を除外
                if x <= 5 or (x + bw) >= (w - 5):
                    continue
            boxes.append((int(x), int(y), int(bw), int(bh)))
            
    # 左から右、上から下の順にソート
    boxes.sort(key=lambda b: (b[1] // 50, b[0]))
    return boxes

# --- 動画のロード ---
uploaded_file = st.file_uploader("1. 動画ファイル (MP4 / MOV) をアップロードしてください", type=["mp4", "mov"], key="uploader")

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            uploaded_file.seek(0)
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
            
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 800
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 600
        cap.release()
        
        is_vertical = h > w
        
        st.session_state['fps'] = fps
        st.session_state['total_frames'] = total_f
        st.session_state['video_path'] = temp_path
        st.session_state['video_w'] = w
        st.session_state['video_h'] = h
        st.session_state['is_vertical'] = is_vertical
        
        gc.collect()
    except Exception as e:
        st.error(f"動画ロードエラー: {e}")

# 分割レイアウトの選択
boxes_list = []
is_auto_detect = False

if st.session_state.get('video_path') is not None:
    w = st.session_state['video_w']
    h = st.session_state['video_h']
    is_vertical = st.session_state['is_vertical']
    video_path = st.session_state['video_path']
    
    st.markdown("---")
    st.subheader("📐 分割レイアウトの選択")
    
    if is_vertical:
        st.info(f"📱 **縦構図動画** を検出（解像度: {w} × {h}）")
        default_options = [
            "🤖 白枠・イラスト領域を自動検出 (推奨)",
            "6カット (2列 × 3行)",
            "8カット (2列 × 4行)",
            "4カット (2列 × 2行)",
            "カスタムグリッド指定"
        ]
    else:
        st.info(f"🖥️ **横構図動画** を検出（解像度: {w} × {h}）")
        default_options = [
            "🤖 白枠・イラスト領域を自動検出 (推奨)",
            "10カット (5列 × 2行)",
            "12カット (4列 × 3行)",
            "6カット (2列 × 3行)",
            "カスタムグリッド指定"
        ]

    grid_mode = st.radio(
        "動画の切り出しレイアウトを選択してください",
        default_options,
        horizontal=True,
        key="grid_mode_radio"
    )
    
    if "自動検出" in grid_mode:
        is_auto_detect = True
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_cut = st.checkbox("左右の端で見切れている不完全な枠を除外する（完全な中央2枚のみ抽出）", value=False)
        with col_f2:
            min_area_val = st.slider("検出最小サイズ比率 (%)", min_value=1, max_value=20, value=3, step=1)
            
        cap_first = cv2.VideoCapture(video_path)
        ret_f, first_frame = cap_first.read()
        cap_first.release()
        
        if ret_f:
            boxes_list = detect_content_boxes(first_frame, min_area_pct=min_area_val/100.0, filter_cut_edges=filter_cut)
            total_stamps = len(boxes_list)
            st.success(f"🎯 **{total_stamps} 個** のカード領域を自動検出しました！")
        else:
            boxes_list = []
            total_stamps = 0
    else:
        is_auto_detect = False
        if "10カット" in grid_mode:
            ROWS, COLS = 2, 5
        elif "12カット" in grid_mode:
            ROWS, COLS = 3, 4
        elif "8カット" in grid_mode:
            ROWS, COLS = 4, 2
        elif "6カット" in grid_mode:
            ROWS, COLS = 3, 2
        elif "4カット" in grid_mode:
            ROWS, COLS = 2, 2
        else:
            col_c, col_r = st.columns(2)
            with col_c:
                COLS = st.number_input("横の列数 (Columns)", min_value=1, max_value=10, value=2 if is_vertical else 5, step=1)
            with col_r:
                ROWS = st.number_input("縦の行数 (Rows)", min_value=1, max_value=10, value=4 if is_vertical else 2, step=1)
        total_stamps = ROWS * COLS
        st.session_state['ROWS'] = ROWS
        st.session_state['COLS'] = COLS

    st.session_state['total_stamps'] = total_stamps

# --- 切り出し処理 ---
def get_stamp_crop_coords(idx, frame_shape, is_auto, boxes, rows, cols, offset_x, offset_y, cell_expand, custom_offsets):
    h_f, w_f = frame_shape[:2]
    c_ox, c_oy, c_exp = custom_offsets.get(idx, (0, 0, 0))
    
    if is_auto and idx < len(boxes):
        bx, by, bw, bh = boxes[idx]
        x1 = bx + offset_x + c_ox - (cell_expand + c_exp)
        y1 = by + offset_y + c_oy - (cell_expand + c_exp)
        x2 = bx + bw + offset_x + c_ox + (cell_expand + c_exp)
        y2 = by + bh + offset_y + c_oy + (cell_expand + c_exp)
    else:
        r = idx // cols
        c = idx % cols
        cell_h = h_f // rows
        cell_w = w_f // cols
        x1 = c * cell_w + offset_x + c_ox - (cell_expand + c_exp)
        y1 = r * cell_h + offset_y + c_oy - (cell_expand + c_exp)
        x2 = (c + 1) * cell_w + offset_x + c_ox + (cell_expand + c_exp)
        y2 = (r + 1) * cell_h + offset_y + c_oy + (cell_expand + c_exp)
        
    x1_c = max(0, min(w_f - 10, int(x1)))
    y1_c = max(0, min(h_f - 10, int(y1)))
    x2_c = max(x1_c + 10, min(w_f, int(x2)))
    y2_c = max(y1_c + 10, min(h_f, int(y2)))
    return x1_c, y1_c, x2_c, y2_c

def load_stamp_frames(video_path, start_f, end_f, idx, is_auto, boxes, rows, cols, offset_x, offset_y, cell_expand, custom_offsets):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_f - 1))
    extracted_frames = []
    current_f = start_f
    
    while current_f <= end_f and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        x1, y1, x2, y2 = get_stamp_crop_coords(idx, frame.shape, is_auto, boxes, rows, cols, offset_x, offset_y, cell_expand, custom_offsets)
        extracted_frames.append(frame[y1:y2, x1:x2])
        current_f += 1
        
    cap.release()
    gc.collect()
    return extracted_frames

def draw_preview_boxes(frame_bgr, is_auto, boxes, rows, cols, offset_x=0, offset_y=0, cell_expand=0, selected_idx=0, custom_offsets=None):
    preview = frame_bgr.copy()
    if custom_offsets is None:
        custom_offsets = {}
        
    total = len(boxes) if is_auto else (rows * cols)
    for idx in range(total):
        x1, y1, x2, y2 = get_stamp_crop_coords(idx, frame_bgr.shape, is_auto, boxes, rows, cols, offset_x, offset_y, cell_expand, custom_offsets)
        color = (0, 255, 255) if idx == selected_idx else (0, 0, 255)
        thickness = 3 if idx == selected_idx else 2
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(preview, f"#{idx+1}", (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(preview, f"#{idx+1}", (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

def crop_cell_margins(cell_bgr, crop_left_pct=0, crop_right_pct=0, crop_top_pct=0, crop_bottom_pct=0):
    h, w, _ = cell_bgr.shape
    top = int(h * crop_top_pct / 100.0)
    bottom = h - int(h * crop_bottom_pct / 100.0)
    left = int(w * crop_left_pct / 100.0)
    right = w - int(w * crop_right_pct / 100.0)
    if bottom <= top + 10: bottom = top + 10
    if right <= left + 10: right = left + 10
    return cell_bgr[top:bottom, left:right]

def remove_isolated_noise_alpha(alpha_channel, min_size_pct=0.015):
    h, w = alpha_channel.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((alpha_channel > 127).astype(np.uint8))
    if num_labels <= 2: return alpha_channel
    new_alpha = np.zeros_like(alpha_channel)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    new_alpha[labels == areas[0][0]] = 255
    min_area = (h * w) * min_size_pct
    for idx, area in areas[1:]:
        if area >= min_area: new_alpha[labels == idx] = 255
    return new_alpha

def remove_background_floodfill_sharp(cell_bgr, tolerance=70, filter_noise=True, sharp_edge=True):
    h, w, _ = cell_bgr.shape
    if h < 5 or w < 5: return Image.fromarray(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGBA))
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    img_work = cell_bgr.copy()
    bg_color = cell_bgr[0, 0].astype(np.float32)
    seeds = []
    step = max(1, min(h, w) // 5)
    for x in range(0, w, step):
        seeds.append((x, 0)); seeds.append((x, h - 1))
    for y in range(0, h, step):
        seeds.append((0, y)); seeds.append((w - 1, y))
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    for seed_x, seed_y in seeds:
        if mask[seed_y + 1, seed_x + 1] == 0:
            pixel_color = cell_bgr[seed_y, seed_x].astype(np.float32)
            if np.linalg.norm(pixel_color - bg_color) <= tolerance * 2.0:
                cv2.floodFill(img_work, mask, seedPoint=(seed_x, seed_y), newVal=(0, 0, 0), loDiff=lo_diff, upDiff=up_diff, flags=flags)
    bg_mask = mask[1:h+1, 1:w+1]
    alpha = np.where(bg_mask == 255, 0, 255).astype(np.uint8)
    if filter_noise: alpha = remove_isolated_noise_alpha(alpha)
    if not sharp_edge:
        alpha = cv2.threshold(cv2.GaussianBlur(alpha, (3, 3), 0), 127, 255, cv2.THRESH_BINARY)[1]
    b, g, r = cv2.split(cell_bgr)
    return Image.fromarray(cv2.cvtColor(cv2.merge([b, g, r, alpha]), cv2.COLOR_BGRA2RGBA))

def process_frame_sequence_strict(frames, target_frame_count=15, ping_pong=False, trim_end=False):
    sub = frames
    if not sub: return sub
    if trim_end and len(sub) > 2: sub = sub[:-1]
    if ping_pong and len(sub) > 2: sub = sub + sub[-2:0:-1]
    target_count = max(5, min(20, target_frame_count))
    if len(sub) != target_count:
        indices = np.linspace(0, len(sub) - 1, target_count, dtype=int)
        sub = [sub[i] for i in indices]
    return sub

def create_preview_gif(frame_list, duration_ms):
    if not frame_list: return b""
    buf = io.BytesIO()
    frame_list[0].save(buf, format="GIF", save_all=True, append_images=frame_list[1:], duration=duration_ms, loop=0, disposal=2)
    return buf.getvalue()

def export_apng_lossless(img_list, durations, loop_count):
    buf = io.BytesIO()
    img_list[0].save(buf, format="PNG", save_all=True, append_images=img_list[1:], duration=durations, loop=loop_count)
    return buf.getvalue()

# --- ワークスペース ---
if st.session_state.get('video_path') is not None and total_stamps > 0:
    ROWS = st.session_state.get('ROWS', 2)
    COLS = st.session_state.get('COLS', 5)
    total_original_frames = st.session_state.get('total_frames', 30)
    video_path = st.session_state['video_path']
    
    st.divider()
    st.header("🎛️ スタンプ編集ワークスペース")
    
    col_left, col_center, col_right = st.columns([1.1, 1.3, 1.1])
    
    with col_left:
        st.subheader("🎞️ アニメ ＆ 部分選択設定")
        stamp_num_val = st.selectbox("編集するスタンプ番号を選択", options=list(range(1, total_stamps + 1)), format_func=lambda x: f"スタンプ #{x}", key="stamp_selectbox")
        selected_stamp_idx = stamp_num_val - 1
        
        st.markdown("---")
        st.markdown(f"##### ✂️ 動画の部分選択（全 {total_original_frames} コマ）")
        frame_range = st.slider("使用区間", min_value=1, max_value=max(1, total_original_frames), value=(1, max(1, total_original_frames)), key="frame_range_slider")

        st.markdown("##### ⏱️ LINE出力設定")
        target_frame_count = st.slider("LINE出力コマ数 (5〜20コマ)", min_value=5, max_value=20, value=min(20, max(5, total_original_frames)), step=1, key="target_count_slider")
        ping_pong = st.checkbox("🔄 往復再生（ピンポン）", value=False, key="ping_pong_cb")
        trim_end = st.checkbox("✂️ ループ末尾カット", value=True, key="trim_end_cb")
        
        st.markdown("---")
        col_sec, col_loop = st.columns(2)
        with col_sec:
            target_sec = st.selectbox("総再生時間 (秒)", [1, 2, 3, 4], index=1, key="target_sec_sb")
        with col_loop:
            loop_count = st.selectbox("ループ回数", [1, 2, 3, 4], index=1, key="loop_count_sb")

        st.markdown("---")
        with st.expander("📐 位置・余白微調整", expanded=False):
            grid_offset_x = st.slider("全体を左右移動 (px)", -100, 100, 0, 1, key="g_ox")
            grid_offset_y = st.slider("全体を上下移動 (px)", -100, 100, 0, 1, key="g_oy")
            grid_expand = st.slider("切り出し枠拡大 (px)", 0, 100, 0, 2, key="g_exp")
            
            st.markdown(f"**スタンプ #{stamp_num_val} 個別微調整**")
            ind_ox = st.slider("個別左右移動", -50, 50, 0, 1, key="ind_ox")
            ind_oy = st.slider("個別上下移動", -50, 50, 0, 1, key="ind_oy")
            ind_exp = st.slider("個別枠拡大", -30, 50, 0, 1, key="ind_exp")
            
        custom_offsets = {selected_stamp_idx: (ind_ox, ind_oy, ind_exp)}

    with col_right:
        st.subheader("✂️ マスク ＆ 画質設定")
        filter_noise = st.checkbox("🧹 ゴミ自動除去", value=True, key="filter_noise_cb")
        sharp_edge = st.checkbox("🔪 シャープ透過", value=True, key="sharp_edge_cb")
        
        st.markdown("**端のカット (黒枠・外枠除去)**")
        crop_right_pct = st.slider("右端削り (%)", 0, 30, 0, 1, key="crop_r")
        crop_left_pct = st.slider("左端削り (%)", 0, 30, 0, 1, key="crop_l")
        crop_top_pct = st.slider("上端削り (%)", 0, 30, 0, 1, key="crop_t")
        crop_bottom_pct = st.slider("下端削り (%)", 0, 30, 0, 1, key="crop_b")
        
        st.markdown("---")
        tolerance = st.slider("透過の強さ (背景が白い場合はこのままでOK)", 0, 150, 75, 5, key="tolerance_slider")
        
        st.markdown("---")
        with st.expander(f"🖼️ 全 {total_stamps} カットの切り出し枠確認", expanded=True):
            try:
                cap_first = cv2.VideoCapture(video_path)
                ret_f, first_frame = cap_first.read()
                cap_first.release()
                if ret_f:
                    grid_preview_img = draw_preview_boxes(
                        first_frame, is_auto_detect, boxes_list, ROWS, COLS,
                        offset_x=grid_offset_x, offset_y=grid_offset_y,
                        cell_expand=grid_expand, selected_idx=selected_stamp_idx,
                        custom_offsets=custom_offsets
                    )
                    st.image(grid_preview_img, caption="黄枠＝現在選択中のスタンプ", use_container_width=True)
            except Exception as e:
                st.error(f"ガイド表示エラー: {e}")

    # オンデマンド抽出
    try:
        stamp_raw_cells = load_stamp_frames(
            video_path, frame_range[0], frame_range[1], selected_stamp_idx,
            is_auto_detect, boxes_list, ROWS, COLS,
            grid_offset_x, grid_offset_y, grid_expand, custom_offsets
        )
        
        cropped_cells = [crop_cell_margins(cell, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct) for cell in stamp_raw_cells]
        trans_frames = [remove_background_floodfill_sharp(cell, tolerance=tolerance, filter_noise=filter_noise, sharp_edge=sharp_edge) for cell in cropped_cells]
        
        edited_frames = process_frame_sequence_strict(
            trans_frames, target_frame_count=target_frame_count, ping_pong=ping_pong, trim_end=trim_end
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
                st.image(preview_gif, caption=f"スタンプ #{selected_stamp_idx + 1} | 区間: #{frame_range[0]}〜#{frame_range[1]} コマ ➔ 出力: {frame_cnt}コマ", use_container_width=True)
                
                single_apng_data = export_apng_lossless(edited_frames, durations_list, loop_count)
                st.download_button(
                    label=f"💾 スタンプ #{selected_stamp_idx + 1} を無劣化個別ダウンロード (APNG)",
                    data=single_apng_data,
                    file_name=f"stamp_{selected_stamp_idx+1:02d}.png",
                    mime="image/png",
                    key="single_dl_btn"
                )
                st.success(f"✨ 全{frame_cnt}コマ / 1コマ **{frame_duration_ms}ms**")
            except Exception as e:
                st.error(f"プレビュー描画エラー: {e}")
                
    st.divider()
    
    # --- 一括書き出し ---
    st.subheader(f"📦 全 {total_stamps} 個のスタンプを一括書き出し")
    
    if st.button(f"🚀 無劣化フルカラーAPNGを全 {total_stamps} 個一括生成してダウンロード (ZIP)", key="batch_dl_btn", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            zip_buffer = io.BytesIO()
            total_ms = target_sec * 1000
            loop_ms = total_ms // loop_count
            
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(total_stamps):
                    status_text.text(f"スタンプ #{idx+1}/{total_stamps} を無劣化フルカラーでAPNG変換中...")
                    
                    stamp_cells = load_stamp_frames(
                        video_path, frame_range[0], frame_range[1], idx,
                        is_auto_detect, boxes_list, ROWS, COLS,
                        grid_offset_x, grid_offset_y, grid_expand, custom_offsets
                    )
                    
                    c_cells = [crop_cell_margins(cell, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct) for cell in stamp_cells]
                    trans_frames = [remove_background_floodfill_sharp(cell, tolerance=tolerance, filter_noise=filter_noise, sharp_edge=sharp_edge) for cell in c_cells]
                    proc_f = process_frame_sequence_strict(
                        trans_frames, target_frame_count=target_frame_count, ping_pong=ping_pong, trim_end=trim_end
                    )
                    
                    f_cnt = len(proc_f)
                    b_ms = loop_ms // f_cnt
                    rem = loop_ms % f_cnt
                    d_list = [b_ms] * f_cnt
                    for i in range(rem):
                        d_list[i] += 1
                        
                    apng_data = export_apng_lossless(proc_f, d_list, loop_count)
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    progress_bar.progress((idx + 1) / total_stamps)
                    
            status_text.text("🎉 すべての変換が完了しました！")
            st.success(f"🎉 全 {total_stamps} 個の無劣化アニメーションスタンプの生成が完了しました！")
            st.download_button(
                label="📦 一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip",
                key="zip_dl_btn"
            )
        except Exception as e:
            st.error(f"一括変換中にエラーが発生しました: {e}")
