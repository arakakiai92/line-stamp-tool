import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os
import tempfile
import traceback
import gc

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成＆高度編集ツール", layout="wide")

st.title("🎬 LINEアニメーションスタンプ自動生成 ＆ 高度編集ツール")
st.caption("見切れボツカット自動除外＆不均等レイアウト対応【16:9横長・多分割対応版】")

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
if 'custom_offsets_dict' not in st.session_state:
    st.session_state['custom_offsets_dict'] = {}

# --- 見切れボツカット自動除外付き 領域検出（16:9・多分割対応） ---
def detect_content_boxes_robust(frame_bgr, auto_filter_cut=True):
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # 白背景（>235）以外の領域を抽出
    non_bg = (gray < 235).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(non_bg, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    raw_boxes = []
    
    # 12分割以上にも対応できるよう最小面積を緩和 (画面全体の0.8%以上)
    min_area = h * w * 0.008
    min_dim = min(w, h) * 0.04  # 最小サイズも解像度基準に動的化
    
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area >= min_area and bw > min_dim and bh > min_dim:
            raw_boxes.append((int(x), int(y), int(bw), int(bh)))
            
    if not raw_boxes:
        return []
        
    if auto_filter_cut:
        # 画面の端（余白5px以下）に触れている見切れを除外
        complete_boxes = []
        for x, y, bw, bh in raw_boxes:
            is_edge_cut = (x <= 5) or ((x + bw) >= (w - 5)) or (y <= 5) or ((y + bh) >= (h - 5))
            if not is_edge_cut:
                complete_boxes.append((x, y, bw, bh))
                
        if len(complete_boxes) > 0:
            median_w = np.median([b[2] for b in complete_boxes])
            filtered = [b for b in complete_boxes if b[2] >= median_w * 0.6]
            boxes = filtered if filtered else complete_boxes
        else:
            boxes = raw_boxes
    else:
        boxes = raw_boxes
        
    if not boxes:
        return []

    # 行判定のしきい値をスタンプ平均高さの半分に動的設定（上から下・左から右へ確実に整列）
    avg_h = np.mean([b[3] for b in boxes])
    row_bin = max(20, int(avg_h * 0.6))
    boxes.sort(key=lambda b: (b[1] // row_bin, b[0]))
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
        
        st.session_state['fps'] = fps
        st.session_state['total_frames'] = total_f
        st.session_state['video_path'] = temp_path
        st.session_state['video_w'] = w
        st.session_state['video_h'] = h
        st.session_state['is_vertical'] = (h > w)
        
        gc.collect()
    except Exception as e:
        st.error(f"動画ロードエラー: {e}")

# 分割レイアウトの選択
boxes_list = []
is_auto_detect = True
total_stamps = 0
ROWS, COLS = 1, 2

if st.session_state.get('video_path') is not None:
    w = st.session_state['video_w']
    h = st.session_state['video_h']
    is_vertical = st.session_state['is_vertical']
    video_path = st.session_state['video_path']
    
    st.markdown("---")
    st.subheader("📐 分割レイアウトの選択")
    
    default_options = [
        "🤖 自動領域検出（見切れボツカットを自動除外）",
        "均等 12カット (4列 × 3行) 【16:9横長向け】",
        "均等 8カット (4列 × 2行)",
        "均等 6カット (2列 × 3行)",
        "均等 8カット (2列 × 4行)",
        "均等 10カット (5列 × 2行)",
        "カスタム均等グリッド指定"
    ]

    grid_mode = st.radio(
        "切り出しレイアウトを選択してください",
        default_options,
        index=0,
        horizontal=True,
        key="grid_mode_radio"
    )
    
    if "自動領域検出" in grid_mode:
        is_auto_detect = True
        include_cut_edges = st.checkbox("端の見切れボツカットも含めてすべて抽出する（通常はOFFのままでOK）", value=False)
        
        cap_first = cv2.VideoCapture(video_path)
        ret_f, first_frame = cap_first.read()
        cap_first.release()
        
        if ret_f:
            boxes_list = detect_content_boxes_robust(first_frame, auto_filter_cut=(not include_cut_edges))
            total_stamps = len(boxes_list)
            st.success(f"🎯 **有効なスタンプ {total_stamps} 個** を自動検出しました！（解像度: {w}x{h}）")
        else:
            boxes_list = []
            total_stamps = 0
    else:
        is_auto_detect = False
        if "12カット" in grid_mode:
            ROWS, COLS = 3, 4
        elif "8カット (4列" in grid_mode:
            ROWS, COLS = 2, 4
        elif "6カット" in grid_mode:
            ROWS, COLS = 3, 2
        elif "8カット (2列" in grid_mode:
            ROWS, COLS = 4, 2
        elif "10カット" in grid_mode:
            ROWS, COLS = 2, 5
        else:
            col_c, col_r = st.columns(2)
            with col_c:
                COLS = st.number_input("横の列数", min_value=1, max_value=12, value=4 if not is_vertical else 2, step=1)
            with col_r:
                ROWS = st.number_input("縦の行数", min_value=1, max_value=12, value=3 if not is_vertical else 4, step=1)
        total_stamps = ROWS * COLS

    st.session_state['total_stamps'] = total_stamps
    st.session_state['boxes_list'] = boxes_list
    st.session_state['is_auto_detect'] = is_auto_detect
    st.session_state['ROWS'] = ROWS
    st.session_state['COLS'] = COLS

# --- 座標計算関数 ---
def get_stamp_crop_coords(idx, frame_shape, is_auto, boxes, rows, cols, offset_x, offset_y, cell_expand, custom_offsets):
    h_f, w_f = frame_shape[:2]
    c_ox, c_oy, c_exp = custom_offsets.get(idx, (0, 0, 0))
    
    if is_auto:
        if idx < len(boxes):
            bx, by, bw, bh = boxes[idx]
        else:
            bx, by, bw, bh = 0, 0, w_f, h_f
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
    extracted = []
    curr = start_f
    while curr <= end_f and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        x1, y1, x2, y2 = get_stamp_crop_coords(idx, frame.shape, is_auto, boxes, rows, cols, offset_x, offset_y, cell_expand, custom_offsets)
        extracted.append(frame[y1:y2, x1:x2])
        curr += 1
    cap.release()
    gc.collect()
    return extracted

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
        cv2.putText(preview, f"#{idx+1}", (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(preview, f"#{idx+1}", (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

def crop_cell_margins(cell_bgr, crop_l=0, crop_r=0, crop_t=0, crop_b=0):
    h, w, _ = cell_bgr.shape
    top = int(h * crop_t / 100.0)
    bottom = h - int(h * crop_b / 100.0)
    left = int(w * crop_l / 100.0)
    right = w - int(w * crop_r / 100.0)
    if bottom <= top + 10: bottom = top + 10
    if right <= left + 10: right = left + 10
    return cell_bgr[top:bottom, left:right]

def remove_background_floodfill(cell_bgr, tolerance=70, filter_noise=True, sharp_edge=True):
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
    total_original_frames = st.session_state.get('total_frames', 30)
    video_path = st.session_state['video_path']
    
    st.divider()
    st.header("🎛️ スタンプ編集ワークスペース")
    
    col_left, col_center, col_right = st.columns([1.1, 1.3, 1.1])
    
    with col_left:
        st.subheader("🎞️ アニメ ＆ 部分選択設定")
        
        stamp_options = list(range(1, total_stamps + 1))
        if 'cur_stamp_num' not in st.session_state or st.session_state['cur_stamp_num'] not in stamp_options:
            st.session_state['cur_stamp_num'] = stamp_options[0]
            
        stamp_num_val = st.selectbox(
            "編集するスタンプ番号を選択",
            options=stamp_options,
            format_func=lambda x: f"スタンプ #{x}",
            index=stamp_options.index(st.session_state['cur_stamp_num']),
            key="stamp_selectbox"
        )
        st.session_state['cur_stamp_num'] = stamp_num_val
        selected_stamp_idx = stamp_num_val - 1
        
        st.markdown("---")
        st.markdown(f"##### ✂️ 再生区間設定（全 {total_original_frames} コマ）")
        frame_range = st.slider("コマ区間", min_value=1, max_value=max(1, total_original_frames), value=(1, max(1, total_original_frames)), key="frame_range_slider")

        st.markdown("##### ⏱️ LINE出力設定")
        target_frame_count = st.slider("出力コマ数 (5〜20コマ)", min_value=5, max_value=20, value=min(20, max(5, total_original_frames)), step=1, key="target_count_slider")
        ping_pong = st.checkbox("🔄 往復再生（ピンポン再生）", value=False, key="ping_pong_cb")
        trim_end = st.checkbox("✂️ ループ末尾カット", value=True, key="trim_end_cb")
        
        st.markdown("---")
        col_sec, col_loop = st.columns(2)
        with col_sec:
            target_sec = st.selectbox("総再生時間 (秒)", [1, 2, 3, 4], index=1, key="target_sec_sb")
        with col_loop:
            loop_count = st.selectbox("ループ回数", [1, 2, 3, 4], index=1, key="loop_count_sb")

        st.markdown("---")
        with st.expander("📐 位置・余白微調整", expanded=False):
            grid_offset_x = st.slider("全体左右移動 (px)", -100, 100, 0, 1, key="g_ox")
            grid_offset_y = st.slider("全体上下移動 (px)", -100, 100, 0, 1, key="g_oy")
            grid_expand = st.slider("切り出し枠拡大 (px)", -20, 100, 0, 2, key="g_exp")
            
            st.markdown(f"**スタンプ #{stamp_num_val} 個別微調整**")
            ind_ox = st.slider("個別左右移動", -50, 50, 0, 1, key="ind_ox")
            ind_oy = st.slider("個別上下移動", -50, 50, 0, 1, key="ind_oy")
            ind_exp = st.slider("個別枠拡大", -30, 50, 0, 1, key="ind_exp")
            
        st.session_state['custom_offsets_dict'][selected_stamp_idx] = (ind_ox, ind_oy, ind_exp)
        custom_offsets = st.session_state['custom_offsets_dict']

    with col_right:
        st.subheader("✂️ 透過 ＆ マスク設定")
        
        use_transparency = st.checkbox("🎨 白背景を透過する", value=True, help="チェックを外すと、背景白のままフルカラー出力されます。")
        
        if use_transparency:
            tolerance = st.slider("透過感度 (しきい値)", 10, 150, 75, 5, key="tolerance_slider")
            sharp_edge = st.checkbox("🔪 シャープ透過（輪郭クッキリ）", value=True, key="sharp_edge_cb")
        else:
            tolerance = 0
            sharp_edge = True
            
        st.markdown("---")
        st.markdown("**カード端の削り (枠線の除去用)**")
        crop_r = st.slider("右端削り (%)", 0, 30, 0, 1, key="crop_r")
        crop_l = st.slider("左端削り (%)", 0, 30, 0, 1, key="crop_l")
        crop_t = st.slider("上端削り (%)", 0, 30, 0, 1, key="crop_t")
        crop_b = st.slider("下端削り (%)", 0, 30, 0, 1, key="crop_b")
        
        st.markdown("---")
        with st.expander(f"🖼️ 切り出し枠の確認（全 {total_stamps} 個）", expanded=True):
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
                    st.image(grid_preview_img, caption="黄枠＝現在編集中のスタンプ", use_container_width=True)
            except Exception as e:
                st.error(f"ガイド表示エラー: {e}")

    try:
        stamp_raw_cells = load_stamp_frames(
            video_path, frame_range[0], frame_range[1], selected_stamp_idx,
            is_auto_detect, boxes_list, ROWS, COLS,
            grid_offset_x, grid_offset_y, grid_expand, custom_offsets
        )
        
        cropped_cells = [crop_cell_margins(cell, crop_l, crop_r, crop_t, crop_b) for cell in stamp_raw_cells]
        
        if use_transparency:
            processed_frames = [remove_background_floodfill(cell, tolerance=tolerance, sharp_edge=sharp_edge) for cell in cropped_cells]
        else:
            processed_frames = [Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGBA)) for cell in cropped_cells]
            
        edited_frames = process_frame_sequence_strict(
            processed_frames, target_frame_count=target_frame_count, ping_pong=ping_pong, trim_end=trim_end
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
        st.code(traceback.format_exc())
        edited_frames = []

    with col_center:
        st.subheader("👁️ プレビュー確認 (大画面)")
        if edited_frames:
            try:
                preview_gif = create_preview_gif(edited_frames, frame_duration_ms)
                st.image(preview_gif, caption=f"スタンプ #{selected_stamp_idx + 1} | コマ数: {frame_cnt}コマ / 1コマ {frame_duration_ms}ms", use_container_width=True)
                
                single_apng_data = export_apng_lossless(edited_frames, durations_list, loop_count)
                st.download_button(
                    label=f"💾 スタンプ #{selected_stamp_idx + 1} をダウンロード (APNG)",
                    data=single_apng_data,
                    file_name=f"stamp_{selected_stamp_idx+1:02d}.png",
                    mime="image/png",
                    key="single_dl_btn"
                )
                st.success(f"✨ 正常稼働中: 全{frame_cnt}コマ (1コマあたり {frame_duration_ms}ms)")
            except Exception as e:
                st.error(f"プレビュー描画エラー: {e}")
                
    st.divider()
    
    # --- 一括書き出し ---
    st.subheader(f"📦 有効な全 {total_stamps} 個のスタンプを一括書き出し")
    
    if st.button(f"🚀 有効な全 {total_stamps} 個を一括生成してダウンロード (ZIP)", key="batch_dl_btn", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            zip_buffer = io.BytesIO()
            total_ms = target_sec * 1000
            loop_ms = total_ms // loop_count
            
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(total_stamps):
                    status_text.text(f"スタンプ #{idx+1}/{total_stamps} を高画質変換中...")
                    
                    stamp_cells = load_stamp_frames(
                        video_path, frame_range[0], frame_range[1], idx,
                        is_auto_detect, boxes_list, ROWS, COLS,
                        grid_offset_x, grid_offset_y, grid_expand, custom_offsets
                    )
                    
                    c_cells = [crop_cell_margins(cell, crop_l, crop_r, crop_t, crop_b) for cell in stamp_cells]
                    if use_transparency:
                        p_frames = [remove_background_floodfill(cell, tolerance=tolerance, sharp_edge=sharp_edge) for cell in c_cells]
                    else:
                        p_frames = [Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGBA)) for cell in c_cells]
                        
                    proc_f = process_frame_sequence_strict(
                        p_frames, target_frame_count=target_frame_count, ping_pong=ping_pong, trim_end=trim_end
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
            st.success(f"🎉 有効な全 {total_stamps} 個のスタンプ生成が完了しました！")
            st.download_button(
                label="📦 一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip",
                key="zip_dl_btn"
            )
        except Exception as e:
            st.error(f"一括変換中にエラーが発生しました: {e}")
            st.code(traceback.format_exc())
