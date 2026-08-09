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
st.caption("カット番号の切り替えによるメモリ蓄積を完全に解消した【超軽量・オンデマンド処理版】です。")

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

# --- 動画のロード（ファイルパスのみ保持してメモリを一切圧迫しない方式） ---
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
        
        gc.collect()
    except Exception as e:
        st.error(f"動画ロードエラー: {e}")

# 分割レイアウトの選択
if st.session_state.get('video_path') is not None:
    st.markdown("---")
    st.subheader("📐 分割レイアウトの選択")
    grid_mode = st.radio(
        "動画の切り出しレイアウトを選択してください",
        ["12カット (4列 × 3行 = 計12個)", "6カット (2列 × 3行 = 計6個)"],
        horizontal=True,
        key="grid_mode_radio"
    )
    
    if "12カット" in grid_mode:
        ROWS, COLS = 3, 4
        total_stamps = 12
    else:
        ROWS, COLS = 3, 2
        total_stamps = 6
        
    st.session_state['ROWS'] = ROWS
    st.session_state['COLS'] = COLS
    st.session_state['total_stamps'] = total_stamps

# --- オンデマンド抽出関数（指定されたカット・指定されたフレーム範囲だけをピンポイントで動画ファイルから読み込む） ---
def load_specific_stamp_frames(video_path, start_f, end_f, r, c, rows, cols, offset_x, offset_y, cell_expand, custom_offsets):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_f - 1))
    
    extracted_frames = []
    current_f = start_f
    
    while current_f <= end_f and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # そのフレームから該当カットのセルだけを抽出
        h_f, w_f, _ = frame.shape
        cell_h = h_f // rows
        cell_w = w_f // cols
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
        
        x1_c = max(0, min(w_f - 10, int(x1)))
        y1_c = max(0, min(h_f - 10, int(y1)))
        x2_c = max(x1_c + 10, min(w_f, int(x2)))
        y2_c = max(y1_c + 10, min(h_f, int(y2)))
        
        cell = frame[y1_c:y2_c, x1_c:x2_c]
        extracted_frames.append(cell)
        current_f += 1
        
    cap.release()
    gc.collect()
    return extracted_frames

def draw_grid_preview(frame_bgr, rows, cols, offset_x=0, offset_y=0, cell_expand=0, selected_idx=0, custom_offsets=None):
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
            
            color = (0, 255, 255) if idx == selected_idx else (0, 0, 255)
            thickness = 3 if idx == selected_idx else 2
            cv2.rectangle(preview, (x1_c, y1_c), (x2_c, y2_c), color, thickness)
            cv2.putText(preview, f"#{idx+1}", (x1_c + 8, y1_c + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(preview, f"#{idx+1}", (x1_c + 8, y1_c + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
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
    total_area = h * w
    min_area = total_area * min_size_pct
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha_channel > 127).astype(np.uint8))
    if num_labels <= 2: return alpha_channel
    new_alpha = np.zeros_like(alpha_channel)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    largest_idx = areas[0][0]
    new_alpha[labels == largest_idx] = 255
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
    if ping_pong and len(sub) > 2:
        sub = sub + sub[-2:0:-1]
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

# --- Step 2: ワークスペース（完全オンデマンド処理） ---
if st.session_state.get('video_path') is not None:
    ROWS = st.session_state.get('ROWS', 3)
    COLS = st.session_state.get('COLS', 4)
    total_stamps = st.session_state.get('total_stamps', 12)
    total_original_frames = st.session_state.get('total_frames', 30)
    video_path = st.session_state['video_path']
    
    st.divider()
    st.header("🎛️ 超軽量・オンデマンドスタンプ編集ワークスペース")
    
    col_left, col_center, col_right = st.columns([1.1, 1.3, 1.1])
    
    with col_left:
        st.subheader("🎞️ アニメ ＆ 部分選択設定")
        
        # プルダウンまたは数値入力で選択
        stamp_num_val = st.selectbox("編集するスタンプ番号を選択", options=list(range(1, total_stamps + 1)), format_func=lambda x: f"スタンプ #{x}", key="stamp_selectbox")
        selected_stamp_idx = stamp_num_val - 1
        
        st.markdown("---")
        st.markdown(f"##### ✂️ 動画の部分選択（全 {total_original_frames} コマ）")
        
        if total_original_frames > 1:
            frame_range = st.slider(
                f"使用区間 (1 〜 {total_original_frames} コマ)",
                min_value=1, max_value=total_original_frames,
                value=(1, total_original_frames),
                key="frame_range_slider"
            )
        else:
            frame_range = (1, 1)

        st.markdown("##### ⏱️ LINE出力設定")
        default_target_count = min(20, max(5, total_original_frames))
        target_frame_count = st.slider(
            "LINE出力コマ数 (5〜20コマ)",
            min_value=5, max_value=20, value=default_target_count, step=1,
            key="target_count_slider"
        )

        ping_pong = st.checkbox("🔄 往復再生（ピンポン）", value=False, key="ping_pong_cb")
        trim_end = st.checkbox("✂️ ループ末尾カット", value=True, key="trim_end_cb")
        
        st.markdown("---")
        col_sec, col_loop = st.columns(2)
        with col_sec:
            target_sec = st.selectbox("総再生時間 (秒)", [1, 2, 3, 4], index=1, key="target_sec_sb")
        with col_loop:
            loop_count = st.selectbox("ループ回数", [1, 2, 3, 4], index=1, key="loop_count_sb")

        st.markdown("---")
        with st.expander("📐 グリッド微調整 (全体のズレ・個別枠拡大)", expanded=False):
            grid_offset_x = st.slider("全体を左右移動 (px)", min_value=-100, max_value=100, value=0, step=1, key="g_ox")
            grid_offset_y = st.slider("全体を上下移動 (px)", min_value=-100, max_value=100, value=0, step=1, key="g_oy")
            grid_expand = st.slider("切り出し枠外側拡大 (px)", min_value=0, max_value=100, value=10, step=2, key="g_exp")
            
            st.markdown(f"**スタンプ #{stamp_num_val} 個別微調整**")
            ind_ox = st.slider("個別左右移動", min_value=-50, max_value=50, value=0, step=1, key="ind_ox")
            ind_oy = st.slider("個別上下移動", min_value=-50, max_value=50, value=0, step=1, key="ind_oy")
            ind_exp = st.slider("個別枠拡大", min_value=-20, max_value=50, value=0, step=1, key="ind_exp")
            
        custom_offsets = {selected_stamp_idx: (ind_ox, ind_oy, ind_exp)}

    with col_right:
        st.subheader("✂️ マスク ＆ 画質設定")
        filter_noise = st.checkbox("🧹 見切れゴミ（Zzz等）自動除去", value=True, key="filter_noise_cb")
        sharp_edge = st.checkbox("🔪 輪郭をぼかさずクッキリ保つ (シャープ透過)", value=True, key="sharp_edge_cb")
        
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
        with st.expander(f"🖼️ 全 {total_stamps} カットの切り出し枠確認", expanded=True):
            try:
                cap_first = cv2.VideoCapture(video_path)
                ret_f, first_frame = cap_first.read()
                cap_first.release()
                if ret_f:
                    grid_preview_img = draw_grid_preview(
                        first_frame, ROWS, COLS,
                        offset_x=grid_offset_x, offset_y=grid_offset_y,
                        cell_expand=grid_expand, selected_idx=selected_stamp_idx,
                        custom_offsets=custom_offsets
                    )
                    st.image(grid_preview_img, caption="黄枠＝現在選択中のスタンプ", use_container_width=True)
            except Exception as e:
                st.error(f"ガイド表示エラー: {e}")

    # ★【超軽量オンデマンド処理】選択された「そのカットのその範囲」だけをピンポイントで読み込んで処理するから絶対に重くならない！
    try:
        r = selected_stamp_idx // COLS
        c = selected_stamp_idx % COLS
        
        # ユーザーがスライダーで選んだ区間（frame_range）のフレームだけをファイルから直接ロード
        stamp_raw_cells = load_specific_stamp_frames(
            video_path, frame_range[0], frame_range[1], r, c, ROWS, COLS,
            grid_offset_x, grid_offset_y, grid_expand, custom_offsets
        )
        
        cropped_cells = [crop_cell_margins(cell, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct) for cell in stamp_raw_cells]
        trans_frames = [remove_background_floodfill_sharp(cell, tolerance=tolerance, filter_noise=filter_noise, sharp_edge=sharp_edge) for cell in cropped_cells]
        centered_frames = trans_frames
        
        edited_frames = process_frame_sequence_strict(
            centered_frames,
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
                st.image(preview_gif, caption=f"スタンプ #{selected_stamp_idx + 1} | 区間: コマ#{frame_range[0]}〜#{frame_range[1]} ➔ 出力: {frame_cnt}コマ", use_container_width=True)
                
                single_apng_data = export_apng_lossless(edited_frames, durations_list, loop_count)
                st.download_button(
                    label=f"💾 スタンプ #{selected_stamp_idx + 1} を無劣化個別ダウンロード (APNG)",
                    data=single_apng_data,
                    file_name=f"stamp_{selected_stamp_idx+1:02d}.png",
                    mime="image/png",
                    key="single_dl_btn"
                )
                st.success(f"✨ **超軽量オンデマンド稼働中**: 全{frame_cnt}コマ / 1コマ **{frame_duration_ms}ms**")
            except Exception as e:
                st.error(f"プレビュー描画エラー: {e}")
                
    st.divider()
    
    # --- 一括書き出しもオンデマンドで安全に実行 ---
    st.subheader(f"📦 編集した設定で全 {total_stamps} 個のスタンプを一括書き出し")
    
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
                    
                    r = idx // COLS
                    c = idx % COLS
                    
                    stamp_cells = load_specific_stamp_frames(
                        video_path, frame_range[0], frame_range[1], r, c, ROWS, COLS,
                        grid_offset_x, grid_offset_y, grid_expand, custom_offsets
                    )
                    
                    c_cells = [crop_cell_margins(cell, crop_left_pct, crop_right_pct, crop_top_pct, crop_bottom_pct) for cell in stamp_cells]
                    trans_frames = [remove_background_floodfill_sharp(cell, tolerance=tolerance, filter_noise=filter_noise, sharp_edge=sharp_edge) for cell in c_cells]
                    cent_frames = trans_frames
                    
                    proc_f = process_frame_sequence_strict(
                        cent_frames,
                        target_frame_count=target_frame_count, ping_pong=ping_pong, trim_end=trim_end
                    )
                    
                    frame_cnt = len(proc_f)
                    base_ms = loop_ms // frame_cnt
                    remainder = loop_ms % frame_cnt
                    durations_list = [base_ms] * frame_cnt
                    for i in range(remainder):
                        durations_list[i] += 1
                        
                    apng_data = export_apng_lossless(proc_f, durations_list, loop_count)
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
            
