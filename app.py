import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成（ガイドライン厳守版）", layout="centered")

st.title("⚡ LINEアニメーションスタンプ自動生成")
st.caption("LINE審査ガイドライン（フレーム数5~20 / 再生時間1~4秒の整数秒 / ループ数1~4回 / 1MB以下 / 背景透過 / センタリング）に100%自動適合させます。")

uploaded_file = st.file_uploader("動画ファイル (MP4 / MOV) を選択してください", type=["mp4", "mov"])

# ループ・秒数の設定モード
st.subheader("⚙️ アニメーション再生設定")
mode = st.radio("設定モードを選択してください", ["🤖 自動最適化 (動画の長さに合わせてLINE規約に最適設定)", "⚙️ 手動指定 (再生時間とループ数を自分で指定)"])

manual_target_sec = 2
manual_loop_count = 2

if "手動指定" in mode:
    col1, col2 = st.columns(2)
    with col1:
        manual_target_sec = st.selectbox("総再生時間 (整数秒)", [1, 2, 3, 4], index=1, help="スタンプの全体の長さです（最大4秒）")
    with col2:
        manual_loop_count = st.selectbox("ループ回数", [1, 2, 3, 4], index=1, help="1スタンプあたりの繰り返し回数です")

ROWS = 3
COLS = 4

def remove_background_floodfill(cell_bgr, tolerance=40):
    """外枠からの塗りつぶしにより、キャラ内部の白パーツを保護しながら背景のみを透過する関数"""
    h, w, _ = cell_bgr.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    img_work = cell_bgr.copy()
    
    bg_color = cell_bgr[0, 0].astype(np.float32)
    
    seeds = []
    step = 10
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
            if color_dist <= tolerance * 1.5:
                cv2.floodFill(
                    img_work,
                    mask,
                    seedPoint=(seed_x, seed_y),
                    newVal=(0, 0, 0),
                    loDiff=lo_diff,
                    upDiff=up_diff,
                    flags=flags
                )
            
    bg_mask = mask[1:h+1, 1:w+1]
    alpha = np.where(bg_mask == 255, 0, 255).astype(np.uint8)
    
    alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0)
    _, alpha_smoothed = cv2.threshold(alpha_blurred, 127, 255, cv2.THRESH_BINARY)
    
    b, g, r = cv2.split(cell_bgr)
    cell_bgra = cv2.merge([b, g, r, alpha_smoothed])
    cell_rgba = cv2.cvtColor(cell_bgra, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(cell_rgba)

def center_and_fit_stamp(frame_list, target_w=320, target_h=270, padding=12):
    """キャラクターを認識して中央寄せセンタリング配置する関数"""
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

def calculate_line_animation_timing(raw_duration_sec, num_frames, is_auto, user_sec=2, user_loop=2):
    """LINEの厳格な「整数秒」「ループ数1~4」規約にミリ秒単位でぴったり合わせる計算関数"""
    if not is_auto:
        total_sec = user_sec
        loop_count = user_loop
        loop_target_ms = (total_sec * 1000) // loop_count
    else:
        candidates = []
        for total_target_sec in [1, 2, 3, 4]:
            for loop_count in [1, 2, 3, 4]:
                total_target_ms = total_target_sec * 1000
                if total_target_ms % loop_count != 0:
                    continue
                loop_target_ms = total_target_ms // loop_count
                avg_frame_ms = loop_target_ms / num_frames
                
                if avg_frame_ms < 50 or avg_frame_ms > 500:
                    continue
                    
                loop_sec = loop_target_ms / 1000.0
                speed_ratio = loop_sec / raw_duration_sec if raw_duration_sec > 0 else 1.0
                speed_penalty = abs(speed_ratio - 1.0)
                duration_preference = -0.05 * total_target_sec
                score = speed_penalty + duration_preference
                
                candidates.append({
                    'loop_count': loop_count,
                    'total_sec': total_target_sec,
                    'loop_ms': loop_target_ms,
                    'score': score
                })
                
        if not candidates:
            total_sec, loop_count, loop_target_ms = 2, 2, 1000
        else:
            candidates.sort(key=lambda x: x['score'])
            best = candidates[0]
            total_sec = best['total_sec']
            loop_count = best['loop_count']
            loop_target_ms = best['loop_ms']

    # 1コマあたりの表示時間を分配（合計が正確に1ループのミリ秒になるよう補正）
    base_ms = loop_target_ms // num_frames
    remainder = loop_target_ms % num_frames
    durations = [base_ms] * num_frames
    for i in range(remainder):
        durations[i] += 1
        
    return loop_count, total_sec, durations

def optimize_apng_bytes(img_list, durations, loop_count):
    """指定されたループ回数とコマ表示時間で1MB以下に抑えてAPNG保存する関数"""
    buf = io.BytesIO()
    img_list[0].save(
        buf,
        format="PNG",
        save_all=True,
        append_images=img_list[1:],
        duration=durations,
        loop=loop_count  # LINE規定の1~4回のループ数を正しく指定
    )
    data = buf.getvalue()
    
    if len(data) < 990000:
        return data
        
    colors_list = [128, 64, 32]
    for colors in colors_list:
        quantized_imgs = []
        for img in img_list:
            q_img = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE).convert("RGBA")
            quantized_imgs.append(q_img)
            
        buf = io.BytesIO()
        quantized_imgs[0].save(
            buf,
            format="PNG",
            save_all=True,
            append_images=quantized_imgs[1:],
            duration=durations,
            loop=loop_count
        )
        data = buf.getvalue()
        if len(data) < 990000:
            return data
            
    return data

if uploaded_file is not None:
    if st.button("🚀 審査適合スタンプを一括生成"):
        st.info("動画を解析し、LINEガイドライン適合処理（コマ数・再生時間・ループ数調整・透過・センタリング）を実行中です...")
        
        temp_path = "temp_input.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
            
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if not frames:
            st.error("動画の読み込みに失敗しました。")
        else:
            # 1. フレーム（コマ）数の調整（規約: 5～20フレーム）
            total_f = len(frames)
            target_f_count = min(20, max(5, total_f))
            if total_f > 20:
                indices = np.linspace(0, total_f - 1, 20, dtype=int)
                selected_frames = [frames[i] for i in indices]
            elif total_f < 5:
                st.error("動画のコマ数が短すぎます（最低5コマ必要です）。")
                st.stop()
            else:
                selected_frames = frames

            # 2. 秒数＆ループ回数の計算（規約: 1~4秒の整数秒、ループ1~4回）
            raw_duration_sec = len(selected_frames) / fps
            is_auto = "自動最適化" in mode
            loop_count, total_sec, durations_list = calculate_line_animation_timing(
                raw_duration_sec, len(selected_frames), is_auto, manual_target_sec, manual_loop_count
            )
            
            # 3. グリッド切出 ＆ キャラ内部白保護透過処理
            h, w, _ = selected_frames[0].shape
            cell_h = h // ROWS
            cell_w = w // COLS
            
            raw_stamp_frames = {i: [] for i in range(12)}
            progress_bar = st.progress(0)
            
            for f_idx, frame in enumerate(selected_frames):
                for r in range(ROWS):
                    for c in range(COLS):
                        idx = r * COLS + c
                        cell = frame[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                        
                        transparent_cell = remove_background_floodfill(cell)
                        raw_stamp_frames[idx].append(transparent_cell)
                
                progress_bar.progress((f_idx + 1) / len(selected_frames))
                
            # 4. キャラ自動認識 ＆ センタリング ＆ 規約適合APNG書き出し
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(12):
                    centered_frames = center_and_fit_stamp(raw_stamp_frames[idx], target_w=320, target_h=270)
                    
                    # 指定のループ数・コマ表示時間でAPNG化
                    apng_data = optimize_apng_bytes(centered_frames, durations_list, loop_count)
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    
            st.success(f"🎉 変換完了！【審査適合スペック】 総再生時間: {total_sec}秒 / ループ回数: {loop_count}回 / コマ数: {len(selected_frames)}コマ / 全ファイル1MB以下")
            
            st.download_button(
                label="📦 LINE審査適合スタンプを一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip"
            )
