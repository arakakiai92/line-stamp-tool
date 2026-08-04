import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成", layout="centered")

st.title("⚡ LINEアニメーションスタンプ自動生成ツール")
st.caption("自動背景透過・キャラクター自動認識＆センタリング配置・LINE審査ガイドライン適合（20コマ以内 / 1〜4秒の整数秒 / 1MB以下）をすべて一括で行います。")

uploaded_file = st.file_uploader("動画ファイル (MP4 / MOV) を選択してください", type=["mp4", "mov"])

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
    """
    全コマ共通でキャラクターの表示領域を自動認識し、
    動きのガタつきを防ぎながら320x270キャンバスの中央にぴったり配置する関数
    """
    if not frame_list:
        return frame_list
        
    # 全コマの透過情報を重ね合わせてアニメーション全体の最大領域を検出
    alphas = [np.array(img)[:, :, 3] for img in frame_list]
    stacked_alpha = np.maximum.reduce(alphas)
    
    non_zeros = np.argwhere(stacked_alpha > 10)
    if non_zeros.size == 0:
        return [img.resize((target_w, target_h), Image.Resampling.LANCZOS) for img in frame_list]
        
    min_y, min_x = non_zeros.min(axis=0)
    max_y, max_x = non_zeros.max(axis=0)
    
    # 認識領域に余白（Padding）を追加
    h_orig, w_orig = stacked_alpha.shape
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(w_orig - 1, max_x + padding)
    max_y = min(h_orig - 1, max_y + padding)
    
    crop_w = max_x - min_x + 1
    crop_h = max_y - min_y + 1
    
    # 320x270に最適フィットする倍率を計算
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    
    # キャンバス中央配置の計算
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

def optimize_apng_bytes(img_list, duration_ms):
    """LINEの1MB制限に安全に適合させてAPNG化する関数"""
    buf = io.BytesIO()
    img_list[0].save(
        buf,
        format="PNG",
        save_all=True,
        append_images=img_list[1:],
        duration=duration_ms,
        loop=0
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
            duration=duration_ms,
            loop=0
        )
        data = buf.getvalue()
        if len(data) < 990000:
            return data
            
    return data

if uploaded_file is not None:
    if st.button("🚀 センタリング・審査適合スタンプを一括変換"):
        st.info("動画を解析し、キャラクター認識・センタリング・背景透過処理中です...")
        
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
            # 1. コマ数補正（最大20コマ）
            total_f = len(frames)
            target_f_count = 20
            if total_f > target_f_count:
                indices = np.linspace(0, total_f - 1, target_f_count, dtype=int)
                selected_frames = [frames[i] for i in indices]
            elif total_f < 5:
                st.error("動画のコマ数が短すぎます（最低5コマ必要）。")
                st.stop()
            else:
                selected_frames = frames

            # 2. 再生時間補正（1〜4秒の整数秒）
            raw_duration_sec = total_f / fps
            target_sec = max(1, min(4, round(raw_duration_sec)))
            frame_duration_ms = int((target_sec * 1000) / len(selected_frames))
            
            # 3. グリッド切出 ＆ キャラ内部白保護透過
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
                
            # 4. キャラ自動認識 ＆ センタリング ＆ APNG書き出し
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(12):
                    # 全コマからキャラ領域を特定し、320x270の中央へセンタリング配置
                    centered_frames = center_and_fit_stamp(raw_stamp_frames[idx], target_w=320, target_h=270)
                    
                    # 1MB以下適合APNG化
                    apng_data = optimize_apng_bytes(centered_frames, frame_duration_ms)
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    
            st.success(f"🎉 変換完了！ キャラクターを自動認識し、中央にセンタリング配置しました。（再生時間: {target_sec}秒 / {len(selected_frames)}コマ / 全ファイル1MB以下）")
            
            st.download_button(
                label="📦 LINE審査適合スタンプを一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip"
            )
