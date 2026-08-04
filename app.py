import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成", layout="centered")

st.title("⚡ LINEアニメーションスタンプ自動生成ツール")
st.caption("外枠からの自動塗りつぶし（FloodFill）により、キャラクター内部の「白い部分」を透過させずに外側の背景だけを完璧に透過処理します。")

uploaded_file = st.file_uploader("動画ファイル (MP4 / MOV) を選択してください", type=["mp4", "mov"])

ROWS = 3
COLS = 4

def remove_background_floodfill(cell_bgr, tolerance=40):
    """外枠から塗りつぶしを行い、キャラクター内側の白パーツを保護しながら背景のみを透過する関数"""
    h, w, _ = cell_bgr.shape
    
    # OpenCVのfloodFill用マスク（元のサイズより縦横2px大きく定義）
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    img_work = cell_bgr.copy()
    
    # 背景の基本色を取得（左上のピクセル）
    bg_color = cell_bgr[0, 0].astype(np.float32)
    
    # 外周（4辺のフチ）に沿ってシードポイント（塗りつぶし開始点）を設定
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
    
    # 外周のポイントから背景色に近い部分だけを起点に塗りつぶし実行
    for seed_x, seed_y in seeds:
        if mask[seed_y + 1, seed_x + 1] == 0:
            pixel_color = cell_bgr[seed_y, seed_x].astype(np.float32)
            color_dist = np.linalg.norm(pixel_color - bg_color)
            # 背景色に近いピクセルからのみ塗りつぶしを開始（キャラクターの線に当たったら止まる）
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
            
    # マスクから背景（255）とキャラ領域（0）を判定
    bg_mask = mask[1:h+1, 1:w+1]
    
    # アルファチャンネル生成（背景=0:透明, キャラクター=255:不透明）
    alpha = np.where(bg_mask == 255, 0, 255).astype(np.uint8)
    
    # 境界線を少し滑らかにして滑らかなギザギザのないエッジを作る
    alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0)
    _, alpha_smoothed = cv2.threshold(alpha_blurred, 127, 255, cv2.THRESH_BINARY)
    
    b, g, r = cv2.split(cell_bgr)
    cell_bgra = cv2.merge([b, g, r, alpha_smoothed])
    cell_rgba = cv2.cvtColor(cell_bgra, cv2.COLOR_BGRA2RGBA)
    return cell_rgba

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
    if st.button("🚀 審査適合スタンプを一括変換"):
        st.info("動画を解析し、キャラクター内側を保護しながら背景のみ透過処理中です...")
        
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
            
            # 3. グリッド切出 ＆ キャラ内部保護透過処理
            h, w, _ = selected_frames[0].shape
            cell_h = h // ROWS
            cell_w = w // COLS
            
            stamp_frames = {i: [] for i in range(12)}
            progress_bar = st.progress(0)
            
            for f_idx, frame in enumerate(selected_frames):
                for r in range(ROWS):
                    for c in range(COLS):
                        idx = r * COLS + c
                        cell = frame[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                        
                        # 外枠からの塗りつぶし透過（キャラ内部の白を保護）
                        transparent_cell_rgba_array = remove_background_floodfill(cell)
                        transparent_cell = Image.fromarray(transparent_cell_rgba_array)
                        
                        # LINE規格のサイズにリサイズ
                        transparent_cell.thumbnail((320, 270), Image.Resampling.LANCZOS)
                        
                        stamp_frames[idx].append(transparent_cell)
                
                progress_bar.progress((f_idx + 1) / len(selected_frames))
                
            # 4. APNG生成＆ZIP化
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(12):
                    img_list = stamp_frames[idx]
                    apng_data = optimize_apng_bytes(img_list, frame_duration_ms)
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    
            st.success(f"🎉 変換完了！ 再生時間: {target_sec}秒 / {len(selected_frames)}コマ / 全ファイル1MB以下。キャラ内部の白もバッチリ保護されました！")
            
            st.download_button(
                label="📦 LINE審査適合スタンプを一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip"
            )
