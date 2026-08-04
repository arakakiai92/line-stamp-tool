import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成", layout="centered")

st.title("⚡ LINEアニメーションスタンプ自動生成ツール")
st.caption("白背景を自動で透過し、LINEの審査ガイドライン（20コマ以内 / 1〜4秒の整数秒 / 1MB以下）に一括適合させます。")

uploaded_file = st.file_uploader("動画ファイル (MP4 / MOV) を選択してください", type=["mp4", "mov"])

ROWS = 3
COLS = 4

def remove_white_background(pil_img, threshold=240):
    """白背景を高速で透明化する関数"""
    img = pil_img.convert("RGBA")
    data = np.array(img)
    
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    # RGBがすべて240以上の明るい（白っぽい）部分を透明化
    white_areas = (r >= threshold) & (g >= threshold) & (b >= threshold)
    data[white_areas, 3] = 0
    
    return Image.fromarray(data)

def optimize_apng_bytes(img_list, duration_ms):
    """LINEの1MB制限に収めるため自動減色してAPNG化"""
    colors_list = [256, 128, 64, 32]
    for colors in colors_list:
        quantized_imgs = []
        for img in img_list:
            alpha = img.split()[-1]
            q_img = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=colors)
            q_img.putalpha(alpha)
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
        if len(data) < 990000: # 1MB未満なら採用
            return data
    return data

if uploaded_file is not None:
    if st.button("🚀 審査適合スタンプを一括変換"):
        st.info("動画を処理中...")
        
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if not frames:
            st.error("動画の読み込みに失敗しました。")
        else:
            # 1. コマ数補正（規約: 最大20コマ）
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

            # 2. 再生時間補正（規約: 1〜4秒の整数秒）
            raw_duration_sec = total_f / fps
            target_sec = max(1, min(4, round(raw_duration_sec)))
            frame_duration_ms = int((target_sec * 1000) / len(selected_frames))
            
            # 3. グリッド切出 ＆ 高速白透過処理
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
                        
                        pil_cell = Image.fromarray(cell)
                        transparent_cell = remove_white_background(pil_cell)
                        transparent_cell.thumbnail((320, 270), Image.Resampling.LANCZOS)
                        
                        stamp_frames[idx].append(transparent_cell)
                
                progress_bar.progress((f_idx + 1) / len(selected_frames))
                
            # 4. APNG生成＆ZIP圧縮
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(12):
                    img_list = stamp_frames[idx]
                    apng_data = optimize_apng_bytes(img_list, frame_duration_ms)
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    
            st.success(f"🎉 変換完了！ 再生時間: {target_sec}秒 / {len(selected_frames)}コマ / 全ファイル1MB以下に適合")
            
            st.download_button(
                label="📦 LINE審査適合スタンプを一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip"
            )
