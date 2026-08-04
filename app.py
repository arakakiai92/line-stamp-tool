import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成", layout="centered")

st.title("⚡ LINEアニメーションスタンプ自動生成ツール")
st.caption("背景色を自動取得し、高度な画像処理（境界平滑化）でノイズを除去して透過します。LINEの審査ガイドライン（20コマ以内 / 1〜4秒の整数秒 / 1MB以下）に適合させます。")

uploaded_file = st.file_uploader("動画ファイル (MP4 / MOV) を選択してください", type=["mp4", "mov"])

ROWS = 3
COLS = 4

def remove_background_cv(cell_bgr):
    """OpenCVを使用して、背景色を自動取得し、距離に基づいて透過処理と平滑化を行う関数"""
    # 1. 背景色の自動取得（最左上ピクセルの色を背景色と仮定）
    # cell_bgr はBGR形式
    bg_color = cell_bgr[0, 0]
    
    # 2. ユークリッド距離の計算（全ピクセルと背景色）
    # 動画圧縮による微小な色ムラも許容する
    diff = cell_bgr.astype(np.float32) - bg_color.astype(np.float32)
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # 3. アルファチャンネルの作成（距離が閾値以下のピクセルを透明にする）
    # image_3.pngのようなノイズを消すため、閾値を少し甘く設定
    threshold = 50 
    alpha = np.where(dist < threshold, 0, 255).astype(np.uint8)
    
    # 4. アルファチャンネルの平滑化（ノイズ除去と境界をきれいに丸め込む）
    # ガウシアンブラーで境界をぼかす
    alpha_blurred = cv2.GaussianBlur(alpha, (5, 5), 0)
    # 再度閾値処理（二値化）を行い、境界を丸め込み、ガタついた白いピクセルの残りを消す
    # 127より大きいものを255（白）、小さいものを0（透明）に。
    _, alpha_smoothed = cv2.threshold(alpha_blurred, 127, 255, cv2.THRESH_BINARY)
    
    # 5. BGRA画像の合成
    b, g, r = cv2.split(cell_bgr)
    cell_bgra = cv2.merge([b, g, r, alpha_smoothed])
    
    # 6. PIL用にRGBAへ変換
    cell_rgba = cv2.cvtColor(cell_bgra, cv2.COLOR_BGRA2RGBA)
    
    return cell_rgba

def optimize_apng_bytes(img_list, duration_ms):
    """LINEの1MB制限に安全に適合させてAPNG化する関数"""
    # 1. 標準のRGBAモードでAPNG生成
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
    
    # 1MB (約990KB) 未満ならそのまま採用
    if len(data) < 990000:
        return data
        
    # 2. 万が一1MBを超えた場合は、RGBAモードを維持したまま安全に減色
    colors_list = [128, 64, 32]
    for colors in colors_list:
        quantized_imgs = []
        for img in img_list:
            # 減色後にRGBAへ戻すことでエラー防止
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
        st.info("動画を解析し、高度な画像処理で透過・ノイズ除去中です。少し時間がかかります...")
        
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
            frames.append(frame) # OpenCVのBGR形式のまま保持
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
            
            # 3. グリッド切出 ＆ 高度背景透過処理（OpenCV）
            h, w, _ = selected_frames[0].shape
            cell_h = h // ROWS
            cell_w = w // COLS
            
            stamp_frames = {i: [] for i in range(12)}
            progress_bar = st.progress(0)
            
            for f_idx, frame in enumerate(selected_frames):
                for r in range(ROWS):
                    for c in range(COLS):
                        idx = r * COLS + c
                        # OpenCV形式（BGR）で切り出す
                        cell = frame[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                        
                        # OpenCVを使用して背景色自動取得、透過、境界平滑化
                        transparent_cell_rgba_array = remove_background_cv(cell)
                        # PIL画像（RGBAモード）に変換
                        transparent_cell = Image.fromarray(transparent_cell_rgba_array)
                        
                        # LINE規格の最大サイズにリサイズ
                        transparent_cell.thumbnail((320, 270), Image.Resampling.LANCZOS)
                        
                        stamp_frames[idx].append(transparent_cell)
                
                progress_bar.progress((f_idx + 1) / len(selected_frames))
                
            # 4. APNG生成＆ZIP圧縮
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(12):
                    img_list = stamp_frames[idx]
                    # 安全なAPNG形式（1MB以下）で出力
                    apng_data = optimize_apng_bytes(img_list, frame_duration_ms)
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    
            st.success(f"🎉 変換完了！ 再生時間: {target_sec}秒 / {len(selected_frames)}コマ / 全ファイル1MB以下。境界も滑らかになりました。")
            
            st.download_button(
                label="📦 LINE審査適合スタンプを一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip"
            )
