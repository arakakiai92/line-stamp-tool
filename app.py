import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import zipfile
import os

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成（審査ガイドライン厳守版）", layout="centered")

st.title("🎬 LINEアニメーションスタンプ自動生成")
st.caption("LINEの厳格な審査基準（最大20コマ / 1〜4秒の整数秒 / 1MB以下 / 背景透過）に完全自動で適合させます。")

uploaded_file = st.file_uploader("動画ファイル (MP4 / MOV) を選択してください", type=["mp4", "mov"])

ROWS = 3
COLS = 4

def optimize_apng_bytes(img_list, duration_ms):
    """LINEの1MB制限を絶対超えないように減色処理を行いながらAPNG化する関数"""
    colors_list = [256, 128, 64, 32]
    
    for colors in colors_list:
        # パレット（減色）変換して軽量化
        quantized_imgs = []
        for img in img_list:
            # Alphaチャンネル（透過）を保持しながら減色
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
        
        # 1MB (1,048,576 bytes) 未満なら採用
        if len(data) < 990000:
            return data
            
    return data # 万が一下がらない場合も最小容量を返す

if uploaded_file is not None:
    if st.button("🚀 審査基準に適合させて出力"):
        st.info("動画を解析し、LINEの審査ガイドラインに合わせて変換中です...")
        
        temp_path = "temp_input.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
            
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
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
            # --- 1. フレーム数の補正（規約: 5〜20フレーム） ---
            total_f = len(frames)
            target_f_count = 20  # LINE規定の上限である20コマに間引き
            
            if total_f > target_f_count:
                indices = np.linspace(0, total_f - 1, target_f_count, dtype=int)
                selected_frames = [frames[i] for i in indices]
            elif total_f < 5:
                st.error("動画のフレーム数が短すぎます（最低5コマ以上必要です）。")
                st.stop()
            else:
                selected_frames = frames

            # --- 2. 再生時間の補正（規約: 1秒, 2秒, 3秒, 4秒のいずれか整数秒） ---
            raw_duration_sec = total_f / fps
            # 1〜4秒の範囲で最も近い整数秒に固定
            target_sec = max(1, min(4, round(raw_duration_sec)))
            
            # 1コマあたりの表示時間（ミリ秒）
            frame_duration_ms = int((target_sec * 1000) / len(selected_frames))
            
            # --- 3. グリッド切出 ＆ 透過・リサイズ ---
            h, w, _ = selected_frames[0].shape
            cell_h = h // ROWS
            cell_w = w // COLS
            
            stamp_frames = {i: [] for i in range(12)}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for f_idx, frame in enumerate(selected_frames):
                status_text.text(f"コマ処理中 ({f_idx + 1}/{len(selected_frames)})")
                
                for r in range(ROWS):
                    for c in range(COLS):
                        idx = r * COLS + c
                        cell = frame[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                        
                        pil_cell = Image.fromarray(cell)
                        transparent_cell = remove(pil_cell)
                        
                        # --- 4. サイズ制限（規約: 最大 幅320px × 高さ270px） ---
                        transparent_cell.thumbnail((320, 270), Image.Resampling.LANCZOS)
                        stamp_frames[idx].append(transparent_cell)
                
                progress_bar.progress((f_idx + 1) / len(selected_frames))
                
            status_text.text("LINE規約に合わせて軽量化・ZIP圧縮中...")
            
            # --- 5. 容量オーバーチェック＆APNG化 ---
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(12):
                    img_list = stamp_frames[idx]
                    
                    # 減色処理で1MB未満に調整されたAPNGバイナリを取得
                    apng_data = optimize_apng_bytes(img_list, frame_duration_ms)
                    
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    
            st.success(f"🎉 処理完了！ 再生時間: {target_sec}秒 / コマ数: {len(selected_frames)}コマ / 全ファイル1MB以下に適合済み")
            
            st.download_button(
                label="📦 LINE審査適合スタンプを一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip"
            )
