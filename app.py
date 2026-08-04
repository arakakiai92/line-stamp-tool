import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os

st.set_page_config(page_title="LINEアニメーションスタンプ自動生成＆高度編集ツール", layout="centered")

st.title("🎬 LINEアニメーションスタンプ自動生成 ＆ 高度編集ツール")
st.caption("動画解析後、プレビューを見ながら「コマ削除」「往復再生」「速度調整」を自由に行い、LINE審査ガイドライン適合APNGを一括出力できます。")

# セッション状態の初期化
if 'processed_stamps' not in st.session_state:
    st.session_state['processed_stamps'] = None
if 'fps' not in st.session_state:
    st.session_state['fps'] = 10

ROWS = 3
COLS = 4

def remove_background_floodfill(cell_bgr, tolerance=40):
    """キャラ内部の白を保護しながら背景のみ透過"""
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
                cv2.floodFill(img_work, mask, seedPoint=(seed_x, seed_y), newVal=(0, 0, 0),
                              loDiff=lo_diff, upDiff=up_diff, flags=flags)
            
    bg_mask = mask[1:h+1, 1:w+1]
    alpha = np.where(bg_mask == 255, 0, 255).astype(np.uint8)
    alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0)
    _, alpha_smoothed = cv2.threshold(alpha_blurred, 127, 255, cv2.THRESH_BINARY)
    
    b, g, r = cv2.split(cell_bgr)
    cell_bgra = cv2.merge([b, g, r, alpha_smoothed])
    return Image.fromarray(cv2.cvtColor(cell_bgra, cv2.COLOR_BGRA2RGBA))

def center_and_fit_stamp(frame_list, target_w=320, target_h=270, padding=12):
    """キャラを認識して中央寄せ配置"""
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

def process_frame_sequence(frames, start_frame, end_frame, ping_pong=False, trim_end=False):
    """コマの選択・トリミング・往復再生・重複削除処理"""
    sub = frames[start_frame - 1 : end_frame]
    if not sub:
        return frames
        
    if trim_end and len(sub) > 2:
        sub = sub[:-1]
        
    if ping_pong and len(sub) > 2:
        reverse_part = sub[-2:0:-1]
        sub = sub + reverse_part
        
    # LINE規定: 20コマ以内に収める
    if len(sub) > 20:
        indices = np.linspace(0, len(sub) - 1, 20, dtype=int)
        sub = [sub[i] for i in indices]
    elif len(sub) < 5:
        # 5コマ未満の場合は5コマまで補完
        while len(sub) < 5:
            sub.append(sub[-1])
            
    return sub

def create_preview_gif(frame_list, duration_ms):
    """プレビュー用Web GIF作成"""
    buf = io.BytesIO()
    frame_list[0].save(
        buf, format="GIF", save_all=True,
        append_images=frame_list[1:], duration=duration_ms, loop=0, disposal=2
    )
    return buf.getvalue()

def optimize_apng_bytes(img_list, durations, loop_count):
    """LINE 1MB以下適合APNG出力"""
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

# --- Step 1: 動画のアップロード ＆ 解析 ---
uploaded_file = st.file_uploader("1. 動画ファイル (MP4 / MOV) をアップロードしてください", type=["mp4", "mov"])

if uploaded_file is not None:
    if st.button("🔍 動画を解析して編集画面へ進む"):
        with st.spinner("動画のコマ分割・透過・センタリング処理中..."):
            temp_path = "temp_input.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
                
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 10
            st.session_state['fps'] = fps
            
            raw_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)
            cap.release()
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if raw_frames:
                h, w, _ = raw_frames[0].shape
                cell_h, cell_w = h // ROWS, w // COLS
                raw_stamps = {i: [] for i in range(12)}
                
                for frame in raw_frames:
                    for r in range(ROWS):
                        for c in range(COLS):
                            idx = r * COLS + c
                            cell = frame[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                            transparent_cell = remove_background_floodfill(cell)
                            raw_stamps[idx].append(transparent_cell)
                            
                # センタリング処理を適用
                centered_stamps = {}
                for idx in range(12):
                    centered_stamps[idx] = center_and_fit_stamp(raw_stamps[idx])
                    
                st.session_state['processed_stamps'] = centered_stamps
                st.success("解析が完了しました！下部の編集エリアで調整を行ってください。")

# --- Step 2: プレビュー ＆ 高度編集エリア ---
if st.session_state['processed_stamps'] is not None:
    st.divider()
    st.header("🎛️ スタンプ編集 ＆ リアルタイムプレビュー")
    
    stamps_data = st.session_state['processed_stamps']
    max_raw_frames = len(stamps_data[0])
    
    col_preview, col_controls = st.columns([1, 1.2])
    
    with col_controls:
        st.subheader("🛠️ 編集コントロール")
        
        # スタンプ選択
        selected_stamp_idx = st.number_input("確認・編集するスタンプ番号", min_value=1, max_value=12, value=1) - 1
        
        # コマ範囲（トリミング）選択
        frame_range = st.slider(
            "使用するコマ（フレーム）範囲の切り出し",
            min_value=1, max_value=max_raw_frames,
            value=(1, max_raw_frames),
            help="不要な最初の動きや最後の静止フレームをカットできます"
        )
        
        # ループオプション
        ping_pong = st.checkbox("🔄 往復再生（ピンポンループ）を有効にする", value=False, help="1➔2➔3➔2 のように動きを往復させて滑らかな無限ループを作成します")
        trim_end = st.checkbox("✂️ ループ時の最後の重複コマを1枚カットする", value=True, help="ループ直前のカクつき・一瞬の停止を防ぎます")
        
        # 時間・ループ数
        st.markdown("---")
        target_sec = st.selectbox("総再生時間 (LINE規定: 1~4秒の整数秒)", [1, 2, 3, 4], index=1)
        loop_count = st.selectbox("1スタンプあたりのループ回数", [1, 2, 3, 4], index=1)
        
        # コマ処理の適用
        current_raw_frames = stamps_data[selected_stamp_idx]
        edited_frames = process_frame_sequence(current_raw_frames, frame_range[0], frame_range[1], ping_pong, trim_end)
        
        # 1コマの表示時間計算
        total_ms = target_sec * 1000
        loop_ms = total_ms // loop_count
        frame_duration_ms = max(50, loop_ms // len(edited_frames))
        
        st.info(f"💡 現在の構成: 全 **{len(edited_frames)}コマ** / 1コマ当たり **{frame_duration_ms}ms** / **{loop_count}回再生**で計 **{target_sec}秒**")
        
    with col_preview:
        st.subheader("👁️ リアルタイムアニメーション")
        # プレビューGIF生成＆表示
        preview_gif = create_preview_gif(edited_frames, frame_duration_ms)
        st.image(preview_gif, caption=f"スタンプ #{selected_stamp_idx + 1} プレビュー", use_container_width=True)
        
    st.divider()
    
    # --- Step 3: 全スタンプ一括出力 ---
    st.subheader("📦 編集した設定で全12個のスタンプを一括書き出し")
    
    if st.button("🚀 LINE審査適合APNGを一括ダウンロード (ZIP)"):
        with st.spinner("12個のスタンプを設定に従って一括書き出し中..."):
            zip_buffer = io.BytesIO()
            
            # 各コマの表示時間リスト（ミリ秒誤差なし）
            base_ms = loop_ms // len(edited_frames)
            remainder = loop_ms % len(edited_frames)
            durations_list = [base_ms] * len(edited_frames)
            for i in range(remainder):
                durations_list[i] += 1
                
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx in range(12):
                    raw_f = stamps_data[idx]
                    proc_f = process_frame_sequence(raw_f, frame_range[0], frame_range[1], ping_pong, trim_end)
                    apng_data = optimize_apng_bytes(proc_f, durations_list, loop_count)
                    zip_file.writestr(f"stamp_{idx+1:02d}.png", apng_data)
                    
            st.success("🎉 全12個のアニメーションスタンプの出力が完了しました！")
            
            st.download_button(
                label="📦 LINE審査適合スタンプ一括ダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="line_animation_stamps.zip",
                mime="application/zip"
            )
