import os
import cv2
import numpy as np
import pandas as pd
import math

# ============================================================
# ================= CONFIGURATION & CONSTANTS ================
# ============================================================


# --- Visual Settings ---
CELL_W, CELL_H = 150, 300       # Dimension of image cells
TEXT_CELL_W = 130               # Width of text info columns
BORDER = 6                      # Thickness of cell borders
GLOBAL_FLIP_HORIZ = False       # True: Horizontally flip all images
GLOBAL_ROTATE_DEG = 0           # 0, 90, 180, 270  (clockwise rotation)
# --- Display Parameters ---
DISPLAY_ROWS = 5                # Number of vertical rows (Tracks) to show (Must be odd)
DISPLAY_GT_COLS = 10            # Number of horizontal columns (GT Images) to show

# --- Calculated Dimensions ---
HALF_WIN = DISPLAY_ROWS // 2    # How many rows above/below center to show
FINAL_W = CELL_W + BORDER * 2   
FINAL_H = CELL_H + BORDER * 2   
FINAL_TEXT_W = TEXT_CELL_W + BORDER * 2 

# --- Viewport / Animation ---
VIEWPORT_H = FINAL_H * DISPLAY_ROWS
ANIM_STEPS_PER_ROW = 6          
MAX_ANIM_FRAMES = 12            

# --- Color Definitions (BGR Format) ---
CLR_BLACK  = (0, 0, 0)
CLR_WHITE  = (255, 255, 255)
CLR_BLUE   = (255, 0, 0)      # Status: Manual Match (1)
CLR_ORANGE = (0, 165, 255)    # Status: Auto/Smart Fill (2)
CLR_RED    = (0, 0, 255)      # Status: Manual Ignore (3)
CLR_PINK   = (147, 20, 255)   # Status: Auto Ignore Remaining (4)
CLR_PURPLE = (200, 0, 200)    # UI: Mode indicators
CLR_GRAY   = (100, 100, 100)  # UI: Headers/Secondary text
CLR_GREEN  = (0, 255, 0)      # UI: Active Row Highlight
CLR_MARK_BG= (220, 180, 255)  # UI: Marked Row Background (Light Purple)


# ============================================================
# ====================== DATA HELPERS ========================
# ============================================================

def apply_global_transforms(img):
    """Applies global rotation and flipping to the image."""
    if img is None: return None
    
    # 1. Rotate
    if GLOBAL_ROTATE_DEG == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif GLOBAL_ROTATE_DEG == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif GLOBAL_ROTATE_DEG == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
    # 2. Flip
    if GLOBAL_FLIP_HORIZ:
        img = cv2.flip(img, 1) # 1 horizontal, 0 vertical, -1 dual
        
    return img

def clean_format(val):
    """Safely converts numeric IDs (floats) to integer strings."""
    if pd.isna(val) or val == "" or str(val).lower() in ["nan", "none"]:
        return None
    try:
        f_val = float(val)
        if f_val.is_integer():
            return str(int(f_val))
        return str(val)
    except (ValueError, TypeError):
        return str(val)

def get_color_by_flag(flag):
    """Returns the BGR color tuple corresponding to the status flag."""
    f = str(flag)
    if f == '1': return CLR_BLUE      # Manual
    elif f == '2': return CLR_ORANGE  # Interpolated
    elif f == '3': return CLR_RED     # Manual Ignore
    elif f == '4': return CLR_PINK    # Auto Ignore Remaining
    else: return CLR_WHITE

def extract_ignore_number(val):
    """Helper to find max ignore number from string 'ignore_123'"""
    if pd.isna(val): return 0
    s = str(val)
    if s.startswith("ignore_"):
        try:
            return int(s.split("_")[1])
        except:
            return 0
    return 0

def recalc_ignore_counter(track_df, gt_df, GT_ID_IDX):
    """Scans both dataframes to find the current maximum ignore_N."""
    max_ign_track = track_df["cowID"].apply(extract_ignore_number).max()
    if pd.isna(max_ign_track): max_ign_track = 0

    gt_id_col = gt_df.columns[GT_ID_IDX]
    max_ign_gt = gt_df[gt_id_col].apply(extract_ignore_number).max()
    if pd.isna(max_ign_gt): max_ign_gt = 0

    return max(max_ign_track, max_ign_gt) + 1

# ============================================================
# ===================== VISUALIZATION HELPERS ================
# ============================================================

def create_text_cell(text, width=TEXT_CELL_W, height=CELL_H, color_code=CLR_WHITE, is_marked=False):
    """
    Creates a text-only image cell.
    If color_code is not WHITE, fills with semi-transparent color.
    If is_marked is True, uses a special background color.
    """
    # 1. Base White Image
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 2. Determine Background Color
    bg_color = color_code
    alpha = 0.3
    
    # Priority: If Marked, override background to Light Purple
    # if is_marked:
    #     bg_color = CLR_MARK_BG
    #     alpha = 0.5 # Slightly stronger for mark
    
    # 3. Apply Semi-Transparent Background
    if bg_color != CLR_WHITE:
        overlay = np.full_like(img, bg_color)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 4. Draw Text
    y0, dy = 30, 25
    for i, line in enumerate(text.split("\n")):
        # Highlight "MARKED" text
        c = CLR_BLACK
        font_thick = 1
        if "[MARKED]" in line:
            c = CLR_PURPLE
            font_thick = 2
            
        cv2.putText(img, line, (5, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, font_thick, cv2.LINE_AA)
    
    # 5. Add Solid Colored Border (Status Color)
    img = cv2.copyMakeBorder(
        img, BORDER, BORDER, BORDER, BORDER,
        cv2.BORDER_CONSTANT, value=color_code
    )
    
    return cv2.resize(img, (FINAL_TEXT_W, FINAL_H), interpolation=cv2.INTER_NEAREST)

def prepare_display_img(img, text, border_color=CLR_WHITE, outer_border_color=None):
    if img is None:
        img = np.ones((CELL_H, CELL_W, 3), dtype=np.uint8) * 255
    
    img = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)

    if text:
        y0, dy = 25, 30
        for i, line in enumerate(text.split("\n")):
            cv2.putText(img, line, (10, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_RED, 1, cv2.LINE_AA)

    img = cv2.copyMakeBorder(
        img, BORDER, BORDER, BORDER, BORDER,
        cv2.BORDER_CONSTANT, value=border_color
    )

    if outer_border_color is not None:
        img = cv2.copyMakeBorder(
            img, 4, 4, 4, 4,  
            cv2.BORDER_CONSTANT, value=outer_border_color
        )

    return cv2.resize(img, (FINAL_W, FINAL_H), interpolation=cv2.INTER_NEAREST)

def build_sidebar(canvas_height, compact_mode, track_df, gt_df):
    clean_flags = track_df["flag"].apply(clean_format)
    total_tracks = len(track_df)
    auto_assigned = track_df["cowID"].notnull().sum()
    
    n_match  = (clean_flags == "1").sum()
    n_fill   = (clean_flags == "2").sum()
    n_ignore = (clean_flags == "3").sum()
    n_auto_ign = (clean_flags == "4").sum()
    
    # Count Marks
    n_marked = track_df["mark"].sum()

    total_gt = len(gt_df)
    gt_assigned = gt_df["assignedTrackID"].notnull().sum()
    gt_left = total_gt - gt_assigned

    mode_str = "COMPACT" if compact_mode else "STANDARD"

    content = [
        (f"COW MATCHER", CLR_BLACK),
        (f"View: {mode_str}", CLR_PURPLE),
        ("---------------------", CLR_GRAY),
        (f"TRACKS: {auto_assigned}/{total_tracks}", CLR_BLACK),
        (f"Marks: {n_marked}", CLR_PURPLE),
        (f" > Manual Match: {n_match}", CLR_BLUE),
        (f" > Auto Match: {n_fill}", CLR_ORANGE),
        (f" > Manual Ign: {n_ignore}", CLR_RED),
        (f" > Auto Ign: {n_auto_ign}", CLR_PINK),
        (" ", CLR_BLACK),
        (f"GT DATA: {gt_assigned}/{total_gt}", CLR_BLACK),
        (f"Left: {gt_left}", CLR_BLACK),
        ("---------------------", CLR_GRAY),
        ("ACTIONS:", CLR_GRAY),
        (" 1: Match", CLR_BLUE),
        (" 2: Auto Match&Ign / Clear", CLR_ORANGE),
        (" 3: Ignore", CLR_RED),
        (" r: Mark4Later/Unmark", CLR_PURPLE),
        (" x: Clear", CLR_BLACK),
        (" ", CLR_BLACK),
        ("NAVIGATION:", CLR_GRAY),
        (" w/s: Track    Up/Down", CLR_BLACK),
        (" a/d: GT       Up/Down", CLR_BLACK),
        (" i/k:  Tendem  Up/Down", CLR_BLACK),
        (" t: -> Next Mark", CLR_PURPLE),
        (" o: -> Next AutoIgn", CLR_PINK),
        (" p: -> Pred", CLR_BLACK),
        (" f: Find CowID", CLR_BLACK),
        (" g: Find Track", CLR_BLACK),
        (" q: Save & Quit", CLR_BLACK),
        ("---------------------", CLR_GRAY),
        ("Version: 1.11", CLR_GRAY),
    ]

    width = 230 
    text_img = np.ones((canvas_height, width, 3), dtype=np.uint8) * 255 
    margin_x, margin_y, line_h = 10, 30, 25

    for i, (line_text, color) in enumerate(content):
        y = margin_y + i * line_h
        if y >= canvas_height - 10: break
        
        is_header = any(x in line_text for x in ["MATCHER", "TRACKS", "GT DATA"])
        thickness = 2 if is_header else 1
        font_scale = 0.55 if is_header else 0.5

        cv2.putText(text_img, line_text, (margin_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return text_img

# ============================================================
# ================ SMOOTH SCROLL RENDER ENGINE ==============
# ============================================================

def get_row_cells(idx, is_track, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX):
    if is_track:
        if 0 <= idx < len(track_df):
            row = track_df.iloc[idx]
            mid = clean_format(row["cowID"])
            pid = clean_format(row["pred_cow_id"])
            flag = clean_format(row["flag"])
            color_code = get_color_by_flag(flag)
            
            # Check Mark Status
            is_marked = (row["mark"] == 1)
            mark_str = "\n[MARKED]" if is_marked else ""

            status_str = f"ID: {mid}" if mid else "Unassigned"
            pred_str = f"Pred: {pid}" if pid else "Pred: NaN"
            info_txt = f"Idx: {idx+1}\nTrk: {row['track_id']}\n{pred_str}\n{status_str}{mark_str}"

            p = os.path.join(track_img_dir, row["filename"])
            raw_img = cv2.imread(p) if os.path.exists(p) else None
            img = apply_global_transforms(raw_img)
        else:
            info_txt, img, flag, color_code, is_marked = "No Data", None, None, CLR_WHITE, False

        cells = [
            create_text_cell(info_txt, color_code=color_code, is_marked=is_marked),
            prepare_display_img(img, "", border_color=CLR_WHITE)
        ]
        return cells
    else:
        if 0 <= idx < len(gt_df):
            cid = clean_format(gt_df.iloc[idx, GT_ID_IDX])
            assigned_trk = clean_format(gt_df.iloc[idx]["assignedTrackID"])
            g_flag = clean_format(gt_df.iloc[idx]["flag"])
            color_code = get_color_by_flag(g_flag)

            assign_str = f"Trk: {assigned_trk}" if assigned_trk else "Unassigned"
            info_txt = f"GT: {idx+1}\nCow: {cid}\n\n{assign_str}"

            imgs = []
            for folder in gt_subfolders:
                path = os.path.join(folder, f"{cid}.jpg")
                raw_img = cv2.imread(path) if os.path.exists(path) else None
                imgs.append(apply_global_transforms(raw_img))
            overlays = gt_date_overlays
        else:
            info_txt, imgs, g_flag, overlays, color_code = "No Data", [], None, [], CLR_WHITE

        if compact_mode:
            valid = [(im, txt) for im, txt in zip(imgs, overlays) if im is not None]
            if valid:
                curr_imgs, curr_txts = list(zip(*valid))
            else:
                curr_imgs, curr_txts = [], []
        else:
            curr_imgs, curr_txts = imgs, overlays

        curr_imgs = list(curr_imgs)[:DISPLAY_GT_COLS]
        curr_txts = list(curr_txts)[:DISPLAY_GT_COLS]
        needed = DISPLAY_GT_COLS - len(curr_imgs)
        if needed > 0:
            curr_imgs += [None] * needed
            curr_txts += [""] * needed

        cells = [create_text_cell(info_txt, color_code=color_code)]
        for img, txt in zip(curr_imgs, curr_txts):
            cells.append(prepare_display_img(img, txt, CLR_WHITE))
        return cells

def generate_scrollable_strip(center_idx_float, is_track, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX):
    center_int = int(math.floor(center_idx_float))
    pixel_offset = int((center_idx_float - center_int) * FINAL_H)

    start_row = center_int - HALF_WIN - 1
    end_row = center_int + HALF_WIN + 1

    strip_rows = []

    for r_idx in range(start_row, end_row + 1):
        cells = get_row_cells(r_idx, is_track, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX)
        if not cells:
            w = FINAL_TEXT_W + FINAL_W if is_track else FINAL_TEXT_W + (FINAL_W * DISPLAY_GT_COLS)
            strip_rows.append(np.zeros((FINAL_H, w, 3), dtype=np.uint8))
        else:
            strip_rows.append(np.hstack(cells))

    full_strip = np.vstack(strip_rows)
    crop_y_start = FINAL_H + pixel_offset

    if crop_y_start < 0: crop_y_start = 0
    if crop_y_start + VIEWPORT_H > full_strip.shape[0]:
        crop_y_start = max(0, full_strip.shape[0] - VIEWPORT_H)

    return full_strip[crop_y_start : crop_y_start + VIEWPORT_H, :, :]

def draw_centered_overlay(img, x_offset, total_width):
    y_start = HALF_WIN * FINAL_H
    pad = 2
    pt1 = (x_offset + pad, y_start + pad)
    pt2 = (x_offset + total_width - pad, y_start + FINAL_H - pad)
    cv2.rectangle(img, pt1, pt2, CLR_GREEN, 2)

def render_scene(t_idx_float, g_idx_float, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX):
    track_img = generate_scrollable_strip(t_idx_float, True, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX)
    gt_img = generate_scrollable_strip(g_idx_float, False, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX)

    main_view = np.hstack([track_img, gt_img])

    track_w = FINAL_TEXT_W + FINAL_W
    draw_centered_overlay(main_view, 0, track_w)
    gt_w = FINAL_TEXT_W + (FINAL_W * DISPLAY_GT_COLS)
    draw_centered_overlay(main_view, track_w, gt_w)

    sidebar = build_sidebar(main_view.shape[0], compact_mode, track_df, gt_df)
    return np.hstack([sidebar, main_view])

def smooth_scroll(start_t, end_t, start_g, end_g, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX):
    diff_t = end_t - start_t
    diff_g = end_g - start_g
    if diff_t == 0 and diff_g == 0: return

    dist = max(abs(diff_t), abs(diff_g))
    frames = min(int(dist * ANIM_STEPS_PER_ROW), MAX_ANIM_FRAMES)
    frames = max(frames, 4)

    for i in range(1, frames + 1):
        alpha = i / frames
        ease = 1 - pow(1 - alpha, 3) 
        cur_t = start_t + (diff_t * ease)
        cur_g = start_g + (diff_g * ease)
        img = render_scene(cur_t, cur_g, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX)
        cv2.imshow("Cow Matcher", img)
        cv2.waitKey(1)