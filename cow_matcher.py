import os
import cv2
import re
import numpy as np
import pandas as pd
import argparse
import sys
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

def recalc_ignore_counter(track_df, gt_df):
    """Scans both dataframes to find the current maximum ignore_N."""
    max_ign_track = track_df["assignedCowID"].apply(extract_ignore_number).max()
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
    auto_assigned = track_df["assignedCowID"].notnull().sum()
    
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

def get_row_cells(idx, is_track, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays):
    if is_track:
        if 0 <= idx < len(track_df):
            row = track_df.iloc[idx]
            mid = clean_format(row["assignedCowID"])
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

def generate_scrollable_strip(center_idx_float, is_track, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays):
    center_int = int(math.floor(center_idx_float))
    pixel_offset = int((center_idx_float - center_int) * FINAL_H)

    start_row = center_int - HALF_WIN - 1
    end_row = center_int + HALF_WIN + 1

    strip_rows = []

    for r_idx in range(start_row, end_row + 1):
        cells = get_row_cells(r_idx, is_track, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays)
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

def render_scene(t_idx_float, g_idx_float, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays):
    track_img = generate_scrollable_strip(t_idx_float, True, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays)
    gt_img = generate_scrollable_strip(g_idx_float, False, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays)

    main_view = np.hstack([track_img, gt_img])

    track_w = FINAL_TEXT_W + FINAL_W
    draw_centered_overlay(main_view, 0, track_w)
    gt_w = FINAL_TEXT_W + (FINAL_W * DISPLAY_GT_COLS)
    draw_centered_overlay(main_view, track_w, gt_w)

    sidebar = build_sidebar(main_view.shape[0], compact_mode, track_df, gt_df)
    return np.hstack([sidebar, main_view])

def smooth_scroll(start_t, end_t, start_g, end_g, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays):
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
        img = render_scene(cur_t, cur_g, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays)
        cv2.imshow("Cow Matcher", img)
        cv2.waitKey(1)

# ============================================================
# ======================== SETUP & ARGS ======================
# ============================================================

parser = argparse.ArgumentParser(description="Manual Cow Matching Tool")
parser.add_argument("--track", required=True, help="Folder containing track images")
parser.add_argument("--gt", required=True, help="Path to the Ground Truth CSV")
parser.add_argument("--bank", default="./Bank/", help="Path to bank folder (default: ./Bank/)")
args = parser.parse_args()

# --- Path Configuration ---
save_folder = "./Progress/"
os.makedirs(save_folder, exist_ok=True)

track_img_dir = args.track
gt_root_dir = args.bank
if not os.path.exists(gt_root_dir):
    print(f"[Warning] GT Root directory '{gt_root_dir}' does not exist.")

gt_subfolders = sorted(
    [os.path.join(gt_root_dir, d) for d in os.listdir(gt_root_dir) if os.path.isdir(os.path.join(gt_root_dir, d))],
    reverse=True
)
gt_date_overlays = [os.path.basename(f)[-10:] for f in gt_subfolders]

track_save_csv = os.path.join(save_folder, os.path.splitext(os.path.basename(args.gt))[0] + "_corrected.csv")
gt_save_csv = os.path.join(save_folder, os.path.splitext(os.path.basename(args.gt))[0] + "_assigned.csv")

# ============================================================
# ===================== LOAD & MERGE DATA ====================
# ============================================================

# 1. Scan Track Images
print("[Info] Scanning Image Directory...")
if not os.path.isdir(track_img_dir): sys.exit(f"[Error] Dir not found: {track_img_dir}")

pattern = re.compile(r"trackId_(\d+)(?:_pred_(\d+))?\.jpg")
files_data = []

for f in sorted(os.listdir(track_img_dir)):
    match = pattern.match(f)
    if match:
        tid = int(match.group(1))
        pid = match.group(2) 
        files_data.append({'track_id': tid, 'filename': f, 'pred_cow_id': pid})

if not files_data: sys.exit("[Error] No matching images found in directory.")
base_df = pd.DataFrame(files_data)
base_df.sort_values("track_id", inplace=True)
base_df.reset_index(drop=True, inplace=True)

# 2. Load/Resume Track Data
if os.path.exists(track_save_csv):
    print(f"[Info] Resuming Track from: {track_save_csv}")
    saved_df = pd.read_csv(track_save_csv)
    if "track_id" in saved_df.columns:
        saved_df["track_id"] = saved_df["track_id"].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
    
    # Merge existing columns
    cols_to_merge = [c for c in ["track_id", "assignedCowID", "flag", "mark"] if c in saved_df.columns]
    track_df = pd.merge(base_df, saved_df[cols_to_merge], on="track_id", how="left")
else:
    print("[Info] Starting fresh Track log...")
    track_df = base_df
    track_df["assignedCowID"] = None
    track_df["flag"] = None
    track_df["mark"] = 0

# --- Mark Initialization Logic ---
if "mark" not in track_df.columns:
    track_df["mark"] = 0

# Ensure mark is integer type
track_df["mark"] = track_df["mark"].fillna(0).astype(int)

# Force Start and End to be Marked (1)
if len(track_df) > 0:
    track_df.at[0, "mark"] = 1
    track_df.at[len(track_df)-1, "mark"] = 1

# Generate Initial Marked List
marked_list = sorted(track_df.index[track_df["mark"] == 1].tolist())

# 3. Load/Resume GT Data
if os.path.exists(gt_save_csv):
    print(f"[Info] Resuming GT from: {gt_save_csv}")
    gt_df = pd.read_csv(gt_save_csv)
    rename_map = {"assigned_trackID": "assignedTrackID", "assigned_flag": "flag"}
    gt_df.rename(columns=rename_map, inplace=True)
    if "assigned" in gt_df.columns: gt_df.drop(columns=["assigned"], inplace=True)
else:
    print("[Info] Loading fresh GT CSV...")
    gt_df = pd.read_csv(args.gt)

if "CowID" in gt_df.columns:
    GT_ID_IDX = gt_df.columns.get_loc("CowID")
    print(f"[Info] Found 'CowID' column at Index {GT_ID_IDX}")
else:
    GT_ID_IDX = 1
    print(f"[Warning] 'CowID' not found. Defaulting to Index 1.")

if "assignedTrackID" not in gt_df.columns: gt_df["assignedTrackID"] = None
if "flag" not in gt_df.columns: gt_df["flag"] = None

# ============================================================
# ========== CALCULATE GLOBAL IGNORE COUNTER =================
# ============================================================

ignore_counter = recalc_ignore_counter(track_df, gt_df)
print(f"[Info] Global Ignore Counter initialized at: {ignore_counter}")

# ============================================================
# ============= FILL MISSING GT IDS ==========================
# ============================================================

gt_id_col = gt_df.columns[GT_ID_IDX]
missing_mask = gt_df[gt_id_col].isna() | (gt_df[gt_id_col].astype(str).str.strip() == "") | (gt_df[gt_id_col].astype(str).str.lower() == "nan")

if missing_mask.any():
    count = missing_mask.sum()
    print(f"[Info] Found {count} GT rows with missing Cow ID. Filling with 'ignore_miss_{ignore_counter}...'")
    for idx in gt_df.index[missing_mask]:
        gt_df.at[idx, gt_id_col] = f"ignore_miss_{ignore_counter}"
        ignore_counter += 1
    gt_df.to_csv(gt_save_csv, index=False)

# Ensure correct data types
track_df["assignedCowID"] = track_df["assignedCowID"].astype("object")
track_df["flag"] = track_df["flag"].astype("object")
gt_df["assignedTrackID"] = gt_df["assignedTrackID"].astype("object")
gt_df["flag"] = gt_df["flag"].astype("object")

# ============================================================
# ==================== SET START POSITIONS ===================
# ============================================================

total_tracks = len(track_df)
total_gt = len(gt_df)

last_auto = track_df["assignedCowID"].last_valid_index()
cur_track_idx = last_auto + 1 if last_auto is not None else 0

last_gt = gt_df["assignedTrackID"].last_valid_index()
cur_gt_index = last_gt + 1 if last_gt is not None else 0

cur_track_idx = min(cur_track_idx, total_tracks - 1)
cur_gt_index = min(cur_gt_index, total_gt - 1)

print(f"[Info] Ready. Start at Track Row {cur_track_idx+1}, GT Row {cur_gt_index+1}")

# ============================================================
# ======================== MAIN LOOP =========================
# ============================================================

compact_mode = False  

img0 = render_scene(cur_track_idx, cur_gt_index, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays)
cv2.imshow("Cow Matcher", img0)
cv2.waitKey(1)

while True:
    img = render_scene(cur_track_idx, cur_gt_index, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays)
    cv2.imshow("Cow Matcher", img)
    key = cv2.waitKey(0)

    prev_track = cur_track_idx
    prev_gt = cur_gt_index

    current_track_row = track_df.iloc[cur_track_idx]
    track_id = int(float(current_track_row["track_id"]))
    
    if cur_gt_index < total_gt:
        cow_id_raw = gt_df.iloc[cur_gt_index, GT_ID_IDX]
        cow_id = clean_format(cow_id_raw) 
    else:
        cow_id = None

    # === Key: 1 (MANUAL MATCH) ===
    if key == ord('1'):
        if cur_gt_index >= total_gt: continue
        
        gt_assigned = clean_format(gt_df.at[cur_gt_index, "assignedTrackID"])
        track_assigned = clean_format(track_df.at[cur_track_idx, "assignedCowID"])
        
        if gt_assigned:
            print(f"[BLOCK] GT Cow {cow_id} already assigned to Track {gt_assigned}. Cancel ('x') first.")
            continue 
        if track_assigned:
            print(f"[BLOCK] Current Track {track_id} already assigned to {track_assigned}. Cancel ('x') first.")
            continue

        track_df.at[cur_track_idx, "assignedCowID"] = cow_id
        track_df.at[cur_track_idx, "flag"] = "1"
        gt_df.at[cur_gt_index, "assignedTrackID"] = track_id
        gt_df.at[cur_gt_index, "flag"] = "1" 
        
        next_gt = cur_gt_index + 1
        while next_gt < total_gt and pd.notna(gt_df.iloc[next_gt]["assignedTrackID"]):
            next_gt += 1
        if next_gt < total_gt: cur_gt_index = next_gt
        
        cur_track_idx = min(cur_track_idx + 1, total_tracks - 1)

    # === Key: 2 (AUTO FILL & REMAINING FILL) ===
    elif key == ord('2'):
        has_auto = track_df["flag"].apply(clean_format).isin(["2", "4", "2.0", "4.0"]).any() or gt_df["flag"].apply(clean_format).isin(["2", "4", "2.0", "4.0"]).any()
        
        if has_auto:
            print("\n[Auto] Toggling OFF: Clearing all Auto-Fill (2) and Remaining-Ignore (4)...")
            gt_mask = gt_df["flag"].apply(clean_format).isin(["2", "4", "2.0", "4.0"])
            gt_df.loc[gt_mask, "assignedTrackID"] = None
            gt_df.loc[gt_mask, "flag"] = None
            
            track_mask = track_df["flag"].apply(clean_format).isin(["2", "4", "2.0", "4.0"])
            track_df.loc[track_mask, "assignedCowID"] = None
            track_df.loc[track_mask, "flag"] = None
            print("[Auto] Cleared.")
        else:
            print("\n[Auto] Toggling ON: Running Smart Fill + Ignore Remaining...")
            
            # Recalc Ignore Counter BEFORE filling
            ignore_counter = recalc_ignore_counter(track_df, gt_df)
            print(f"   -> [Info] Ignore Counter reset to: {ignore_counter}")
            
            # Phase A: Smart Fill
            is_one = track_df["flag"].apply(lambda x: clean_format(x) == "1")
            ones_idx = track_df.index[is_one].tolist()
            filled_count = 0
            
            if len(ones_idx) >= 2:
                for i in range(len(ones_idx) - 1):
                    s, e = ones_idx[i], ones_idx[i + 1]
                    track_range_idx = range(s+1, e)
                    targets = [x for x in track_range_idx if pd.isna(track_df.at[x, "flag"])]
                    if not targets: continue
                    
                    c_start = clean_format(track_df.at[s,"assignedCowID"])
                    c_end = clean_format(track_df.at[e,"assignedCowID"])
                    
                    gt_str_ids = gt_df.iloc[:, GT_ID_IDX].apply(clean_format)
                    g_s_list = gt_df.index[gt_str_ids == c_start].tolist()
                    g_e_list = gt_df.index[gt_str_ids == c_end].tolist()
                    
                    if not g_s_list or not g_e_list: continue
                    
                    g_s, g_e = g_s_list[0], g_e_list[0]
                    gt_gap_len = g_e - g_s - 1
                    track_gap_len = len(targets)

                    if gt_gap_len == track_gap_len:
                        conflict = False
                        for k in range(len(targets)):
                            if pd.notna(gt_df.at[g_s + 1 + k, "assignedTrackID"]):
                                conflict = True
                                break
                        if not conflict:
                            for k, a_idx in enumerate(targets):
                                gt_idx = g_s + 1 + k
                                gt_cid = clean_format(gt_df.iloc[gt_idx, GT_ID_IDX])
                                a_tid  = track_df.iloc[a_idx]["track_id"]
                                
                                track_df.at[a_idx, "assignedCowID"] = gt_cid
                                track_df.at[a_idx, "flag"] = "2"
                                gt_df.at[gt_idx, "assignedTrackID"] = a_tid
                                gt_df.at[gt_idx, "flag"] = "2"
                                filled_count += 1
            print(f"   -> Interpolated {filled_count} rows (Orange).")

            # Phase B: Fill Remaining with Ignore
            empty_mask = track_df["assignedCowID"].isna() | (track_df["assignedCowID"] == "")
            empty_indices = track_df.index[empty_mask].tolist()
            
            ignore_filled_count = 0
            for idx in empty_indices:
                ign_val = f"ignore_{ignore_counter}"

                track_df.at[idx, "assignedCowID"] = ign_val
                track_df.at[idx, "flag"] = "4"

                curr_track_id = track_df.at[idx, "track_id"]
                
                ignore_counter += 1
                ignore_filled_count += 1
            gt_empty_mask = gt_df["assignedTrackID"].isna() | (gt_df["assignedTrackID"] == "")
            gt_empty_indices = gt_df.index[gt_empty_mask].tolist()
            
            gt_ignore_count = 0
            for g_idx in gt_empty_indices:
                gt_df.at[g_idx, "assignedTrackID"] = f"ignore_{ignore_counter}" 
                gt_df.at[g_idx, "flag"] = "4"
                
                ignore_counter += 1
                gt_ignore_count += 1
            
            print(f"   -> Track: Filled remaining {ignore_filled_count} rows with Ignore (Pink).")
            print(f"   -> GT:    Filled remaining {gt_ignore_count} rows with Ignore (Pink).")
            print("[Auto] Complete. Press '2' again to Undo.")

    # === Key: 3 (MANUAL IGNORE) ===
    elif key == ord('3'):
        if pd.notna(track_df.at[cur_track_idx, "assignedCowID"]):
             print(f"[BLOCK] Track already assigned. Cancel first.")
        else:
            ignore_counter = recalc_ignore_counter(track_df, gt_df)
            
            track_df.at[cur_track_idx, "assignedCowID"] = f"ignore_{ignore_counter}"
            track_df.at[cur_track_idx, "flag"] = "3"
            
            ignore_counter += 1
            cur_track_idx = min(cur_track_idx + 1, total_tracks - 1)

    # === Key: r (MARK / UNMARK) ===
    elif key in (ord('r'), ord('R')):
        if cur_track_idx == 0 or cur_track_idx == total_tracks - 1:
            print("[Block] Cannot change Mark for Start/End rows.")
        else:
            current_val = track_df.at[cur_track_idx, "mark"]
            new_val = 1 if current_val == 0 else 0
            track_df.at[cur_track_idx, "mark"] = new_val
            print(f"[Mark] Row {cur_track_idx+1} set to {new_val}.")
            
            # Refresh List
            marked_list = sorted(track_df.index[track_df["mark"] == 1].tolist())

    # === Key: t (JUMP TO NEXT MARK) ===
    elif key in (ord('t'), ord('T')):
        # Refresh list just in case
        marked_list = sorted(track_df.index[track_df["mark"] == 1].tolist())
        
        # Find next index > current
        next_indices = [i for i in marked_list if i > cur_track_idx]
        
        if next_indices:
            cur_track_idx = next_indices[0]
            print(f"[Jump] To next Mark at {cur_track_idx + 1}")
        else:
            # Wrap around to start
            if marked_list:
                cur_track_idx = marked_list[0]
                print(f"[Jump] Wrapped to first Mark at {cur_track_idx + 1}")

    # === Navigation ===
    elif key in (ord('d'), 40, 1): cur_gt_index = min(cur_gt_index + 1, total_gt - 1)
    elif key in (ord('a'), 38, 0): cur_gt_index = max(cur_gt_index - 1, 0)
    elif key in (ord('s'), 39, 3): cur_track_idx = min(cur_track_idx + 1, total_tracks - 1)
    elif key in (ord('w'), 37, 2): cur_track_idx = max(0, cur_track_idx - 1)

    # Tandem navigation - activated with caps lock
    elif key in (ord('W'), ord('A'), ord('i')):
        cur_track_idx = max(0, cur_track_idx - 1)
        cur_gt_index = max(cur_gt_index - 1, 0)
    elif key in (ord('S'), ord('D'), ord('k')):
        cur_track_idx = min(cur_track_idx + 1, total_tracks - 1)
        cur_gt_index = min(cur_gt_index + 1, total_gt - 1)

    # === Key: x (CANCEL / CLEAR) ===
    elif key in (ord('x'), ord('X')):
        assigned_cow = clean_format(track_df.at[cur_track_idx, "assignedCowID"])
        flag_val = str(clean_format(track_df.at[cur_track_idx, "flag"]))
        
        if flag_val in ("1", "2") and assigned_cow and not str(assigned_cow).startswith("ignore"):
            gt_str_ids = gt_df.iloc[:, GT_ID_IDX].apply(clean_format)
            matches = gt_df.index[gt_str_ids == assigned_cow].tolist()
            if matches:
                gt_idx = matches[0]
                if str(clean_format(gt_df.at[gt_idx, "assignedTrackID"])) == str(track_id):
                    gt_df.at[gt_idx, "assignedTrackID"] = None
                    gt_df.at[gt_idx, "flag"] = None
                    print(f"[Info] Unassigned GT CowID {assigned_cow} from Track {track_id}.")

        track_df.at[cur_track_idx, "assignedCowID"] = None
        track_df.at[cur_track_idx, "flag"] = None

    # === Tools ===
    elif key in (ord('m'), ord('M')):
        compact_mode = not compact_mode
        print(f"[Info] Compact Mode: {'ON' if compact_mode else 'OFF'}")

    elif key in (ord('p'), ord('P')):
        pid = clean_format(track_df.iloc[cur_track_idx]["pred_cow_id"])
        if pid:
            gt_str_ids = gt_df.iloc[:, GT_ID_IDX].apply(clean_format)
            matches = gt_df.index[gt_str_ids == pid].tolist()
            if matches:
                cur_gt_index = matches[0]
                print(f"[Info] Jumped to GT Index {cur_gt_index+1} for PredID {pid}")
            else:
                print(f"[Warning] PredID {pid} not found in GT list.")
        else:
            print("[Warning] No Predicted CowID for current Track.")

    elif key in (ord('f'), ord('F')):
        assigned = clean_format(track_df.at[cur_track_idx, "assignedCowID"])
        if assigned and not str(assigned).startswith("ignore"):
            gt_str_ids = gt_df.iloc[:, GT_ID_IDX].apply(clean_format)
            matches = gt_df.index[gt_str_ids == assigned]
            if len(matches) > 0:
                cur_gt_index = matches[0]
                print(f"[Info] Found CowID {assigned} at GT Index {cur_gt_index + 1}.")
            else:
                print(f"[Warning] CowID {assigned} not found in GT list.")
        else:
            print("[Warning] Current Track has no valid assigned CowID.")

    elif key in (ord('g'), ord('G')):
        assigned_trk_val = clean_format(gt_df.at[cur_gt_index, "assignedTrackID"])
        s_val = str(assigned_trk_val)

        if assigned_trk_val and s_val.startswith("ignore") == False:
            target_trk = int(assigned_trk_val)
            found_rows = track_df.index[track_df["track_id"] == target_trk].tolist()
            if found_rows:
                cur_track_idx = found_rows[0]
                print(f"[Info] Jumped to Track Row {cur_track_idx+1}")
            else:
                print(f"[Warning] Assigned TrackID {target_trk} not found in Track list.")
        else:
            print("[Warning] Current GT row has no assigned TrackID.")
    
    # === Key: o (JUMP TO NEXT FLAG 4) ===
    elif key in (ord('o'), ord('O')):

        flag4_indices = track_df.index[track_df["flag"].apply(clean_format) == "4"].tolist()
    
        if not flag4_indices:
            print("[Jump] No Flag 4 (Ignore) rows found.")
        else:
            next_indices = [i for i in flag4_indices if i > cur_track_idx]
            
            if next_indices:
                cur_track_idx = next_indices[0]
                print(f"[Jump] To next Flag 4 at {cur_track_idx + 1}")
            else:
                cur_track_idx = flag4_indices[0]
                print(f"[Jump] Wrapped to first Flag 4 at {cur_track_idx + 1}")

    # === Key: q (SAVE & QUIT) ===
    elif key in (ord('q'), ord('Q')):
        cv2.destroyAllWindows()
        track_df.to_csv(track_save_csv, index=False)
        gt_df.to_csv(gt_save_csv, index=False)
        print("Progress Saved. Exiting...")
        sys.exit(0)

    if cur_track_idx != prev_track or cur_gt_index != prev_gt:
        smooth_scroll(prev_track, cur_track_idx, prev_gt, cur_gt_index, 
                      track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays)