import os
import cv2
import re
import numpy as np
import pandas as pd
import argparse
import sys
import math
from utils import render_scene, clean_format, recalc_ignore_counter, smooth_scroll


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
    cols_to_merge = [c for c in ["track_id", "cowID", "flag", "mark"] if c in saved_df.columns]
    track_df = pd.merge(base_df, saved_df[cols_to_merge], on="track_id", how="left")
else:
    print("[Info] Starting fresh Track log...")
    track_df = base_df
    track_df["cowID"] = None
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

ignore_counter = recalc_ignore_counter(track_df, gt_df, GT_ID_IDX)
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
track_df["cowID"] = track_df["cowID"].astype("object")
track_df["flag"] = track_df["flag"].astype("object")
gt_df["assignedTrackID"] = gt_df["assignedTrackID"].astype("object")
gt_df["flag"] = gt_df["flag"].astype("object")

# ============================================================
# ==================== SET START POSITIONS ===================
# ============================================================

total_tracks = len(track_df)
total_gt = len(gt_df)

last_auto = track_df["cowID"].last_valid_index()
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

img0 = render_scene(cur_track_idx, cur_gt_index, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX)
cv2.imshow("Cow Matcher", img0)
cv2.waitKey(1)

while True:
    img = render_scene(cur_track_idx, cur_gt_index, track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX)
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
        track_assigned = clean_format(track_df.at[cur_track_idx, "cowID"])
        
        if gt_assigned:
            print(f"[BLOCK] GT Cow {cow_id} already assigned to Track {gt_assigned}. Cancel ('x') first.")
            continue 
        if track_assigned:
            print(f"[BLOCK] Current Track {track_id} already assigned to {track_assigned}. Cancel ('x') first.")
            continue

        track_df.at[cur_track_idx, "cowID"] = cow_id
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
            track_df.loc[track_mask, "cowID"] = None
            track_df.loc[track_mask, "flag"] = None
            print("[Auto] Cleared.")
        else:
            print("\n[Auto] Toggling ON: Running Smart Fill + Ignore Remaining...")
            
            # Recalc Ignore Counter BEFORE filling
            ignore_counter = recalc_ignore_counter(track_df, gt_df, GT_ID_IDX)
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
                    
                    c_start = clean_format(track_df.at[s,"cowID"])
                    c_end = clean_format(track_df.at[e,"cowID"])
                    
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
                                
                                track_df.at[a_idx, "cowID"] = gt_cid
                                track_df.at[a_idx, "flag"] = "2"
                                gt_df.at[gt_idx, "assignedTrackID"] = a_tid
                                gt_df.at[gt_idx, "flag"] = "2"
                                filled_count += 1
            print(f"   -> Interpolated {filled_count} rows (Orange).")

            # Phase B: Fill Remaining with Ignore
            empty_mask = track_df["cowID"].isna() | (track_df["cowID"] == "")
            empty_indices = track_df.index[empty_mask].tolist()
            
            ignore_filled_count = 0
            for idx in empty_indices:
                ign_val = f"ignore_{ignore_counter}"

                track_df.at[idx, "cowID"] = ign_val
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
        if pd.notna(track_df.at[cur_track_idx, "cowID"]):
             print(f"[BLOCK] Track already assigned. Cancel first.")
        else:
            ignore_counter = recalc_ignore_counter(track_df, gt_df, GT_ID_IDX)
            
            track_df.at[cur_track_idx, "cowID"] = f"ignore_{ignore_counter}"
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
        assigned_cow = clean_format(track_df.at[cur_track_idx, "cowID"])
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

        track_df.at[cur_track_idx, "cowID"] = None
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
        assigned = clean_format(track_df.at[cur_track_idx, "cowID"])
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
                      track_df, gt_df, compact_mode, gt_subfolders, track_img_dir, gt_date_overlays, GT_ID_IDX)