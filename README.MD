# Cow Matcher
---
Author: Haochuan Hu

## Cow Matcher Tool
Cow Matcher is a GUI tool designed to manually and semi-automatically align sequential "Track" images (from the [AutoCattlogger](https://github.com/VADL-Purdue/AutoCattlogger) system) with "Ground Truth" (GT) identities.

## Features
- Dual-Pane Interface: View Track images (left) and Ground Truth image banks (right) simultaneously.

- Smart Fill (Interpolation): Automatically fills gaps between two manually matched points (Flag 2).

- Auto-Ignore: Automatically fills remaining mismatched rows with unique ignore IDs (Flag 4).

- Visual Status Indicators: Color-coded borders indicate the status of each row (Matched, Interpolated, Ignored).

- Global Transformations: Supports rotating and mirroring input images via configuration.

![demo](GUI.gif)

## File & Naming Requirements
To run the tool, your data must follow specific naming conventions.

### 1. Track Images Folder
The images containing the cows to be identified.

- Filename Format: trackId_{ID}_pred_{PredID}.jpg or trackId_{ID}.jpg

- Regex: trackId_(\d+)(?:_pred_(\d+))?\.jpg

- Example:

    - trackId_105_pred_908.jpg (Track ID 105, Algorithm predicted Cow 908)

    -  trackId_106.jpg (Track ID 106, No prediction)

### 2. Ground Truth (GT) CSV
A CSV file listing the known sequence of cows.

- Required Columns: A column named `CowID`.

- Generated Columns: The tool will automatically add `assignedTrackID` and `flag` columns if they don't exist.

### 3. Image Bank (GT Images)
A directory containing subfolders of cow images.

```
Bank/
├── 2023-10-01/
│   ├── 908.jpg
│   └── 102.jpg
├── 2023-10-02/
│   └── ...
```

## Installation & Usage
Prerequisites
```bash
conda create -n cow_matcher python=3.12 -y && conda activate cow_matcher && pip install numpy pandas opencv-python
```
Running the Tool
```bash
python cow_matcher.py --track "./path/to/track_images" --gt "./path/to/list.csv" --bank "./path/to/bank"
```
- --track: Path to the folder containing the sequence of images to label.

- --gt: Path to the CSV file containing the Ground Truth list.

- --bank: (Optional) Path to the root folder of reference images (default: ./Bank/).

## Controls & Shortcuts
|Key|Function|	Description|
|-----:|---------------------------|------------|
|`1`	|Manual Match	|Links the current Track row to the current GT row (Blue).|
|`2`|	Smart Fill / Auto	|First Press: Interpolates gaps between manual matches (Orange) AND fills remaining empty rows with "Ignore" (Pink).|
|||Second Press: Undo/Clear all auto-filled entries.|
|`3`	|Manual Ignore|	Marks the current Track row as "Ignore" (Red). Used for bad images or non-cows.
|`x`	|Clear / Undo|	Clears the assignment for the current row.
|`r`	|Mark / Bookmark	|Toggles a "MARKED" status on the row. Useful for flagging difficult cases to review later.
|`m`	|Compact Mode	|Toggles the GT view to hide empty image columns.
|`q`	|Save & Quit	|Saves CSVs to the Progress/ folder and closes the tool.

## Navigation
|Key|Function|
|-----:|---------------------------|
|`w / s`|Scroll Track list Up / Down.|
|`a / d`| Scroll Ground Truth list Up / Down.
|`i / k`| Tandem Scroll: Moves both Track and GT lists Up/Down simultaneously (caps lock supported).|
|`t`|Jump to Next Mark (Bookmark).
|`o`|Jump to Next Auto-Ignore (Flag 4).
|`p`|Jump GT list to the ID matching the Prediction (from filename).
|`f`|Find Cow: Jumps GT list to the Cow ID currently assigned to the selected Track.
|`g`|Find Track: Jumps Track list to the ID currently assigned to the selected GT row.

## Output Files & Data Structure
The tool saves progress automatically to the ./Progress/ directory.
### 1. filename_corrected.csv (Track Data)
Contains the tracking data merged with your manual assignments.

|Column	|Description|
|-----:|---------------------------|
|track_id	|Original ID from the image filename.|
|cowID|	The Cow ID you assigned (or ignore_N).|
|flag	|Status code indicating how the assignment was made (see below).|
|mark	|1 if the row is bookmarked, 0 otherwise.|
|pred_cow_id	|The prediction extracted from the filename.|

### 2. filename_assigned.csv (GT Data)
Updates your original GT CSV to track which cows have been found.
|Column	|Description|
|-----:|---------------------------|
|assignedTrackID|	The Track ID assigned to this cow (or ignore_gt_N).|
|flag	|Status code (synchronized with the Track file).|


### 3. Flag Codes
|Flag	|Color	|Meaning|
|-----:|---------------------------|------------|
|`1`|	🔵 |Blue	Manual Match: User explicitly linked Track to GT.|
|`2`|	🟠 |Orange	Interpolated: Automatically filled based on start/end manual matches.|
|`3`|	🔴 |Red	Manual Ignore: User explicitly ignored this image.|
|`4`|	🌸 |Pink	Auto Ignore: Automatically filled because it did not fit the sequence or was remaining at the end.|
