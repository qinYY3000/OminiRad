"""Chest anatomy lexicon for CXR — 14 standardized regions.

Used for MIMIC-CXR `<LOC>` supervision (when reports mention specific lobes).
"""

CHEST_ANATOMY_LEXICON = [
    # ----- Lung lobes (right) -----
    (r"right\s+upper\s+lobe|RUL\b",       "right_upper_lobe"),
    (r"right\s+middle\s+lobe|RML\b",      "right_middle_lobe"),
    (r"right\s+lower\s+lobe|RLL\b",       "right_lower_lobe"),
    # ----- Lung lobes (left) -----
    (r"left\s+upper\s+lobe|LUL\b",        "left_upper_lobe"),
    (r"lingula|lingular",                 "lingula"),
    (r"left\s+lower\s+lobe|LLL\b",        "left_lower_lobe"),
    # ----- Zones (when lobes are not stated) -----
    (r"right\s+(upper|apical)\s+zone",    "right_upper_zone"),
    (r"right\s+(mid|middle)\s+zone",      "right_mid_zone"),
    (r"right\s+(lower|basal)\s+zone",     "right_lower_zone"),
    (r"left\s+(upper|apical)\s+zone",     "left_upper_zone"),
    (r"left\s+(mid|middle)\s+zone",       "left_mid_zone"),
    (r"left\s+(lower|basal)\s+zone",      "left_lower_zone"),
    # ----- Central / mediastinal -----
    (r"retrocardiac",                     "retrocardiac"),
    (r"mediastinum|mediastinal",          "mediastinum"),
    (r"hilum|hilar",                      "hilar"),
    (r"pleura|pleural",                   "pleural"),
    (r"cardiac\s+silhouette|cardiomegaly", "cardiac_silhouette"),
    # ----- Side-only fallbacks -----
    (r"right\s+lung",                     "right_lung_unspecified"),
    (r"left\s+lung",                      "left_lung_unspecified"),
    (r"bilateral|both\s+lungs",           "bilateral_lungs"),
]

CHEST_CODES = [code for _, code in CHEST_ANATOMY_LEXICON]
