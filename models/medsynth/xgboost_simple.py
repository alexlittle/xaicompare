import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from expailens.runner import publish_run


df = pd.read_csv('../../data/medsynth/MedSynth_huggingface_final.csv')

ICD10_CHAPTERS = {
    "A": ("A00", "B99", "Certain infectious and parasitic diseases"),
    "B": ("A00", "B99", "Certain infectious and parasitic diseases"),
    "C": ("C00", "D49", "Neoplasms"),
    "D": ("C00", "D49", "Neoplasms"),  # D00–D49 = neoplasms
    "E": ("E00", "E89", "Endocrine, nutritional and metabolic diseases"),
    "F": ("F01", "F99", "Mental, behavioral and neurodevelopmental disorders"),
    "G": ("G00", "G99", "Diseases of the nervous system"),
    "H": ("H00", "H95", "Diseases of eye/ear/adnexa/mastoid"),  # H00–H95 split but ok as one area
    "I": ("I00", "I99", "Diseases of the circulatory system"),
    "J": ("J00", "J99", "Diseases of the respiratory system"),
    "K": ("K00", "K95", "Diseases of the digestive system"),
    "L": ("L00", "L99", "Diseases of the skin and subcutaneous tissue"),
    "M": ("M00", "M99", "Diseases of musculoskeletal system"),
    "N": ("N00", "N99", "Diseases of the genitourinary system"),
    "O": ("O00", "O9A", "Pregnancy, childbirth and puerperium"),
    "P": ("P00", "P96", "Perinatal conditions"),
    "Q": ("Q00", "Q99", "Congenital malformations"),
    "R": ("R00", "R99", "Symptoms, signs, abnormal findings"),
    "S": ("S00", "T88", "Injury and poisoning"),
    "T": ("S00", "T88", "Injury and poisoning"),
    "V": ("V00", "Y99", "External causes of morbidity"),
    "W": ("V00", "Y99", "External causes of morbidity"),
    "X": ("V00", "Y99", "External causes of morbidity"),
    "Y": ("V00", "Y99", "External causes of morbidity"),
    "Z": ("Z00", "Z99", "Factors influencing health status"),
    "U": ("U00", "U85", "Special purposes")
}

def map_icd10_to_chapter(code):
    """Return official ICD-10 chapter name from any ICD-10 code."""
    if pd.isna(code):
        return None
    code = str(code).strip()
    first_letter = code[0].upper()

    if first_letter in ICD10_CHAPTERS:
        return ICD10_CHAPTERS[first_letter][2]  # return chapter name
    return "Unknown"

df = df.dropna(subset=["Dialogue"]).copy()

df["Dialogue"].apply(type).value_counts()


df["ICD_chapter"] = df["ICD10"].apply(map_icd10_to_chapter)
df["ICD_chapter"].value_counts()

X = df["Dialogue"]

le = LabelEncoder()
y = le.fit_transform(df["ICD_chapter"].astype(str))
class_names = list(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)



pipe = joblib.load("tfidf_xgb_pipeline.joblib")

publish_run(
    model=pipe,
    X_test=X_test,                          # raw texts if pipeline contains TF-IDF
    y_test=y_test,                          # optional
    raw_text=X_test,                        # so the dashboard can show the note
    class_names=getattr(pipe.named_steps["xgb"], "classes_", None),
    run_dir="runs/2026-02-18_chapters_xgb",
    config={"batch_size": 2,                # tiny batches to avoid OOM
            "rows_limit_global": 200,       # compute global on first 200 rows
            "rows_limit_local": 200}        # store local top-k for first 200 rows
)