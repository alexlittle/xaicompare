import requests
import time
import pandas as pd
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm



_ICD10_CHAPTERS = {
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

    if first_letter in _ICD10_CHAPTERS:
        return _ICD10_CHAPTERS[first_letter][2]  # return chapter name
    return "Unknown"

def download_medsynth():
    '''
    Downloads the MedSynth dataset and returns as a dataframe.
    :return: dataframe
    '''
    url = "https://huggingface.co/datasets/Ahmad0067/MedSynth/resolve/main/MedSynth_huggingface_final.csv"
    download_file_if_needed(url,'./data/MedSynth_huggingface_final.csv')
    df = pd.read_csv('./data/MedSynth_huggingface_final.csv')
    return df

def process_medsynth():
    df = download_medsynth()
    df = df.dropna(subset=["Dialogue"]).copy()

    df["Dialogue"].apply(type).value_counts()

    df["ICD_chapter"] = df["ICD10"].apply(map_icd10_to_chapter)
    df["ICD_chapter"].value_counts()

    df = df.rename(columns={" Note": "Note"})
    return df

def extract_vitalsigns_tocols(df):
    bp = df["Note"].str.extract(
        r'Blood\s*Pressure\s*[:\-]?\s*(?P<bp_sys>\d{2,3})\s*[/\-]\s*(?P<bp_dia>\d{2,3})\s*(?:mm\s*Hg|mmHg)?',
        flags=re.I
    )

    hr = df["Note"].str.extract(
        r'(?:Heart\s*Rate|HR)\s*[:\-]?\s*(?P<hr>\d{1,3})\s*(?:bpm|/min)?',
        flags=re.I
    )

    rr = df["Note"].str.extract(
        r'(?:Respiratory\s*Rate|RR)\s*[:\-]?\s*(?P<rr>\d{1,3})\s*(?:breaths?/min|rpm|/min)?',
        flags=re.I
    )

    temp = df["Note"].str.extract(
        r'Temperature\s*[:\-]?\s*(?P<temp>\d{2,3}(?:\.\d+)?)\s*°?\s*(?P<temp_unit>[FC]|(?:Fahrenheit|Celsius))',
        flags=re.I
    )

    spo2 = df["Note"].str.extract(
        r'(?:Oxygen\s*Saturation|SpO2)\s*[:\-]?\s*(?P<spo2>\d{2,3})\s*%(?:\s*(?:on|via)\s*(?P<o2_device>[^;\n]+))?',
        flags=re.I
    )

    # --- Combine and clean types ---
    vitals = pd.concat([bp, hr, rr, temp, spo2], axis=1)

    # Convert to numeric
    for col in ["bp_sys", "bp_dia", "hr", "rr", "temp", "spo2"]:
        if col in vitals:
            vitals[col] = pd.to_numeric(vitals[col], errors="coerce")

    # Normalize temperature units; create both °C and °F
    def _norm_unit(u):
        if u is None or (isinstance(u, float) and np.isnan(u)):
            return np.nan
        u = str(u).strip().lower()
        return "f" if u.startswith("f") else ("c" if u.startswith("c") else np.nan)

    vitals["temp_unit"] = vitals["temp_unit"].map(_norm_unit)

    vitals["temp_c"] = np.where(
        vitals["temp_unit"].str.lower().eq("f"),
        (vitals["temp"] - 32) * 5 / 9,
        vitals["temp"]
    )
    vitals["temp_f"] = np.where(
        vitals["temp_unit"].str.lower().eq("c"),
        vitals["temp"] * 9 / 5 + 32,
        vitals["temp"]
    )

    # Clean oxygen device text (e.g., "room air", "nasal cannula 2 L/min")
    if "o2_device" in vitals:
        vitals["o2_device"] = vitals["o2_device"].str.strip().str.rstrip(".")

    # Join back to df
    df = df.join(vitals)

    # rename columns
    df = df.rename(columns={
        "bp_sys": "BP_systolic",
        "bp_dia": "BP_diastolic",
        "hr": "Heart_Rate",
        "rr": "Respiratory_Rate",
        "spo2": "Oxygen_Saturation",
        "o2_device": "Oxygen_Device"
    })
    return df

def download_file_if_needed(
    url: str,
    dest_path: str | Path,
    chunk_size: int = 1024 * 1024,  # 1 MB
    max_retries: int = 5,
    timeout: int = 30,
    show_progress: bool = True,
    resume: bool = True,
):
    """
    Download a file from `url` to `dest_path` if not already present.
    - Streams data to avoid loading into memory.
    - Supports resume (HTTP Range) if a partial file exists and server supports it.
    - Shows a progress bar (tqdm).
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists and size > 0, skip
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"✅ File already exists: {dest_path}")
        return dest_path

    # Try to find total size and resume capability
    headers = {}
    try:
        head = requests.head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        total_size = int(head.headers.get("Content-Length", 0))
        accept_ranges = head.headers.get("Accept-Ranges", "").lower() == "bytes"
    except Exception:
        # HEAD may be blocked/not supported—fallback to unknown size
        total_size = 0
        accept_ranges = True  # optimistic: many servers support it

    # If partially downloaded file exists and resume requested
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    downloaded = temp_path.stat().st_size if temp_path.exists() else 0

    if resume and downloaded > 0 and accept_ranges:
        headers["Range"] = f"bytes={downloaded}-"

    # Progress bar setup
    if show_progress:
        bar_total = None if total_size == 0 else (total_size - downloaded if "Range" in headers else total_size)
        pbar = tqdm(total=bar_total, unit="B", unit_scale=True, desc=dest_path.name, initial=downloaded)
    else:
        pbar = None

    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
                if r.status_code in (200, 206):
                    mode = "ab" if "Range" in headers else "wb"
                    # If we requested a range but server ignored it (sent 200), start fresh
                    if "Range" in headers and r.status_code == 200:
                        downloaded = 0
                        mode = "wb"
                        if pbar:
                            pbar.reset(total=total_size or None)
                            pbar.n = 0
                            pbar.refresh()
                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                                if pbar:
                                    pbar.update(len(chunk))
                    # Rename temp to final
                    temp_path.replace(dest_path)
                    if pbar:
                        pbar.close()
                    print(f"✅ Downloaded to: {dest_path}")
                    return dest_path
                else:
                    r.raise_for_status()
        except Exception as e:
            attempt += 1
            if pbar:
                pbar.set_postfix_str(f"retry {attempt}/{max_retries}")
            # backoff
            time.sleep(min(2 ** attempt, 10))
            # On retry, recompute what’s already downloaded and set Range accordingly
            downloaded = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {}
            if resume and downloaded > 0 and accept_ranges:
                headers["Range"] = f"bytes={downloaded}-"
            if attempt >= max_retries:
                if pbar:
                    pbar.close()
                raise RuntimeError(f"Failed to download after {max_retries} attempts: {e}") from e