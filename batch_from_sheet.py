#!/usr/bin/env python3
"""
ComfyUI Auto Batch from CSV (Sheets Version)

Membaca prompt dari file CSV (misalnya export dari Google Sheets) dengan format:

    Promt P | Promt N | Workflow | Count

Kolom:
- Promt P  : prompt positif (wajib)
- Promt N  : prompt negatif (boleh kosong, khususnya untuk Flux)
- Workflow : salah satu dari:
    Qwen_16:9
    Qwen_1:1
    Qwen_2:1_Vector
    Qwen_lp_vector
    Flux_16:9
    Flux_1:1
    Flux_9:16
- Count    : jumlah gambar yang ingin digenerate (boleh kosong → dianggap 1)

Contoh pakai:

    python batch_from_sheet.py --csv prompts/example.csv --server 127.0.0.1:8188

Script ini tidak mengubah batch_processor.py, hanya memanfaatkan class ComfyUIAutoBatch di dalamnya.
"""

import csv
import argparse
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

from batch_processor import ComfyUIAutoBatch  # memakai client ComfyUI yang sudah ada


# === KONFIGURASI HEADER CSV ===
COL_PROMT_P = "Promt P"
COL_PROMT_N = "Promt N"
COL_WORKFLOW = "Workflow"
COL_COUNT = "Count"

# === MAPPING nama workflow di Sheets -> nama file workflow JSON di folder 'workflows' ===
WORKFLOW_NAME_MAP: Dict[str, str] = {
    "Qwen_16:9": "Qwen_16_9",
    "Qwen_1:1": "Qwen_1_1",
    "Qwen_2:1_Vector": "Qwen_2_1_Vector",
    "Qwen_lp_vector": "Qwen_lp_vector",
    "Flux_16:9": "Flux_16_9",
    "Flux_1:1": "Flux_1_1",
    "Flux_9:16": "Flux_9_16",
}


def read_csv_prompts(csv_path: Path) -> List[Dict[str, str]]:
    """
    Baca file CSV dan kembalikan list baris (dict per row).
    Mengharapkan header: Promt P, Promt N, Workflow, Count
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[Dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Validasi header
        required_cols = {COL_PROMT_P, COL_WORKFLOW}
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV header kurang kolom: {', '.join(missing)}. "
                f"Header yang ditemukan: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader, start=2):  # start=2 karena baris 1 = header
            # Normalize keys → strip whitespace
            normalized = {k.strip(): (v.strip() if isinstance(v, str) else v)
                          for k, v in row.items()}

            promt_p = normalized.get(COL_PROMT_P, "") or ""
            workflow_name = normalized.get(COL_WORKFLOW, "") or ""

            # Skip baris kosong / tidak punya prompt atau workflow
            if not promt_p or not workflow_name:
                continue

            rows.append(normalized)

    return rows


def update_workflow_prompts(
    workflow: Dict[str, Any],
    promt_p: str,
    promt_n: Optional[str],
    workflow_label: str,
    logger=None,
) -> Dict[str, Any]:
    """
    Update workflow JSON dengan:
    - Promt P -> CLIPTextEncode positif (node pertama)
    - Promt N -> CLIPTextEncode negatif (node kedua), HANYA jika:
        - promt_n tidak kosong
        - workflow BUKAN Flux_*

    Selain itu, seed pada KSampler / KSamplerAdvanced akan dirandom ulang.
    """
    clip_nodes: List[Dict[str, Any]] = []

    # Kumpulkan semua node CLIPTextEncode
    for node_id, node_data in workflow.items():
        if isinstance(node_data, dict) and node_data.get("class_type") == "CLIPTextEncode":
            clip_nodes.append(node_data)

    # Positive prompt → node pertama (jika ada)
    if clip_nodes and promt_p:
        inputs = clip_nodes[0].setdefault("inputs", {})
        if "text" in inputs:
            inputs["text"] = promt_p

    # Negative prompt → node kedua (jika ada), hanya untuk non-Flux
    use_negative = bool(promt_n) and not workflow_label.startswith("Flux")
    if use_negative and len(clip_nodes) >= 2:
        inputs_neg = clip_nodes[1].setdefault("inputs", {})
        if "text" in inputs_neg:
            inputs_neg["text"] = promt_n or ""

    # Random seed untuk variasi
    for node_id, node_data in workflow.items():
        if isinstance(node_data, dict):
            class_type = node_data.get("class_type")
            if class_type in ["KSampler", "KSamplerAdvanced"]:
                inputs = node_data.get("inputs", {})
                if "seed" in inputs:
                    inputs["seed"] = random.randint(1, 2**32 - 1)

    if logger:
        if not clip_nodes:
            logger.warning("⚠️  Tidak ada node CLIPTextEncode ditemukan di workflow ini")
        else:
            logger.debug(
                f"✨ Update prompts: P='{promt_p[:40]}', "
                f"N={'<kosong>' if not promt_n else promt_n[:40]}"
            )

    return workflow


def parse_count(value: Optional[str]) -> int:
    """
    Parse kolom Count menjadi integer aman.
    - Jika kosong / tidak valid → 1
    - Jika <1 → 1
    - Jika terlalu besar → dibatasi ke 50 (safety)
    """
    if not value:
        return 1
    try:
        c = int(value)
        if c < 1:
            return 1
        if c > 50:
            return 50
        return c
    except ValueError:
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="ComfyUI Auto Batch dari CSV (Sheets)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="prompts/example.csv",
        help="Path ke file CSV (default: prompts/example.csv)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:8188",
        help="Alamat server ComfyUI (default: 127.0.0.1:8188)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)

    # Inisialisasi client ComfyUI dari batch_processor.py
    processor = ComfyUIAutoBatch(server_address=args.server, max_concurrent=1)

    # Tes koneksi
    if not processor.test_connection():
        print("❌ Tidak bisa terhubung ke server ComfyUI.")
        return

    # Baca CSV
    try:
        rows = read_csv_prompts(csv_path)
    except Exception as e:
        print(f"❌ Gagal membaca CSV: {e}")
        return

    if not rows:
        print("⚠️ Tidak ada baris valid di CSV (Promt P + Workflow wajib diisi).")
        return

    logger = processor.logger
    logger.info(f"📄 Memuat {len(rows)} baris dari CSV: {csv_path}")

    # Hitung total generasi (Count)
    total_generations = 0
    parsed_tasks = []
    for idx, row in enumerate(rows, start=1):
        promt_p = row.get(COL_PROMT_P, "") or ""
        promt_n = row.get(COL_PROMT_N, "") or ""
        workflow_label = row.get(COL_WORKFLOW, "") or ""
        count_raw = row.get(COL_COUNT, "") or ""

        count = parse_count(count_raw)
        total_generations += count

        parsed_tasks.append({
            "row_index": idx,
            "promt_p": promt_p,
            "promt_n": promt_n,
            "workflow_label": workflow_label,
            "count": count,
        })

    logger.info(f"🎯 Total generasi yang akan dijalankan: {total_generations}")

    # Set statistik dasar
    processor.stats["total_queued"] = total_generations
    processor.stats["start_time"] = time.time()

    # Proses tiap baris secara berurutan (tanpa multithreading untuk kesederhanaan)
    completed = 0
    failed = 0
    skipped = 0

    current_gen = 0

    for task in parsed_tasks:
        row_idx = task["row_index"]
        promt_p = task["promt_p"]
        promt_n = task["promt_n"]
        workflow_label = task["workflow_label"]
        count = task["count"]

        mapped_name = WORKFLOW_NAME_MAP.get(workflow_label)
        if not mapped_name:
            logger.warning(
                f"⏭️  Baris {row_idx}: Workflow '{workflow_label}' tidak dikenal, dilewati."
            )
            skipped += count
            processor.stats["skipped"] += count
            continue

        # Load workflow JSON
        workflow = processor.load_workflow(mapped_name)
        if workflow is None:
            logger.warning(
                f"⏭️  Baris {row_idx}: File workflow '{mapped_name}.json' tidak ditemukan, dilewati."
            )
            skipped += count
            processor.stats["skipped"] += count
            continue

        logger.info(
            f"▶️  Baris {row_idx}: Workflow='{workflow_label}' "
            f"Count={count} | P='{promt_p[:40]}...' "
            f"{'| N=<' + promt_n[:30] + '...>' if promt_n else '| N=<kosong>'}"
        )

        for i in range(count):
            current_gen += 1
            job_label = f"row{row_idx}_#{i+1}"

            # Copy workflow & update prompt + seed
            wf_copy = update_workflow_prompts(
                workflow=workflow.copy(),
                promt_p=promt_p,
                promt_n=promt_n,
                workflow_label=workflow_label,
                logger=logger,
            )

            # Queue prompt
            prompt_id = processor.queue_prompt_with_retry(wf_copy)
            if not prompt_id:
                logger.error(f"❌ {job_label}: Gagal queue prompt.")
                failed += 1
                processor.stats["failed"] += 1
                continue

            # Tunggu selesai
            ok = processor.wait_for_completion(prompt_id)
            if ok:
                completed += 1
                processor.stats["completed"] += 1
                logger.info(
                    f"✅ {job_label}: selesai ({current_gen}/{total_generations})"
                )
            else:
                failed += 1
                processor.stats["failed"] += 1
                logger.error(
                    f"❌ {job_label}: timeout/gagal ({current_gen}/{total_generations})"
                )

    processor.stats["end_time"] = time.time()

    logger.info(
        f"🏁 Selesai. Completed={completed}, Failed={failed}, Skipped={skipped}, "
        f"Total={total_generations}"
    )

    try:
        processor.print_final_stats()
    except Exception:
        # Kalau ada error karena stats belum lengkap, minimal jangan bikin script crash
        pass


if __name__ == "__main__":
    main()
