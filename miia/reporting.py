from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PAPER_REFERENCE_MR = {"rsicd": 37.81, "rsitmd": 51.06, "ucm": 56.78}


def write_reports(results: dict[str, Any], output_dir: str | Path, stem: str = "retrieval_results") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)

    rows = []
    for dataset, metrics in results.items():
        i2t = metrics["image_to_text"]
        t2i = metrics["text_to_image"]
        rows.append({
            "dataset": dataset,
            "I2T_R@1": i2t["R@1"],
            "I2T_R@5": i2t["R@5"],
            "I2T_R@10": i2t["R@10"],
            "T2I_R@1": t2i["R@1"],
            "T2I_R@5": t2i["R@5"],
            "T2I_R@10": t2i["R@10"],
            "mR": metrics["mR"],
            "paper_mR": PAPER_REFERENCE_MR.get(dataset),
            "delta_mR": metrics["mR"] - PAPER_REFERENCE_MR[dataset] if dataset in PAPER_REFERENCE_MR else None,
        })
    fields = list(rows[0]) if rows else ["dataset"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MIIA RET-3 retrieval results",
        "",
        "`I2T` means image-to-text and maps to the paper's **Text retrieval** columns; "
        "`T2I` means text-to-image and maps to **Image retrieval**.",
        "",
        "| Dataset | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | mR | Paper mR | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['I2T_R@1']:.2f} | {row['I2T_R@5']:.2f} | {row['I2T_R@10']:.2f} | "
            f"{row['T2I_R@1']:.2f} | {row['T2I_R@5']:.2f} | {row['T2I_R@10']:.2f} | "
            f"{row['mR']:.2f} | {row['paper_mR']:.2f} | {row['delta_mR']:+.2f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}

