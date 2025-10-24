#!/usr/bin/env python3
"""將 mapping / remap 的 NumPy 檔案轉換為 CSV。

預設輸出原始 ID（索引）對應 cluster ID；若提供 remap 檔案，則輸出
原始 ID -> remap ID（代表 token），可選擇是否保留 cluster ID。"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def _convert_mapping_to_csv(
    mapping_path: Path,
    output_path: Path,
    remap_path: Optional[Path] = None,
    include_header: bool = True,
    include_cluster_column: bool = False,
) -> None:
    array = np.load(str(mapping_path))
    if array.ndim != 1:
        raise ValueError("mapping 檔案必須為一維陣列")

    remap_array: Optional[np.ndarray] = None
    if remap_path is not None:
        remap_array = np.load(str(remap_path))
        if remap_array.ndim != 1:
            raise ValueError("remap 檔案必須為一維陣列")
        if array.max(initial=0) >= remap_array.shape[0]:
            raise ValueError("mapping 值超出 remap 長度，請確認兩者是否匹配")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if include_header:
            header = ["original_id"]
            if remap_array is not None:
                header.append("remap_id")
                if include_cluster_column:
                    header.append("cluster_id")
            else:
                header.append("cluster_id")
            writer.writerow(header)

        for original_id, cluster_id in enumerate(array.tolist()):
            row: list[int] = [original_id]
            if remap_array is not None:
                row.append(int(remap_array[int(cluster_id)]))
                if include_cluster_column:
                    row.append(int(cluster_id))
            else:
                row.append(int(cluster_id))
            writer.writerow(row)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="將 mapping npy 轉換為 CSV 格式")
    parser.add_argument(
        "--mapping",
        "-m",
        required=True,
        help="mapping 檔案路徑 (.npy)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="輸出的 CSV 檔案路徑，預設為與 mapping 同名的 .csv",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="輸出時不包含標題列",
    )
    parser.add_argument(
        "--remap",
        "-r",
        default=None,
        help="remap 檔案路徑 (.npy)，輸出原始 ID -> remap ID",
    )
    parser.add_argument(
        "--include-cluster-id",
        action="store_true",
        help="同時輸出 cluster ID 欄位（需搭配 remap）",
    )

    args = parser.parse_args(argv)

    mapping_path = Path(args.mapping)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"找不到 mapping 檔案: {mapping_path}")

    output_path = Path(args.output) if args.output else mapping_path.with_suffix(".csv")
    remap_path = Path(args.remap) if args.remap else None
    if remap_path and not remap_path.is_file():
        raise FileNotFoundError(f"找不到 remap 檔案: {remap_path}")

    _convert_mapping_to_csv(
        mapping_path=mapping_path,
        output_path=output_path,
        remap_path=remap_path,
        include_header=not args.no_header,
        include_cluster_column=args.include_cluster_id,
    )

    print(f"已將 {mapping_path} 轉換為 {output_path}")


if __name__ == "__main__":
    main()
