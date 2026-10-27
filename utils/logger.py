import csv
import os
from typing import Dict, Any, List


class CSVLogger:
    def __init__(self, output_dir: str, filename: str, fieldnames: List[str]):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, filename)
        self.fieldnames = fieldnames
        file_exists = os.path.isfile(self.path)
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not file_exists:
            self._writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
