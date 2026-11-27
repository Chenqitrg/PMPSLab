"""Read experiment summaries without Julia, tensor objects, or a store API."""
import json
import sys
from pathlib import Path


def read_summaries(directory: Path) -> None:
    for file in sorted(directory.glob("*.json")):
        record = json.loads(file.read_text())
        if "measurement" not in record:
            continue
        result = record["measurement"]
        print(record["label"], "error=", result["relative_squared_error"],
              "max sigma2/sigma1=", result["max_second_ratio"],
              "parent=", record.get("parent_run"))


if __name__ == "__main__":
    read_summaries(Path(sys.argv[1]))
