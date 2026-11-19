"""Check saved XLSX values against verified metrics and the original workbook."""

import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
BOOK = ROOT / "outputs/01a1ce1e-2fa6-7d01-b565-d791a397862c/so-sanh-mo-hinh-la-ca-chua.xlsx"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_book(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(node.itertext()) for node in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        names = [sheet.get("name") for sheet in book.find("s:sheets", NS)]
        sheets = []
        for index, name in enumerate(names, 1):
            tree = ET.fromstring(archive.read(f"xl/worksheets/sheet{index}.xml"))
            values, formulas = {}, {}
            for cell in tree.findall(".//s:sheetData/s:row/s:c", NS):
                ref, kind = cell.get("r"), cell.get("t")
                assert kind != "e", (name, ref, ET.tostring(cell))
                value = cell.findtext("s:v", None, NS)
                if kind == "s":
                    value = shared[int(value)]
                elif kind == "inlineStr":
                    value = "".join(cell.find("s:is", NS).itertext())
                elif kind in (None, "n") and value is not None:
                    value = float(value)
                values[ref] = value
                formula = cell.findtext("s:f", None, NS)
                if formula is not None:
                    formulas[ref] = formula
            sheets.append((values, formulas, tree))
        tables = [ET.fromstring(archive.read(f"xl/tables/table{i}.xml")) for i in range(1, 5)]
        return names, sheets, tables


def equal(actual, expected, context):
    if expected == "":
        expected = None
    if isinstance(expected, (float, int)):
        assert isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (context, actual, expected)
    else:
        assert actual == expected, (context, actual, expected)


def main():
    data = json.loads((RUN / "metrics-complete.json").read_text(encoding="utf-8"))
    names, sheets, tables = read_book(BOOK)
    old_names, old_sheets, _ = read_book(RUN / "workbook-before.xlsx")
    assert names == old_names
    checked = 0
    for index, key in enumerate(["scores", "classes", "epochs", "extras"], 1):
        values, formulas, tree = sheets[index]
        assert not formulas
        for row, record in enumerate(data[key], 2):
            for col, expected in enumerate(record):
                ref = f"{chr(65 + col)}{row}"
                equal(values.get(ref), expected, (names[index], ref))
                checked += 1
        old_values = old_sheets[index][0]
        for ref, expected in old_values.items():
            equal(values.get(ref), expected, ("preserved", names[index], ref))
        pane = tree.find("s:sheetViews/s:sheetView/s:pane", NS)
        assert pane is not None and pane.get("ySplit") == "1" and pane.get("state") == "frozen"
        table = tables[index - 1]
        bounds = f"A1:{chr(64 + len(data[key][0]))}{len(data[key]) + 1}"
        assert table.get("ref") == bounds
        assert table.find("s:autoFilter", NS).get("ref") == bounds
    values, formulas, _ = sheets[0]
    info = data["info"]
    score_index = {tuple(row[:3]): row for row in data["scores"]}
    summary_count = 0
    for title, models, specs in [
        ("Benchmark test đã đo", data["field_models"], [(ds, cut, i) for ds in [info["LEGACY"], info["EXT"]] for cut in [info["ALL"], info["FIELD"]] for i in [7, 6]]),
        ("Validation DINO chung", data["common_models"], [(info["DINO_VAL"], cut, i) for cut, i in [(info["FIELD"], 3), (info["FIELD"], 5), (info["FIELD"], 6), (info["FIELD"], 7), (info["ALL"], 5), (info["ALL"], 6), (info["ALL"], 7)]]),
    ]:
        start = next(int(ref[1:]) + 2 for ref, val in values.items() if ref.startswith("A") and isinstance(val, str) and val.startswith(title))
        for row, model in enumerate(models, start):
            assert values[f"A{row}"] == model
            for col, (dataset, cut, metric) in enumerate(specs, 1):
                ref = f"{chr(65 + col)}{row}"
                record = score_index.get((model, dataset, cut))
                expected = record[metric] if record is not None else None
                equal(values.get(ref), "—" if expected is None else expected, (title, model, ref))
                if expected is not None:
                    assert formulas[ref].startswith("SUMIFS("), ref
                summary_count += 1
    assert any(isinstance(v, str) and "994 ảnh lịch sử" in v for v in values.values())
    assert any(isinstance(v, str) and "không phải test độc lập" in v for v in values.values())
    for ref, expected in old_sheets[0][0].items():
        if int("".join(c for c in ref if c.isdigit())) <= 38:
            equal(values.get(ref), expected, ("unchanged overview", ref))
    receipt = {"workbook": str(BOOK), "sha256": hashlib.sha256(BOOK.read_bytes()).hexdigest(), "bytes": BOOK.stat().st_size, "detail_cells_checked": checked, "benchmark_and_validation_cells_checked": summary_count, "all_original_detail_values_preserved": True, "original_overview_top_38_rows_preserved": True, "crc": "pass", "formula_errors": 0, "filters_and_freeze_panes": "pass", "native_excel_execution": "not_run"}
    (RUN / "workbook-verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
