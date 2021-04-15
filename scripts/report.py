import os
import re
from retinanet.utils import create_directory


"""read a csv file representing a table and write a restructured text simple
table"""
import sys
import csv
import io


def get_out(out=None):
    """
    return a file like object from different kinds of values
    None: returns stdout
    string: returns open(path)
    file: returns itself

    otherwise: raises ValueError
    """

    if out is None:
        return sys.stdout
    elif isinstance(out, io.TextIOBase):
        return out
    elif isinstance(out, str):
        return open(out)
    else:
        raise ValueError("out must be None, file or path")


def underline(title, underliner="=", endl="\n", out=None):
    """
    write *title* *underlined* to *out*
    """
    out = get_out(out)

    out.write(title)
    out.write(endl)
    out.write(underliner * len(title))
    out.write(endl * 2)


def separate(sizes, out=None, separator="=", endl="\n"):
    """
    write the separators for the table using sizes to get the
    size of the longest string of a column
    """
    out = get_out(out)

    for size in sizes:
        out.write(separator * size)
        out.write(" ")

    out.write(endl)


def write_row(sizes, items, out=None, endl="\n"):
    """
    write a row adding padding if the item is not the
    longest of the column
    """
    out = get_out(out)

    for item, max_size in zip(items, sizes):
        item_len = len(item)
        out.write(item)

        if item_len < max_size:
            out.write(" " * (max_size - item_len))

        out.write(" ")

    out.write(endl)


def process(in_=None, out=None, title=None):
    """
    read a csv table from in and write the rest table to out
    print title if title is set
    """
    handle = get_out(in_)
    out = get_out(out)

    reader = csv.reader(handle)

    rows = [row for row in reader if row]
    cols = len(rows[0])
    sizes = [0] * cols

    for i in range(cols):
        for row in rows:
            row_len = len(row[i])
            max_len = sizes[i]

            if row_len > max_len:
                sizes[i] = row_len

    if title:
        underline(title)

    separate(sizes, out)
    write_row(sizes, rows[0], out)
    separate(sizes, out)

    for row in rows[1:]:
        write_row(sizes, row, out)

    separate(sizes, out)


def combine_csv_report(
    csv_path_list: list, res_path: str, headers: list = None, add_row_names: bool = True
):
    with open(res_path, "w", encoding="utf-8") as outf:
        ofirst_line = True
        if headers is not None:
            outf.write(",".join(headers) + "\n")
            ofirst_line = False

        for fp in csv_path_list:
            ifirst_line = True
            with open(fp, "r", encoding="utf-8") as inf:

                line = inf.readline()
                if ifirst_line and ofirst_line:
                    ifirst_line = False
                    ofirst_line = False

                    if add_row_names:
                        line = "-," + line.replace("_", " ")
                    print(f"wrote : {line}")
                    outf.write(f"{line}")

                line = inf.readline()
                if add_row_names:
                    line = (
                        " ".join(
                            os.path.splitext(os.path.basename(fp))[0]
                            .split(".")[1]
                            .split("_")
                        )
                        + ","
                        + line.replace("\n", "")
                    )
                print(f"wrote : {line}")
                outf.write(f"{line}\n")


if __name__ == "__main__":
    ## Detection Report
    path = "experiments/logs/"
    detectors_list = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if re.search(r".*det.*\.csv$", f)
    ]
    detectors_list.sort()
    print(detectors_list)
    create_directory("./reports")
    combine_csv_report(detectors_list, "./reports/det_report.csv")
    with open("REPORT.rst", "w") as f:
        print("GENERATE REPORT")
        f.write("Object Detection on Large Images Task\n")
        f.write("=====================================\n\n")
        process("./reports/det_report.csv", out=f, title=None)
        f.write("\n\n")

    ## Classification Report
    path = "experiments/logs/"
    detectors_list = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if re.search(r".*cls.*\.csv$", f)
    ]
    detectors_list.sort()
    print(detectors_list)
    create_directory("./reports")
    combine_csv_report(detectors_list, "./reports/cls_report.csv")
    with open("REPORT.rst", "a") as f:
        print("GENERATE REPORT")
        f.write("Object Detection Task\n")
        f.write("=====================\n\n")
        process("./reports/det_report.csv", out=f, title=None)
        f.write("\n\n")

    ## Large Image Detection Report
    path = "experiments/logs/"
    detectors_list = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if re.search(r".*largeImageEval.*\.txt$", f)
    ]
    detectors_list.sort()
    print(detectors_list)
    create_directory("./reports")
    combine_csv_report(detectors_list, "./reports/large_image_det_report.csv")
    with open("REPORT.rst", "a") as f:
        print("GENERATE REPORT")
        f.write("Image Classification Task\n")
        f.write("=========================\n\n")
        process("./reports/det_report.csv", out=f, title=None)
        f.write("\n\n")