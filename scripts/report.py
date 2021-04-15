import os
import re
from retinanet.utils import create_directory


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
                        line = "," + line.replace("_", " ")
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