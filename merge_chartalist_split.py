import os
import re
import sys

def merge_split_csv(input_dir, split_start, split_end, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    in_files = []
    out_files = []
    pattern = re.compile(r"(\d+)_(in|out)\.csv")

    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if not match:
            continue
        block = int(match.group(1))
        ftype = match.group(2)
        full_path = os.path.join(input_dir, fname)
        if ftype == "in":
            in_files.append((block, full_path))
        else:
            out_files.append((block, full_path))

    in_files.sort(key=lambda x: x[0])
    out_files.sort(key=lambda x: x[0])

    def merge_raw_files(file_list, output_name):
        output_path = os.path.join(output_dir, output_name)
        total_lines = 0
        with open(output_path, "w") as out:
            for _, path in file_list:
                try:
                    with open(path, "r") as f:
                        content = f.read()
                        if content.strip():
                            out.write(content)
                            if not content.endswith("\n"):
                                out.write("\n")
                            total_lines += content.count("\n")
                except Exception as e:
                    print(f"Ошибка при обработке {path}: {e}")
        print(f"{output_name}: объединено {len(file_list)} файлов, ≈{total_lines} строк.")

    merge_raw_files(in_files, f"bitcoin_in_{split_start}_{split_end}.csv")
    merge_raw_files(out_files, f"bitcoin_out_{split_start}_{split_end}.csv")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    input_dir = sys.argv[1]
    split_start = int(sys.argv[2])
    split_end = int(sys.argv[3])
    merge_split_csv(input_dir, split_start, split_end)