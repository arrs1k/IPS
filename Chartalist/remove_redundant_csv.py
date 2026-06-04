import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="Удалить отдельные {num}_in.csv и {num}_out.csv после объединения.")
    parser.add_argument("folder", help="Папка с маленькими CSV-файлами")
    parser.add_argument("--yes", action="store_true", help="Удалить без запроса подтверждения")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет удалено")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Ошибка: папка '{folder}' не существует.")
        return

    pattern = re.compile(r"^\d+_(in|out)\.csv$")
    files_to_delete = []

    for fname in os.listdir(folder):
        if pattern.match(fname):
            files_to_delete.append(os.path.join(folder, fname))

    if not files_to_delete:
        print("Нет файлов для удаления.")
        return

    print("Найдены файлы для удаления:")
    for path in files_to_delete:
        print(f"  {path}")

    if args.dry_run:
        print(f"\nБудет удалено {len(files_to_delete)} файлов (режим dry-run).")
        return

    if not args.yes:
        answer = input(f"\nУдалить {len(files_to_delete)} файлов? [y/N]: ").strip().lower()
        if answer != "y":
            print("Отмена.")
            return

    count = 0
    for path in files_to_delete:
        try:
            os.remove(path)
            count += 1
        except OSError as e:
            print(f"Ошибка при удалении {path}: {e}")

    print(f"Удалено {count} файлов.")

if __name__ == "__main__":
    main()