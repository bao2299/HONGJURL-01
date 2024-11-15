import os


def print_directory_tree_to_file(startpath, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        def write_tree(startpath, prefix=""):
            items = os.listdir(startpath)
            for index, name in enumerate(items):
                path = os.path.join(startpath, name)
                is_last = index == len(items) - 1
                if os.path.isdir(path):
                    f.write(prefix + ("└── " if is_last else "├── ") + name + "\n")
                    write_tree(path, prefix + ("    " if is_last else "│   "))
                else:
                    f.write(prefix + ("└── " if is_last else "├── ") + name + "\n")

        write_tree(startpath)


# 使用当前工作目录
file_path = "directory_tree.txt"
print_directory_tree_to_file(os.getcwd(), file_path)
print(f"Directory structure has been saved to {file_path}")
