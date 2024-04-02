"""
Rewrite all .py files and convert class and function description in comments above the 
definition to docstrings.
"""

import re


def convert_comments_to_docstrings(path_in, path_out):
    with open(path_in, "r") as file:
        lines = file.readlines()

    new_lines = []
    in_comment_block = False
    in_func_block = False
    func_def_done = False
    docstring = ""
    whitespace = ""

    for line in lines:
        # Write the docstring after func def
        if func_def_done:
            assert not in_comment_block
            assert not in_func_block
            # don't write empty docstrings
            if docstring != "":
                # function definition whitespace + extra indent
                indent = whitespace + "    "
                new_lines.append(indent + '"""! ' + docstring + '"""\n')
                docstring = ""
            func_def_done = False
            in_comment_block = False
            in_func_block = False

        # encounter comment block start
        if not in_comment_block and line.strip().startswith("##"):
            assert not in_func_block
            in_comment_block = True
            comment_line = line
            docstring: str = line.strip().lstrip("##") + "\n"
        # next line in comment block
        elif in_comment_block and line.strip().startswith("#"):
            assert not in_func_block
            docstring += line.strip().lstrip("#") + "\n"

        # start of function/class definition
        elif line.strip().startswith("def ") or line.strip().startswith("class "):
            assert not in_func_block
            assert not func_def_done
            in_comment_block = False

            # one line function
            if line.strip().endswith(":"):
                func_def_done = True
                in_func_block = False
            else:
                func_def_done = False
                in_func_block = True
            whitespace = re.match(r"\s*", line).group()  # type: ignore
            new_lines.append(line)
        # next line in func block
        elif in_func_block:
            assert not in_comment_block
            assert not func_def_done
            if line.strip().endswith(":"):
                func_def_done = True
                in_func_block = False
            else:
                func_def_done = False
                in_func_block = True

            new_lines.append(line)

        # so not actually a comment block, just a single line with ##
        elif in_comment_block and not line.strip().startswith("#"):
            assert not in_func_block
            in_comment_block = False
            # make sure that it's only one line
            if not docstring in comment_line:
                print(
                    f"Error: multiple comment lines with ## detected in {path_in} at line {lines.index(line)+1}"
                )
                return -1

            new_lines.append(comment_line)
            docstring = ""
            new_lines.append(line)

        else:
            in_comment_block = False
            new_lines.append(line)

    with open(path_out, "w") as file:
        file.writelines(new_lines)
        print(f"Written file {path_out}")


if __name__ == "__main__":
    import os

    rootdir = "zigzag"

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.split(".")[-1] == "py":
                path = os.path.join(subdir, file)
                convert_comments_to_docstrings(path, path)
                # print(path)

    # path_in = "zigzag/classes/cost_model/cost_model_for_sram_imc.py"
    # path_out = "zigzag/classes/cacti/cacti_parser_new.py"
    # path_in = input("path: ")
