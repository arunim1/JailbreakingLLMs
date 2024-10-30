import re

def main(file_path):
    file_path = file_path + "/files/output.log"
    with open(file_path, "r") as f:
        # find the text between <main.py_output> and </main.py_output>
        text = re.findall(r'<main.py_output>(.*)</main.py_output>', f.read())
        
        # check to see if there is only one match
        if len(text) != 1:
            raise ValueError("Expected exactly one match for <main.py_output> and </main.py_output>, found {}".format(len(text)))
        
        text = text[0] # this should be a string representation of a list of tuples

        # parse the string into a list of tuples
        data = eval(text)
        print(data)


if __name__ == "__main__":
    # example input: /Users/arunim/Documents/github/reply/wandb/run-20241025_134316-a5n1h71p
    fpath = "/Users/arunim/Documents/github/reply/wandb/latest-run"
    main(fpath)
