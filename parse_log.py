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

        # count how many of these have max_score number of queries = None
        num_failed = sum(1 for og_index, max_score, num_queries in data if num_queries is None)
        print(f"Number of failed jailbreak attempts: {num_failed} / {len(data)}")

        # assert that all of these have max_score = 0
        assert all(max_score == 0 for og_index, max_score, num_queries in data if num_queries is None), "Expected all failed jailbreak attempts to have max_score = 0"

        # average the number of queries over the non-failed attempts
        avg_queries = sum(num_queries for og_index, max_score, num_queries in data if num_queries is not None) / (len(data) - num_failed)
        print(f"Average number of queries needed to jailbreak over non-failed attempts: {avg_queries}")


if __name__ == "__main__":
    # example input: /Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/run-20241025_222841-pm3y0vic
    fpath = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/latest-run"
    main(fpath)
