import sys

def get_input(local=False):
    pass

def run_algo(local=False):
    filename = get_input(local)

if __name__ == "__main__":
    local = (len(sys.argv) == 2 and sys.argv[1] == "local")
    run_algo(local)