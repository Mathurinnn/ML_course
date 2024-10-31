import sys
import run
import importlib

part1Cache = None
if __name__ == "__main__":
    while True:
        if not part1Cache:
            print("Running part1")
            part1Cache = run.part1()
        try:
            print("Running part2")
            run.part2(part1Cache)
        except Exception as e:
            print("An error occurred: ", e)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(run)