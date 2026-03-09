from time import process_time

from AdvancedDifferentialEvolution import AdvancedDifferentialEvolution

if __name__ == "__main__":
    start = process_time()
    ADE = AdvancedDifferentialEvolution()
    ADE.optimize()

    end = process_time()
    print(f"Total execution time: {end - start:.2f} seconds")