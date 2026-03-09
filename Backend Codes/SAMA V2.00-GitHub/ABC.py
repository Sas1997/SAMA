from time import process_time

from ArtificialBeeColony import ImprovedArtificialBeeColony

if __name__ == "__main__":
    start = process_time()
    abc = ImprovedArtificialBeeColony()
    abc.optimize()

    end = process_time()
    print(f"Total execution time: {end - start:.2f} seconds")