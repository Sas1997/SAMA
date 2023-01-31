from time import process_time

from swarm import Swarm

if __name__ == "__main__":
    start = process_time()

    swarm = Swarm()
    swarm.optimize()
    swarm.get_final_result(print_result=True, plot_curve=True)

    print(process_time()-start)
