from time import process_time
from Input_Data import InData
Run_Time = InData.Run_Time
from GreyWolfOptimizer import GreyWolfOptimizer, ParallelGreyWolfOptimizer

if __name__ == "__main__":
    USE_PARALLEL = True  # Set to False for regular implementation
    start = process_time()
    if USE_PARALLEL and Run_Time > 1:
        gwo = ParallelGreyWolfOptimizer()
        gwo.optimize_parallel()
    else:
        gwo = GreyWolfOptimizer()
        gwo.optimize()

    print(process_time()-start, "Total execution time [Sec]")