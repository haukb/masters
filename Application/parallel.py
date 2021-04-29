from Versions.mp_parallel_subproblems import MP_parallelSPs
from Versions.mp_2optimality import MP_2opt
from time import time


if __name__ == '__main__':
    #t0 = time()
    #mp_parallelSPs = MP_parallelSPs(INSTANCE='Full', NUM_VESSELS=1, MAX_PORT_VISITS=1, MAX_ITERS=80)
    #mp_parallelSPs.solve()
    #print(f'Time to solve WITH parallel SPs: {time()-t0}')
    t0 = time()
    mp_2opt = MP_2opt(INSTANCE='Full', NUM_VESSELS=1, MAX_PORT_VISITS=1, MAX_ITERS=45)
    mp_2opt.solve()
    print(f'Time to solve WITHOUT parallel SPs: {time()-t0}')