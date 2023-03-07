import cupy as cp
from elixir import Elixir
import math
import pandas as pd
import numpy as np
import tqdm
import random

N=10000000
ITER = 10
PLAYER_NAME = 'simple_6_bald_curve'

advice_df = pd.read_csv('etc/elixir_list.csv')
advice_list = cp.empty((11,279), dtype=cp.int32)
for i, c in enumerate(advice_df.columns[:-1]):
    advice_list[i] = cp.array(advice_df[c].values)

loaded_module = ''
with open('kernels/simulation.cu','r') as f:
    for l in f.readlines():
        loaded_module += l

simul_module = cp.RawModule(code=loaded_module, )
step = simul_module.get_function('step')

loaded_module = ''
with open('kernels/adv_suggestion.cu','r') as f:
    for l in f.readlines():
        loaded_module += l
sug_module = cp.RawModule(code=loaded_module, )
get_state = sug_module.get_function('get_state')

loaded_module = ''
with open('kernels/player.cu','r') as f:
    for l in f.readlines():
        loaded_module += l
player_module = cp.RawModule(code=loaded_module, )
player = player_module.get_function(PLAYER_NAME)

test_elixir = Elixir(N)
advice_idx_given = cp.ones((N,3),dtype=cp.int32)
advice_idx_chosen = cp.ones((N,),dtype=cp.int32)
current_prob_out = cp.empty((N,5),dtype=cp.float32)
adv_gauge_chosen_idx = cp.ones((N,),dtype=cp.int8)
param2_select = cp.ones((N,),dtype=cp.int8)
result_opt_out = np.empty((ITER*N,5),dtype=np.int8)
BLOCK_SIZE = 256
sum_55=0
for i in tqdm.trange(10):
    test_elixir.reset()
    for _ in tqdm.trange(16, leave=False):
        random_seed = cp.array((random.randint(0, 2**60),),dtype='uint64')
        advice_idx_given.fill(-1)
        get_state(
            grid=(math.ceil(N/BLOCK_SIZE),),
            block=(BLOCK_SIZE,),
            args=(
                test_elixir.adv_gauges,
                test_elixir.opts,
                test_elixir.opt_prob_log,
                test_elixir.opt_is_avail,
                test_elixir.enchant_avail_n,
                test_elixir.enchant_n,
                test_elixir.disable_left,
                test_elixir.advice_applied_n,
                test_elixir.advice_sleeping,
                advice_list,
                current_prob_out,
                advice_idx_given,
                random_seed,
                N
            )
        )

        player(
            grid=(math.ceil(N/BLOCK_SIZE),),
            block=(BLOCK_SIZE,),
            args=(
                advice_idx_given,
                test_elixir.opts,
                test_elixir.opt_is_avail,
                test_elixir.enchant_n,
                advice_idx_chosen,
                adv_gauge_chosen_idx,
                param2_select,
                random_seed,
                N
            )
        )
        # advice_idx_chosen = cp.ones((N,),dtype=cp.int32)*224
        step(
            grid=(math.ceil(N/BLOCK_SIZE),),
            block=(BLOCK_SIZE,),
            args=(
                advice_idx_chosen,
                adv_gauge_chosen_idx,
                param2_select,
                advice_list,
                test_elixir.adv_gauges,
                test_elixir.opts,
                test_elixir.enchant_avail_n,
                test_elixir.enchant_n,
                test_elixir.disable_left,
                test_elixir.advice_applied_n,
                test_elixir.advice_sleeping,
                test_elixir.opt_big_probs,
                test_elixir.opt_prob_log,
                test_elixir.opt_is_avail,
                random_seed,
                N
            )
        )
    test_elixir.opts.get(out=result_opt_out[i*N:(i+1)*N])
np.save(f'results/{PLAYER_NAME}_{ITER}_{N}.npy',result_opt_out)