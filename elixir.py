import cupy as cp

class Elixir():
    def __init__(self, N) -> None:
        # 혼돈/질서 게이지
        self.adv_gauges = cp.zeros((N,3), dtype=cp.int8)

        self.reroll = cp.ones(N, dtype=cp.int8)*2

        self.opts = cp.zeros((N,5), dtype=cp.int8)
        self.opt_big_probs = cp.ones((N,5), dtype=cp.float32)*10
        self.opt_prob_log = cp.ones((N,14,2), dtype=cp.int8)*(-1)
        self.opt_is_avail = cp.ones((N,5),dtype=cp.bool_)

        self.enchant_avail_n = cp.ones((N), dtype=cp.int8)*14
        self.enchant_n = cp.ones((N), dtype=cp.int8)
        self.disable_left = cp.ones((N), dtype=cp.int8)*3
        self.advice_applied_n = cp.zeros((N,279), dtype=cp.int8)
        self.advice_sleeping = cp.zeros((N,3), dtype=cp.bool_)
    
    def reset(self):
        self.adv_gauges.fill(0)
        self.reroll.fill(2)
        self.opts.fill(0)
        self.opt_big_probs.fill(10)
        self.opt_prob_log.fill(-1)
        self.opt_is_avail.fill(True)
        self.enchant_avail_n.fill(14)
        self.enchant_n.fill(1)
        self.disable_left.fill(3)
        self.advice_applied_n.fill(0)
        self.advice_sleeping.fill(False)
