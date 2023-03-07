#include "builtin_types.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

__device__ int randint(
    int low,
    int high,
    curandState* seed
){
    float rand_val = 1-curand_uniform(seed);
    int out = low + (high -low)*rand_val;
    return out;
}
__device__ char randint(
    char low,
    char high,
    curandState* seed
){
    float rand_val = 1-curand_uniform(seed);
    char out = low + (high -low)*rand_val;
    return out;
}

__device__ void check_max_opt(
    const char opt[5],
    const bool opt_is_avail[5],
    bool tmp_opt_is_avail[5]
){
    memcpy(tmp_opt_is_avail, opt_is_avail, sizeof(bool)*5);
    for (int j=0; j<5; j++){
        if (opt[j]==10){
            tmp_opt_is_avail[j] = false;
        }
    }
}

extern "C"{
    __global__ void random_player(
        const int advice_idx_input[][3],
        const char opts[][5],
        const bool opt_is_avail[][5],
        int advice_idx_output[],
        char adv_gauge_idx_output[],
        char param2_select_output[],
        const unsigned long long random_seed[],
        const int N
    ){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < N){
            curandState s;
            curand_init(random_seed[0]+i, 0, 0, &s);
            curandState* seed = &s;

            bool opt_is_avail_and_not_full[5];
            check_max_opt(opts[i], opt_is_avail[i], opt_is_avail_and_not_full);
            char avail_not_full_count = 0;
            param2_select_output[i] = 0;
            for (int j=0; j<5; j++){
                if (opt_is_avail_and_not_full[j]){
                    avail_not_full_count++;
                    param2_select_output[i] = j;
                    break;
                }
            }
            if (avail_not_full_count==0){
                for (int j=0; j<5; j++){
                    if (opt_is_avail[i][j]){
                        param2_select_output[i] = j;
                        break;
                    }
                }
            }
            adv_gauge_idx_output[i] = randint(0, 3, seed);
            advice_idx_output[i] = advice_idx_input[i][adv_gauge_idx_output[i]];          
        }
    }
}