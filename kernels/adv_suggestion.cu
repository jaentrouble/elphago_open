#include "builtin_types.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

__device__ int weighedrand(
    const int n,
    const float weights[],
    curandState* seed
){
    float rand_val = 1-curand_uniform(seed);
    for (int j=0; j<n; j++){
        if (rand_val>weights[j]){
            rand_val -= weights[j];
        }
        else{
            return j;
        }
    }
    return -1;
}

__device__ char weighedrand(
    const char n,
    const float weights[],
    curandState* seed
){
    float rand_val = 1-curand_uniform(seed);
    for (char j=0; j<n; j++){
        if (rand_val>weights[j]){
            rand_val -= weights[j];
        }
        else{
            return j;
        }
    }
    return -1;
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


__device__ void opt_probs_update(
    float opt_probs[5],
    const char opt[5],
    const bool opt_is_avail[5],
    const char opt_prob_log[14][2],
    const char one_time_prob[2]
){
    bool tmp_opt_is_avail[5];
    check_max_opt(opt, opt_is_avail, tmp_opt_is_avail);

    float sum=0;
    for (int j=0; j<5; j++){
        sum += tmp_opt_is_avail[j];
    }
    // do not calculate anything if only one is left - it is 100%.
    if (sum==1){
        for (int j=0; j<5; j++){
            opt_probs[j] = tmp_opt_is_avail[j] ? 100.0 : 0.0;
        }
        return;
    }
    
    float init_prob = 100.0 / sum;
    for (int j=0; j<5; j++){
        opt_probs[j] = tmp_opt_is_avail[j] ? init_prob : 0.0;
    }

    for (int j=0; j<15; j++){
        // 0~13:full time update , 14 = one time update
        int target_idx = (j<14)? opt_prob_log[j][0]:one_time_prob[0];
        if (target_idx<0){
            continue;
        }
        float target_prob = opt_probs[target_idx];
        float update_prob = (j<14)?opt_prob_log[j][1]:one_time_prob[1];
        if ((target_prob+update_prob)>=100.0){
            for (int p=0; p<5; p++){
                opt_probs[p] = 0.0;
            }
            opt_probs[target_idx] = 100.0;
            continue;
        }
        else if ((target_prob+update_prob)<=0){
            update_prob = -target_prob;
        }
        
        for (int k=0; k<5; k++){
            if (!tmp_opt_is_avail[k]){
                continue;
            }
            else {
                opt_probs[k] = (k==target_idx)?
                        target_prob+update_prob:
                        opt_probs[k]*(1-(update_prob/(100-target_prob)));
            }
        }
    }
    
}

__device__ int get_adv_idx(
    const char opt[5],
    const bool opt_is_avail[5],
    const char adv_gauge,
    const char adv_gauge_idx,
    const int advice_list[11][279],
    const char enchant_avail_n,
    const char enchant_n,
    const char disable_left,
    const char advice_applied_n[279],
    const int already_picked[3],
    curandState* seed
){
    // 0: pickupratio
    // 1: func_type
    // 2: param1
    // 3: param2 (possible when full, impossible when disabled)
    // 4: attr_type
    // 5: applymax
    // 6: rangestart
    // 7: rangeend
    // 8: targettype
    // 9: full_pos (not used, only for checking) (possible when full, impossible when disabled)
    // 10: full_neg (not used, only for checking) (impossible when full)
    // assert(enchant_avail_n>=disable_left);
    int advice_avail_idx[279];
    int advice_pickup_cummul_ratio[279];
    int advice_avail_n=0;
    bool advice_avail[279];
    bool opt_is_full[5];
    for(int j=0; j<5; j++){
        opt_is_full[j] = (opt[j]==10);
    }

    // Check AttrType
    bool disable_turn = (enchant_avail_n==disable_left);
    for (int j=0; j<279; j++){
        switch(adv_gauge){
            case 3:
                if(disable_turn){
                    advice_avail[j] = (advice_list[4][j]==4);
                }
                else{
                    advice_avail[j] = (advice_list[4][j]==3);
                }
                break;
            case -6:
                if(disable_turn){
                    advice_avail[j] = (advice_list[4][j]==7);
                }
                else{
                    advice_avail[j] = (advice_list[4][j]==6);
                }
                break;
            default:
                if(disable_turn){
                    advice_avail[j] = (advice_list[4][j]==1);
                }
                else{
                    advice_avail[j] = (advice_list[4][j]==0);
                }
                break;
        }
    }
    // check range
    for (int j=0; j<279; j++){
        if (advice_list[6][j]==0){
            continue;
        }
        advice_avail[j] = advice_avail[j] && \
                        (advice_list[6][j]<=enchant_n && \
                        advice_list[7][j]>enchant_n);
    }
    // check applymax
    for (int j=0; j<279; j++){
        advice_avail[j] = advice_avail[j] && \
                        (advice_applied_n[j]<advice_list[5][j]);
    }
    // check already picked
    for (int j=0; j<3; j++){
        if (already_picked[j]>=0){
            advice_avail[already_picked[j]] = false;
        }
    }
    // check chaos6 'deletion' option
    switch(adv_gauge_idx){
        case 0:
            advice_avail[268]=false;
            advice_avail[269]=false;
            break;
        case 1:
            advice_avail[267]=false;
            advice_avail[269]=false;
            break;
        case 2:
            advice_avail[267]=false;
            advice_avail[268]=false;
            break;
    }
    // check impossible advice
    for (int j=0; j<279; j++){
        // check param2
        if (advice_list[3][j]>=0){
            advice_avail[j] = advice_avail[j] && opt_is_avail[advice_list[3][j]];
        }
        // check full_pos
        if (advice_list[9][j]>=0){
            advice_avail[j] = advice_avail[j] && opt_is_avail[advice_list[9][j]];
        }
        // check full_neg
        if (advice_list[10][j]>=0){
            advice_avail[j] = advice_avail[j] && \
                            (!opt_is_full[advice_list[10][j]] && \
                            opt_is_avail[advice_list[10][j]]);
        }
    }
    // changing unavail need at least one to be unavailable
    bool any_unavail = false;
    for (int j=0; j<5; j++){
        any_unavail = any_unavail || (!opt_is_avail[j]);
    }
    if (!any_unavail){
        advice_avail[261] = false;
    }
    // count available advice
    for (int j=0; j<279; j++){
        if (advice_avail[j]){
            advice_avail_idx[advice_avail_n] = j;
            advice_avail_n += 1;
        }
    }
    // pick an advice
    advice_pickup_cummul_ratio[0] = advice_list[0][advice_avail_idx[0]];
    for (int j=1; j<advice_avail_n; j++){
        advice_pickup_cummul_ratio[j] = advice_pickup_cummul_ratio[j-1] + \
                                        advice_list[0][advice_avail_idx[j]];
    }
    assert(advice_avail_n>0);
    double random_ratio;
    random_ratio = curand_uniform_double(seed) * double(advice_pickup_cummul_ratio[advice_avail_n-1]);
    for (int j=0; j<advice_avail_n; j++){
        if (random_ratio <= advice_pickup_cummul_ratio[j]){
            return advice_avail_idx[j];
        }
    }
    assert(false);
    return -1;
}

extern "C"{
    __global__ void get_state(
        const char adv_gauges[][3],
        const char opts[][5],
        const char opt_prob_log[][14][2],
        const bool opt_is_avail[][5],
        const char enchant_avail_n[],
        const char enchant_n[],
        const char disable_left[],
        const char advice_applied_n[][279],
        const bool advice_sleeping[][3],
        const int advice_list[11][279],
        float current_prob_out[][5],
        int advice_idx_out[][3],
        const unsigned long long random_seed[],
        const int N
    ){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i<N){
            curandState s;
            curand_init(random_seed[0]+i, 0, 0, &s);
            curandState* seed =&s;
            int picked_advices[3] = {-1, -1, -1};
            if (enchant_avail_n[i]>0){
                for (int j=0; j<3; j++){
                    if (!advice_sleeping[i][j]){
                        picked_advices[j] = get_adv_idx(
                            opts[i],
                            opt_is_avail[i],
                            adv_gauges[i][j],
                            j,
                            advice_list,
                            enchant_avail_n[i],
                            enchant_n[i],
                            disable_left[i],
                            advice_applied_n[i],
                            picked_advices,
                            seed
                        );
                    }
                }
            }
            for (int j=0; j<3; j++){
                advice_idx_out[i][j] = picked_advices[j];
            }


            const char one_time_prob[2] = {-1, -1};
            opt_probs_update(
                current_prob_out[i],
                opts[i],
                opt_is_avail[i],
                opt_prob_log[i],
                one_time_prob
            );
        }
    }
}