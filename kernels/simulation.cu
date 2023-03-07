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
__device__ char randchar(
    char low,
    char high,
    curandState* seed
){
    float rand_val = 1-curand_uniform(seed);
    char out = low + (high -low)*rand_val;
    return out;
}

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
__device__ void randomsample(
    int samples,
    int n,
    int avail_n[],
    int output[],
    curandState* seed
){
    int tmp_idx;
    for(int j=0;j<samples;j++){
        tmp_idx = randint(0, n-j, seed);
        output[j] = avail_n[tmp_idx];
        for(int k=tmp_idx;k<n-1;k++){
            avail_n[k] = avail_n[k+1];
        }
    }
}
__device__ void randomsample(
    int samples,
    int n,
    char avail_n[],
    char output[],
    curandState* seed
){
    int tmp_idx;
    for(int j=0;j<samples;j++){
        tmp_idx = randint(0, n-j, seed);
        output[j] = avail_n[tmp_idx];
        for(int k=tmp_idx;k<n-1;k++){
            avail_n[k] = avail_n[k+1];
        }
    }
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

// Advice no.1
__device__ void opt_updown(
    char opt[5],
    const bool opt_is_avail[5],
    const char param1,
    const char param2,
    curandState* seed
){
    float rand_val = 1 - curand_uniform(seed);

    char avail_indices[5];
    int avail_n = 0;
    for (int j=0; j<5; j++){
        if (opt_is_avail[j]){
            avail_indices[avail_n] = j;
            avail_n += 1;
        }
    }
    // avail_indices shuffled
    int rand_idx;
    int tmp_idx;
    for (int j=0; j<(avail_n-1); j++){
        rand_idx = randint(j, avail_n, seed);
        tmp_idx = avail_indices[j];
        avail_indices[j] = avail_indices[rand_idx];
        avail_indices[rand_idx] = tmp_idx;
    }

    int max_avail_indices[5] = {0};
    int max_idx_idx = 0;
    int max_idx=0;
    for (int j=1; j<avail_n; j++){
        if (opt[avail_indices[j]]>opt[avail_indices[max_idx_idx]]){
            max_idx = avail_indices[j];
            max_idx_idx = j;
        }
    }
    int max_other_idx = avail_indices[(max_idx_idx+1)%avail_n];

    // same as above, but min instead max
    int min_avail_indices[5] = {0};
    int min_idx_idx = 0;
    int min_idx=0;
    for (int j=1; j<avail_n; j++){
        if (opt[avail_indices[j]]<opt[avail_indices[min_idx_idx]]){
            min_idx = avail_indices[j];
            min_idx_idx = j;
        }
    }
    int min_other_idx = avail_indices[(min_idx_idx+1)%avail_n];

    int random_opt_idx = avail_indices[0];

    switch(param1){
        case 0:
            opt[param2] += randchar(-2, 3, seed);
            break;
        case 1:
            opt[param2] += randchar(-1, 3, seed);
            break;
        case 4:
            opt[param2] += randchar(0, 5, seed);
            break;
        case 5:
            opt[param2] += randchar(2, 4, seed);
            break;
        case 6:
            opt[param2] += randchar(-4, 6, seed);
            break;
        case 7:
            if (rand_val < 0.25){
                opt[param2] += 1;
            }
            break;
        case 8:
            if (rand_val < 0.5){
                opt[param2] += 1;
            }
            break;
        case 10:
            opt[random_opt_idx] += 1;
            break;
        case 11:
            opt[random_opt_idx] += 2;
            break;
        case 12:
            opt[max_idx] += 1;
            break;
        case 13:
            opt[max_idx] += 2;
            break;
        case 14:
            opt[min_idx] += 1;
            break;
        case 80:
            opt[min_idx] += 2;
            break;

        case 81:
            for (int j=0; j<5; j++){
                if (opt[j]==0){
                    opt[j] += 1;
                }
            }
            break;
        case 15:
            for (int j=0; j<5; j++){
                if (opt[j]<=2){
                    opt[j] += 1;
                }
            }
            break;
        case 16:
            for (int j=0; j<5; j++){
                if (opt[j]<=4){
                    opt[j] += 1;
                }
            }
            break;
        case 17:
            for (int j=0; j<5; j++){
                if (opt[j]<=6){
                    opt[j] += 1;
                }
            }
            break;
        case 18:
            opt[param2] += 2;
            break;
        case 19:
            opt[param2] += 1;
            break;
        case 20:
            opt[param2] += 2;
            break;
        case 21:
            opt[0] += 1;
            opt[2] += 1;
            opt[4] += 1;
            opt[1] -= 2;
            opt[3] -= 2;
            break;
        case 22:
            opt[0] -= 2;
            opt[2] -= 2;
            opt[4] -= 2;
            opt[1] += 1;
            opt[3] += 1;
            break;

        case 30:
        case 31:
        case 32:
        case 33:
        case 34:
            opt[param2] += 1;
            opt[param1-30] -= 1;
            break;
        case 35:
            opt[min_idx] += 1;
            opt[max_idx] -= 1;
            break;
        case 36:
            opt[min_idx] += 1;
            opt[min_other_idx] -= 1;
            break;
        case 37:
            opt[max_idx] += 1;
            opt[min_idx] -= 1;
            break;
        case 38:
            opt[max_idx] += 1;
            opt[max_other_idx] -= 1;
            break;

        case 40:
        case 41:
        case 42:
        case 43:
        case 44:
            opt[param2] += 1;
            opt[param1-40] -= 2;
            break;
        case 45:
            opt[min_idx] += 1;
            opt[max_idx] -= 2;
            break;
        case 46:
            opt[min_idx] += 1;
            opt[min_other_idx] -= 2;
            break;
        case 47:
            opt[max_idx] += 1;
            opt[min_idx] -= 2;
            break;
        case 48:
            opt[max_idx] += 1;
            opt[max_other_idx] -= 2;
            break;

        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
            opt[param2] += 2;
            opt[param1-50] -= 1;
            break;
        case 55:
            opt[min_idx] += 2;
            opt[max_idx] -= 1;
            break;
        case 56:
            opt[min_idx] += 2;
            opt[min_other_idx] -= 1;
            break;

        case 60:
        case 61:
        case 62:
        case 63:
        case 64:
            opt[param2] += 2;
            opt[param1-60] -= 2;
            break;
        case 65:
            opt[min_idx] += 2;
            opt[max_idx] -= 2;
            break;
        case 66:
            opt[min_idx] += 2;
            opt[min_other_idx] -= 2;
            break;
        
        case 70:
            opt[param2] = randchar(1, 3, seed);
            break;
        case 71:
            opt[param2] = randchar(2, 4, seed);
            break;
        case 72:
            opt[param2] = randchar(3, 5, seed);
            break;
        case 73:
            opt[param2] = randchar(5, 7, seed);
            break;
    }
    for (int j=0;j<5;j++){
        opt[j] = max(opt[j],0);
        opt[j] = min(opt[j],10);
    }
}

// Advice no.2
__device__ void opt_prob(
    char opt_prob_log[14][2],
    char one_time_prob[2],
    const char param1,
    const char param2,
    curandState* seed
){
    char idx = 0;
    for (int j=0; j<14; j++){
        if (opt_prob_log[j][0]==-1){
            idx = j;
            break;
        }
    }

    char one_time_adv_probs[5] = {100,70,35,-20,-40};
    char always_adv_probs[6] = {10,5,-5,-10,15,-20};

    switch(param1){
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
            one_time_prob[0] = param2;
            one_time_prob[1] = one_time_adv_probs[param1];
            break;
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
            opt_prob_log[idx][0] = param2;
            opt_prob_log[idx][1] = always_adv_probs[param1-10];
            break;
    }
}

// Advice no.3
__device__ void opt_big_prob(
    float opt_big_probs[5],
    float current_opt_big_probs[5],
    const char param1,
    const char param2,
    curandState* seed
){
    switch(param1){
        case 0:
            for (int j=0; j<5; j++){
                current_opt_big_probs[j] += 60;
            }
            break;
        case 1:
            for (int j=0; j<5; j++){
                current_opt_big_probs[j] += 30;
            }
            break;
        case 2:
            current_opt_big_probs[param2] += 100;
            break;

        case 10:
            current_opt_big_probs[param2] += 25;
            opt_big_probs[param2] += 25;
            break;
        case 11:
            current_opt_big_probs[param2] += 15;
            opt_big_probs[param2] += 15;
            break;
        case 12:
            current_opt_big_probs[param2] += 7;
            opt_big_probs[param2] += 7;
            break;
        case 13:
            for (int j=0; j<5; j++){
                current_opt_big_probs[j] += 5;
                opt_big_probs[j] += 5;
            }
            break;
        case 14:
            for (int j=0; j<5; j++){
                current_opt_big_probs[j] += 10;
                opt_big_probs[j] += 10;
            }
            break;
        case 15:
            current_opt_big_probs[1] += 15;
            current_opt_big_probs[3] += 15;
            opt_big_probs[1] += 15;
            opt_big_probs[3] += 15;
            break;
        case 16:
            current_opt_big_probs[0] += 15;
            current_opt_big_probs[2] += 15;
            current_opt_big_probs[4] += 15;
            opt_big_probs[0] += 15;
            opt_big_probs[2] += 15;
            opt_big_probs[4] += 15;
            break;
        case 17:
            for (int j=0; j<5; j++){
                current_opt_big_probs[j] += 15;
                opt_big_probs[j] += 15;
            }
            break;
    }
}

// Advice no.4
__device__ void opt_N(
    char* enchant_type,
    char one_time_prob[2],
    const char param1,
    const char param2
){
    switch(param1){
        case 0:
            *enchant_type = 1;
            break;
        case 1:
        case 2:
            *enchant_type = 2;
            break;
        case 3:
            *enchant_type = 3;
            break;
        case 4:
        case 5:
            *enchant_type = 4;
            break;
        case 6:
        case 7:
            one_time_prob[0] = param2;
            one_time_prob[1] = 100;
            *enchant_type = 3;
            break;
    }
}

// Advice no.5
__device__ void opt_swap(
    char opt[5],
    bool opt_is_avail[5],
    const char param1,
    const char param2,
    curandState* seed
){
    assert(param1!=param2);
    char tmp_opt_list[5] = {0};
    char tmp_opt;
    int tmp_idx;
    char tmp_idx_list[5] = {0};
    char all_opt_idx[50] = {0};
    int all_opt_n = 0;
    int space_count = 0;

    char max_indices[5] = {0};
    int max_idx = 0;
    int max_idx_n = 0;
    for (int j=1; j<5; j++){
        if (opt[j]>opt[max_idx]){
            max_idx = j;
        }
    }
    for (int j=0; j<5; j++){
        if (opt[j]==opt[max_idx]){
            max_indices[max_idx_n] = j;
            max_idx_n += 1;
        }
    }
    max_idx = max_indices[randint(0, max_idx_n, seed)];

    char min_indices[5] = {0};
    int min_idx = 0;
    int min_idx_n = 0;
    for (int j=1; j<5; j++){
        if (opt[j]<opt[min_idx]){
            min_idx = j;
        }
    }
    for (int j=0; j<5; j++){
        if (opt[j]==opt[min_idx]){
            min_indices[min_idx_n] = j;
            min_idx_n += 1;
        }
    }
    min_idx = min_indices[randint(0, min_idx_n, seed)];

    int avail_n = 0;
    int unavail_n = 0;
    int avail_indices[5] = {0};
    int unavail_indices[5] = {0};
    for (int j=0; j<5; j++){
        if (opt_is_avail[j]){
            avail_indices[avail_n] = j;
            avail_n += 1;
        } else{
            unavail_indices[space_count] = j;
            unavail_n += 1;
        }
    }

    switch(param1){
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
            opt[param2] -= 1;
            tmp_opt = opt[param2];
            opt[param2] = opt[param1];
            opt[param1] = tmp_opt;
            break;
        case 5:
            //like case 4, but with min and max
            opt[max_idx] -= 1;
            tmp_opt = opt[max_idx];
            opt[max_idx] = opt[min_idx];
            opt[min_idx] = tmp_opt;
            break;
        
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
            tmp_opt = opt[param2];
            opt[param2] = opt[param1-10];
            opt[param1-10] = tmp_opt;
            break;
        case 15:
            tmp_opt = opt[max_idx];
            opt[max_idx] = opt[min_idx];
            opt[min_idx] = tmp_opt;
            break;

        case 20:
            for (int j=0; j<avail_n; j++){
                tmp_idx_list[j] = avail_indices[j];
            }
            for (int j=0; j<(avail_n-1); j++){
                tmp_idx = randint(j, avail_n, seed);
                tmp_opt = tmp_idx_list[j];
                tmp_idx_list[j] = tmp_idx_list[tmp_idx];
                tmp_idx_list[tmp_idx] = tmp_opt;
            }
            for (int j=0; j<avail_n; j++){
                tmp_opt_list[j] = opt[tmp_idx_list[j]];
            }
            for (int j=0; j<avail_n; j++){
                opt[avail_indices[j]] = tmp_opt_list[j];
            }
            break;
        case 21:
            for (int j=0; j<5; j++){
                if (opt_is_avail[j]){
                    all_opt_n += opt[j];
                }
            }
            for (int j=0; j<avail_n; j++){
                for (int k=0; k<10;k++){
                    all_opt_idx[j*10+k] = avail_indices[j];
                }
            }
            for (int j=0; j<(avail_n*10-1); j++){
                tmp_idx = randint(j, avail_n*10, seed);
                tmp_opt = all_opt_idx[j];
                all_opt_idx[j] = all_opt_idx[tmp_idx];
                all_opt_idx[tmp_idx] = tmp_opt;
            }
            assert(all_opt_n <= avail_n*10);

            for (int j=0; j<5; j++){
                if (opt_is_avail[j]){
                    opt[j] = 0;
                }
            }
            for (int j=0; j<all_opt_n; j++){
                opt[all_opt_idx[j]] += 1;
            }
            break;
        case 22:
            tmp_opt = opt[4];
            for (int j=4; j>0; j--){
                opt[j] = opt[j-1];
            }
            opt[0] = tmp_opt;
            break;
        case 23:
            tmp_opt = opt[0];
            for (int j=0; j<4; j++){
                opt[j] = opt[j+1];
            }
            opt[4] = tmp_opt;
            break;
        case 24:
            int target_idx = param2;
            if (param2==5){
                target_idx = max_idx;
            }
            for (int j=0; j<5; j++){
                if (opt_is_avail[j] && (j!=target_idx)){
                    for (int k=0; k<(10-opt[j]); k++){
                        all_opt_idx[space_count] = j;
                        space_count += 1;
                    }
                }
            }
            char to_split;
            to_split = min(opt[target_idx], space_count);
            opt[target_idx] = max(to_split-space_count,0);
            for (int j=0; j<(space_count-1); j++){
                tmp_idx = randint(j, space_count, seed);
                tmp_opt = all_opt_idx[j];
                all_opt_idx[j] = all_opt_idx[tmp_idx];
                all_opt_idx[tmp_idx] = tmp_opt;
            }
            for (int j=0; j<to_split; j++){
                opt[all_opt_idx[j]] += 1;
            }
            break;
        case 25:
            assert(unavail_n>0);
            assert(avail_n>0);
            int to_move = randint(0, unavail_n, seed);
            int to_move_to = randint(0, avail_n, seed);
            opt_is_avail[unavail_indices[to_move]] = true;
            opt_is_avail[avail_indices[to_move_to]] = false;
            break;

    }

    for (int j=0;j<5;j++){
        opt[j] = max(opt[j],0);
        opt[j] = min(opt[j],10);
    }
}

// Advice no.6
__device__ void opt_disable(
    char opt[5],
    bool opt_is_avail[5],
    char* enchant_type,
    char one_time_prob[2],
    const char param1,
    const char param2,
    curandState* seed
){
    char avail_indices[5];
    int avail_n = 0;
    for (int j=0; j<5; j++){
        if (opt_is_avail[j]){
            avail_indices[avail_n] = j;
            avail_n += 1;
        }
    }
    int target_idx;
    switch(param1){
        case 0:
            opt_is_avail[param2] = false;
            break;
        
        case 10:
            opt_is_avail[param2] = false;
            opt_N(
                enchant_type,
                one_time_prob,
                0,
                param2
            );
            break;
        case 11:
            opt_is_avail[param2] = false;
            opt_N(
                enchant_type,
                one_time_prob,
                3,
                param2
            );
            break;
        case 12:
            opt_is_avail[param2] = false;
            break;
        
        case 20:
            opt_is_avail[param2] = false;
            opt_swap(
                opt,
                opt_is_avail,
                20,
                param2,
                seed
            );
            break;
        case 21:
            opt_is_avail[param2] = false;
            opt_swap(
                opt,
                opt_is_avail,
                21,
                param2,
                seed
            );
            break;
        case 22:
            opt_is_avail[param2] = false;
            opt_updown(
                opt,
                opt_is_avail,
                10,
                param2,
                seed
            );
            break;
        case 23:
            opt_is_avail[param2] = false;
            opt_updown(
                opt,
                opt_is_avail,
                13,
                param2,
                seed
            );
            break;

        case 30:
            target_idx = randint(0, avail_n, seed);
            opt_is_avail[avail_indices[target_idx]] = false;
            break;
        case 31:
        case 32:
            opt_is_avail[param2] = false;
            break;
    }
}

__device__ void adv_gauge_one_update(
    char* adv_gauge,
    const bool chosen
){
    if ((*adv_gauge==3) || (*adv_gauge==-6)) *adv_gauge = 0;
    if (chosen) {
        *adv_gauge = (*adv_gauge<0) ? 1 : (*adv_gauge+1);
    }
    else{
        *adv_gauge = (*adv_gauge>0) ? -1 : (*adv_gauge-1);
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

__device__ void enchant(
    char opt[5],
    const char opt_prob_log[14][2],
    const char one_time_prob[2],
    const bool opt_is_avail[5],
    const float current_opt_big_probs[5],
    const char enchant_type,
    curandState* seed
){
    bool two_up = enchant_type==3;
    bool three_up = enchant_type==4;
    float opt_rand;
    float opt_big_rand;
    int enchant_N = 1;
    if (enchant_type==1){
        enchant_N = 2;
    } else if (enchant_type==2){
        enchant_N = 3;
    }
    bool tmp_opt_is_avail[5];
    check_max_opt(opt, opt_is_avail, tmp_opt_is_avail);
    int count_avail = 0;
    for (int j=0; j<5; j++){
        if (tmp_opt_is_avail[j]) count_avail++;
    }
    enchant_N = min(enchant_N, count_avail);

    float tmp_probs[5];
    for (int e=0; e<enchant_N; e++){
        opt_rand = (1-curand_uniform(seed))*100;
        opt_big_rand = (1-curand_uniform(seed))*100;
        opt_probs_update(
            tmp_probs,
            opt,
            tmp_opt_is_avail,
            opt_prob_log,
            one_time_prob
        );
    
        for (int j = 0; j<5; j++){
            if (opt_rand > tmp_probs[j]){
                opt_rand -= tmp_probs[j];
            }
            else{
                if (opt_big_rand<current_opt_big_probs[j]){
                    opt[j] += 2;
                }
                else{
                    opt[j] += 1;
                }
                if (two_up){
                    opt[j] += 1;
                } else if (three_up){
                    opt[j] += 2;
                }
                opt[j] = min(opt[j], 10);
                tmp_opt_is_avail[j] = false;
                break;
            }
        }
    }
}


__device__ void adv_gauge_update(
    char adv_gauge[3],
    char advice_gauge_idx
){
    adv_gauge_one_update(adv_gauge + advice_gauge_idx, true);
    adv_gauge_one_update(adv_gauge + ((advice_gauge_idx+1)%3), false);
    adv_gauge_one_update(adv_gauge + ((advice_gauge_idx+2)%3), false);
}

extern "C"{
    __global__ void step(
        const int advice_idx[],
        const char adv_gauge_idx[],
        const char param2_select[],
        const int advice_list[11][279],
        char adv_gauges[][3],
        char opts[][5],
        char enchant_avail_n[],
        char enchant_n[],
        char disable_left[],
        char advice_applied_n[][279],
        bool advice_sleeping[][3],
        float opt_big_probs[][5],
        char opt_prob_log[][14][2],
        bool opt_is_avail[][5],
        const unsigned long long random_seed[],
        const int N
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
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<N){
            // Don't do anything if no chance is left
            assert(enchant_avail_n[i]>=0);
            if (enchant_avail_n[i]==0){
                return;
            }
            // Random init
            curandState s;
            curand_init(random_seed[0]+i, 0, 0, &s);
            curandState* seed = &s;

            // Temporary prob
            float current_big_probs[5];
            memcpy(current_big_probs, opt_big_probs[i], sizeof(float)*5);

            char one_time_prob[2] = {-1,-1};

            // Advice step
            char advice_func_type = advice_list[1][advice_idx[i]];
            char param1 = advice_list[2][advice_idx[i]];
            char param2 = advice_list[3][advice_idx[i]];
            if (param2==-1){
                param2 = param2_select[i];
            }
            char enchant_type = 0;
            switch(advice_func_type){
                case 0:
                    break;
                case 1:
                    opt_updown(
                        opts[i],
                        opt_is_avail[i],
                        param1,
                        param2,
                        seed
                    );
                    break;
                case 2:
                    opt_prob(
                        opt_prob_log[i],
                        one_time_prob,
                        param1,
                        param2,
                        seed
                    );
                    break;
                case 3:
                    opt_big_prob(
                        opt_big_probs[i],
                        current_big_probs,
                        param1,
                        param2,
                        seed
                    );
                    break;
                case 4:
                    opt_N(
                        &enchant_type,
                        one_time_prob,
                        param1,
                        param2
                    );
                    break;
                case 5:
                    opt_swap(
                        opts[i],
                        opt_is_avail[i],
                        param1,
                        param2,
                        seed
                    );
                    break;
                case 6:
                    opt_disable(
                        opts[i],
                        opt_is_avail[i],
                        &enchant_type,
                        one_time_prob,
                        param1,
                        param2,
                        seed                       
                    );
                    disable_left[i] -= 1;
                    break;
            }
        
            // Advice gauge update
            adv_gauge_update(adv_gauges[i], adv_gauge_idx[i]);

            // Enchant
            enchant(
                opts[i],
                opt_prob_log[i],
                one_time_prob,
                opt_is_avail[i],
                current_big_probs,
                enchant_type,
                seed
            );
            // unconsuming enchants
            switch(advice_idx[i]){
                // consume 2 enchant chances
                case 54:
                case 55:
                case 56:
                case 57:
                case 58:
                case 59:
                case 61:
                case 63:
                case 225:
                case 226:
                case 227:
                case 228:
                case 229:
                case 230:
                    enchant_avail_n[i] -= 1;
                    break;
                case 247:
                case 257:
                    enchant_avail_n[i] += 1;
                    break;
            }
            // sleeping enchants
            switch(advice_idx[i]){
                case 266:
                    advice_sleeping[i][0] = true;
                    break;
                case 267:
                    advice_sleeping[i][1] = true;
                    break;
                case 268:
                    advice_sleeping[i][2] = true;
                    break;
            }
            if (enchant_avail_n[i]>0){
                enchant_avail_n[i] -= 1;
            }
            enchant_n[i] += 1;
            advice_applied_n[i][advice_idx[i]] += 1;
        }    
    }
}