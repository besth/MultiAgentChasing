#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$pwd

NUM_TRIALS=100
NUM_SIMULATIONS=1000
NUM_SIM_LIST=(100 250 500 1000 1500 2000)
KILL_ZONE_RADIUS=(0.7 0.8)

MAX_RUNNING_STEP=1

# run with incremental max_running_steps
#{
#for i in $(seq 1 ${MAX_RUNNING_STEP})
#do
#    echo Testing with max number of steps: ${i}
#    sleep 2
#    python runMCTSInMujoco.py --num-simulations ${NUM_SIMULATIONS} --num-trials ${NUM_TRIALS} --max-running-steps ${i} &
#    if ((${i} % 10 ==0)); then wait; fi # limit to 10 concurrent subshells.
#done
#}

# run with different number of simulations
#{
#for i in "${NUM_SIM_LIST[@]}"
#do
#    echo Testing with number of simulations: ${i}
#    sleep 2
#    python runMCTSInMujoco.py --num-simulations ${i} --num-trials ${NUM_TRIALS} --max-running-steps ${MAX_RUNNING_STEP} &
#done
#}

# run with different killzone radius
{
for i in "${KILL_ZONE_RADIUS[@]}"
do
    echo Testing with killzone radius: ${i}
    sleep 1
    python runMCTSInMujoco.py --num-simulations 1000 --num-trials 25 --max-running-steps 10 --killzone-radius ${i} &
done
}