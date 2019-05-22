#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$pwd

NUM_TRIALS=25
NUM_SIMULATIONS=250

MAX_RUNNING_STEP=20

# run with incremental max_running_steps
{
for i in $(seq 1 ${MAX_RUNNING_STEP})
do
    echo Testing with max number of steps: ${i}
    sleep 2
    python runMCTSInMujoco.py --num-simulations ${NUM_SIMULATIONS} --num-trials ${NUM_TRIALS} --max-running-steps ${i} &
    if ((${i} % 10 ==0)); then wait; fi # limit to 10 concurrent subshells.
done
}