#!/bin/bash
set -e  # exit if any command in the script itself fails

# Launch Python script and C program in background
python3 ../apps/hrtc.py &
pid1=$!

# Wait for both, capture exit codes
wait $pid1
status1=$?

../build/apps/hrtc &
pid2=$!

wait $pid2
status2=$?

# If both succeeded, run the second Python script
if [ $status1 -eq 0 ] && [ $status2 -eq 0 ]; then
    echo "Both finished successfully, running compare_results.py..."
    python3 ../apps/compare_results.py
else
    echo "Error: one of the programs failed (Python=$status1, C=$status2). Exiting."
    exit 1
fi
