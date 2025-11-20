#!/bin/bash
# Diagnose why training failed

JOB_ID="2650588.0"

echo "=== 1. Checking for error/output files ==="
cd ~/ivf/Raffael/2025-11-19 2>/dev/null || cd ~/ivf_train 2>/dev/null || pwd

if [ -f "logs/train_${JOB_ID//./_}_0.err" ]; then
    echo "ERROR FILE FOUND:"
    cat "logs/train_${JOB_ID//./_}_0.err"
else
    echo "No .err file found"
fi

echo ""
echo "=== 2. Checking output file ==="
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    echo "OUTPUT FILE FOUND (last 100 lines):"
    tail -n 100 "logs/train_${JOB_ID//./_}_0.out"
else
    echo "No .out file found"
fi

echo ""
echo "=== 3. Checking HTCondor log for errors ==="
if [ -f "logs/train_${JOB_ID//./_}_0.log" ]; then
    echo "Errors in HTCondor log:"
    grep -i "error\|failed\|exception\|traceback\|file.*not found" "logs/train_${JOB_ID//./_}_0.log" | tail -20
fi

echo ""
echo "=== 4. Trying condor_tail ==="
condor_tail $JOB_ID 2>&1 | tail -50

echo ""
echo "=== 5. Job status ==="
condor_q -hold $JOB_ID

