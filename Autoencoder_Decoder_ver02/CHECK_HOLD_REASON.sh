#!/bin/bash
# Check why job was held and what errors occurred

JOB_ID="2650588.0"

echo "=== Job Hold Reason ==="
condor_q -hold $JOB_ID

echo ""
echo "=== Full Job Info ==="
condor_q -long $JOB_ID | grep -E "HoldReason|LastHoldReason|ExitCode|ExitBySignal"

echo ""
echo "=== Try to get output from remote node ==="
condor_tail $JOB_ID 2>&1 | tail -n 50

echo ""
echo "=== Check if output files exist in job directory ==="
JOB_DIR=$(condor_q -long $JOB_ID | grep "^Iwd" | awk '{print $3}')
echo "Job directory: $JOB_DIR"
if [ -n "$JOB_DIR" ] && [ -d "$JOB_DIR" ]; then
    echo "Files in job directory:"
    ls -lh "$JOB_DIR" 2>/dev/null | head -20
    echo ""
    echo "Logs directory:"
    ls -lh "$JOB_DIR/logs" 2>/dev/null | grep $JOB_ID || echo "No logs found"
fi

