#!/bin/bash
# This script will create a secret file, output its creation details,
# and then loop for approximately 1 year while printing a progress update every second.

echo "Generating a secret number in 5 seconds..."

sleep 5

SECRET_NUMBER=$((RANDOM % 100 + 1))
SECRET_FILE="secret.txt"

echo "Secret number: $SECRET_NUMBER" > "$SECRET_FILE"

echo "Secret file '$SECRET_FILE' created with a secret inside."

echo "The script will now run a loop for approximately 1 year, printing progress every second."

SECONDS_PER_YEAR=$((365 * 24 * 3600))

for ((i=1; i<=SECONDS_PER_YEAR; i++)); do
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Progress update: $i seconds elapsed"
    sleep 1
done

echo "Loop finished. 1 year has elapsed."
