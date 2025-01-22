#!/bin/bash

# Ask the question
echo "Do you want to hear a secret that might change your life? (y/Y to confirm)"
read -r answer

# Check the answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "Alright, here's the secret:"
    echo "The ducks in the park? They're unclaimed. Legally, you can just take them."
else
    echo "Fine, stay in the dark. Your loss."
fi
