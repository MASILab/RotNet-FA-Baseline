#!/bin/bash

# Input and output directories from command-line arguments
input_root="$1"
output_root="$2"

# Loop through all subject/session directories
for session_dir in "$input_root"/*/*/; do
    subject_name=$(basename "$(dirname "$session_dir")")
    session_name=$(basename "$session_dir")
    dwi_dir="$session_dir/dwi"
    
    # Check if dwi directory exists
    if [ -d "$dwi_dir" ]; then
        dmri_file=$(find "$dwi_dir" -name "*.mha" | head -n 1)
        json_file=$(find "$dwi_dir" -name "*.json" | head -n 1)
        
        # Ensure both required files exist
        if [[ -f "$dmri_file" && -f "$json_file" ]]; then
            output_dir="$output_root/$subject_name/$session_name/dwi"
            mkdir -p "$output_dir"

            # Skip if the output directory is not empty
            if [ -d "$output_dir" ] && [ -n "$(ls -A "$output_dir" 2>/dev/null)" ]; then
                echo "Skipping $session_name: Output directory is not empty."
                continue
            fi

            mkdir -p "$output_dir"
            
            # Run the processing script
            bash preproc_fa.sh "$output_dir" "$dmri_file" "$json_file"
        else
            echo "Skipping $session_name: Required files not found."
        fi
    else
        echo "Skipping $session_name: DWI directory not found."
    fi
done
