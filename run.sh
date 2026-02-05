#!/usr/bin/env bash
set -euo pipefail

# Run the Evident Video Fact Checker pipeline
# Usage: ./run.sh --infile inbox/transcript.txt --channel "ChannelName"
#        ./run.sh --infile inbox/transcript.txt --review

python -m app.main "$@"
