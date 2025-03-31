#!/bin/bash

# Convenience export, no build information and ignores prefix line
conda env export --no-builds | grep -v 'prefix*' > environment.yml
