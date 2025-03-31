#!/bin/bash

# With version for posterity
conda env export > environment_$(hostname).yml

# Without build nor version number - Cross OS compatibility
conda env export --no-builds | grep -v 'prefix*' | sed 's/=.*$//' > environment.yml
