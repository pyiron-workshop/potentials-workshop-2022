#!/bin/bash

cd $HOME
if [[ ! -d workshop_preparation ]]; then
    git clone https://git.noc.ruhr-uni-bochum.de/potentials-workshop-2022/workshop_preparation.git
fi
cd workshop_preparation

echo "Force update of notebooks: This will overwrite your changes, backup now!";
echo "Press enter to continue: "

if read -r; then
    git checkout main -- '*'
    git pull --force
fi
