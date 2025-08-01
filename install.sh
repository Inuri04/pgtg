#!/bin/bash
# install.sh für Poetry-Projekt

# Stelle sicher, dass nur die erste Task pro Node installiert
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
    echo "Installing Poetry dependencies..."
    
    # Poetry installieren falls nicht vorhanden
    pip install --upgrade pip
    pip install poetry
    
    # Poetry konfigurieren (Virtualenv im Projekt)
    poetry config virtualenvs.in-project true
    poetry config virtualenvs.create true
    
    # Dependencies installieren
    poetry install
    
    # Andere Tasks benachrichtigen, dass Installation fertig ist
    touch "${DONEFILE}"
else
    # Warten bis Installation fertig ist
    while [[ ! -f "${DONEFILE}" ]]; do 
        sleep 1
    done
fi
