#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o /u/scratch/c/clairewi/imbh-output/joblog.obs.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=24:00:00,h_data=4G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea

# Runs imbh.py in OBSERVATIONAL cluster-sampler mode (see
# observational_cluster_sampler.py) on each selected elliptical. Outputs are
# tagged with $TAG so they never overwrite the default (simulation-sampler)
# run's cluster_output_<ID>_<SNAP>.dat / cluster_tracks_<ID>/.

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:
module load python

cd /u/home/c/clairewi/project-snaoz/imbh-model
source ~/envs/imbh-model/bin/activate

OUTDIR=/u/scratch/c/clairewi/imbh-output
SNAP=99
# TNG50-1-Dark subhalo IDs of the 5 selected ellipticals
# (from tng_download/selected_ellipticals.json)
# HALO_IDS="685512 697044 588075 467548 665702"
# HALO_IDS="801308 753345 8 745415 826784"
# HALO_IDS = "1235585 1117358 1136724 104439 939095"
HALO_IDS="1235585 1117358 1136724 1044309 939095"

# ---- Observational-mode settings (see python imbh.py --help) ----
# N_RELATION:      bf20 | bf20_seed | harris17_number | harris17_mass
# MASS_FUNCTION:   gclf | icmf
# RADIUS_RELATION: brown_gnedin21 | brown_gnedin21_young | marks_kroupa12 | const_surface_density
# PHASE_SPACE:     analytic | simulation   (simulation needs cluster_sampler.pkl via imbh_config.py)
# HOST_MASS_MODEL: own | group   (host for SATELLITE subhalos in the cluster draw AND the orbit
#                  integration: own = its own mass + its FoF group as a second background;
#                  group = legacy FoF-group M200 as the local host)
N_RELATION=bf20
MASS_FUNCTION=gclf
RADIUS_RELATION=brown_gnedin21
PHASE_SPACE=analytic
HOST_MASS_MODEL=own
N_BOOST=1.0
N_SCATTER_DEX=""   # empty = the relation's own default
SEED=""            # empty = random (imbh.py prints the seed it used)

# Label for this configuration's output files/dirs -- change it (or leave
# it built from the settings) so different observational runs don't
# overwrite each other either.
TAG=obs_${N_RELATION}_${MASS_FUNCTION}_${RADIUS_RELATION}_${PHASE_SPACE}_host${HOST_MASS_MODEL}_boost${N_BOOST}

mkdir -p $OUTDIR

for ID in $HALO_IDS; do
    CSV=subhalo_formation_${ID}.csv
    if [ ! -f "$CSV" ]; then
        echo "WARNING: $CSV not found, skipping halo $ID"
        continue
    fi

    echo " "
    echo "=== Halo $ID started on: " `date`
    CMD="python imbh.py $CSV --output-snap $SNAP \
        --save-path $OUTDIR/cluster_output_${ID}_${SNAP}_${TAG}.dat \
        --save-radius-tracks --track-dir $OUTDIR/cluster_tracks_${ID}_${TAG} \
        --cluster-sampler observational \
        --obs-n-relation $N_RELATION \
        --obs-mass-function $MASS_FUNCTION \
        --obs-radius-relation $RADIUS_RELATION \
        --obs-phase-space $PHASE_SPACE \
        --host-mass-model $HOST_MASS_MODEL \
        --obs-n-boost $N_BOOST"
    if [ -n "$N_SCATTER_DEX" ]; then
        CMD="$CMD --obs-n-scatter-dex $N_SCATTER_DEX"
    fi
    if [ -n "$SEED" ]; then
        CMD="$CMD --seed $SEED"
    fi
    echo $CMD
    $CMD || echo "WARNING: imbh.py exited with an error for halo $ID"
    echo "=== Halo $ID ended on:   " `date`
done

# echo job info on joblog:
echo " "
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
