#!/usr/bin/env perl
#
# submit_analysis.pl -- submit one SGE job per halo running analysis.py on the
# outputs written by submit_halos.pl.
#
# Takes the SAME options as submit_halos.pl (mode, observational settings,
# host-mass-model, label, ...) and rebuilds the same output names from them, so
# you can reuse the exact command line you submitted the model with:
#
#   perl submit_halos.pl    --mode observational --n-relation bf20_seed
#   perl submit_analysis.pl --mode observational --n-relation bf20_seed
#
# Options that don't affect file names (e.g. --seed) are accepted and ignored.
#
# Extra options:
#   --alpha 1.2          power-law index passed to analysis.py (default 1.2)
#   --wait-for-model     hold each analysis job until that halo's model job
#                        (job name imbh_<def|obs>_<ID>) has finished
#   --h_rt / --h_data    resources per job (default 4:00:00 / 4G)
#   --dry-run            write + print the job scripts, don't submit
#
# Reads, per halo (same names as submit_halos.pl):
#   $OUTDIR/cluster_output_<ID>_<SNAP><suffix>.dat
#   $OUTDIR/cluster_tracks_<ID><suffix>/            (if present)
# Writes analysis.py's outputs (tde_rates_<ID>_alpha<A>.csv, tde_radial_...npz,
# tde_contributors_...csv) into:
#   default mode, no label:  $OUTDIR                (same as run_analysis.sh)
#   otherwise:               $OUTDIR/analysis<suffix>/
# where <suffix> = [_<obs tag>][_<label>], so different configurations never
# overwrite each other's analysis files.

use strict;
use warnings;
use Getopt::Long;
use File::Path qw(make_path);

# ---------------------------------------------------------------- settings
my %opt = (
    'model-dir'       => '/u/home/c/clairewi/project-snaoz/imbh-model',
    'outdir'          => '/u/scratch/c/clairewi/imbh-output',
    'venv'            => '~/envs/imbh-model/bin/activate',
    'snap'            => 99,
    'halos'           => '1235585 1117358 1136724 1044309 939095',
    'mode'            => 'default',          # default | observational
    'host-mass-model' => 'own',              # own | group
    'label'           => '',
    'seed'            => '',                 # accepted for symmetry with submit_halos.pl; unused
    'alpha'           => '1.2',
    'h_rt'            => '4:00:00',
    'h_data'          => '4G',
    'radius-tracks'   => 1,                  # accepted for symmetry; tracks are used if present
    # observational-mode settings (only used to rebuild the output names)
    'n-relation'      => 'bf20',
    'mass-function'   => 'gclf',
    'radius-relation' => 'brown_gnedin21',
    'phase-space'     => 'analytic',
    'n-boost'         => '1.0',
    'n-scatter-dex'   => '',                 # not part of the names; accepted and ignored
    'wait-for-model'  => 0,
    'dry-run'         => 0,
);

GetOptions(\%opt,
    'model-dir=s', 'outdir=s', 'venv=s', 'snap=i', 'halos=s', 'mode=s',
    'host-mass-model=s', 'label=s', 'seed=s', 'alpha=s', 'h_rt=s', 'h_data=s',
    'radius-tracks!', 'n-relation=s', 'mass-function=s', 'radius-relation=s',
    'phase-space=s', 'n-boost=s', 'n-scatter-dex=s', 'wait-for-model!', 'dry-run!', 'help',
) or die "Bad options -- see the header of $0\n";
if ($opt{help}) { system('sed', '-n', '2,31p', $0); exit 0; }

my %allowed = (
    'mode'            => [qw(default observational)],
    'host-mass-model' => [qw(own group)],
    'n-relation'      => [qw(bf20 bf20_seed harris17_number harris17_mass)],
    'mass-function'   => [qw(gclf icmf)],
    'radius-relation' => [qw(brown_gnedin21 brown_gnedin21_young marks_kroupa12 const_surface_density)],
    'phase-space'     => [qw(analytic simulation)],
);
for my $k (sort keys %allowed) {
    die "--$k must be one of: @{$allowed{$k}} (got '$opt{$k}')\n"
        unless grep { $_ eq $opt{$k} } @{$allowed{$k}};
}
die "--label may only contain letters, digits, '.', '_' or '-'\n" if $opt{label} =~ /[^\w.\-]/;
die "--alpha must be a number\n" unless $opt{alpha} =~ /^[0-9]*\.?[0-9]+$/;

# ---------------------------------------------------------------- naming
# MUST match submit_halos.pl exactly.
my $obs = $opt{mode} eq 'observational';
my $obs_tag = $obs
    ? "obs_$opt{'n-relation'}_$opt{'mass-function'}_$opt{'radius-relation'}_$opt{'phase-space'}"
      . "_host$opt{'host-mass-model'}_boost$opt{'n-boost'}"
    : '';
my $suffix = join('', map { "_$_" } grep { length } ($obs_tag, $opt{label}));
my $mode_short = $obs ? 'obs' : 'def';

my $analysis_dir = length($suffix) ? "$opt{outdir}/analysis$suffix" : $opt{outdir};
my $jobdir = "$opt{outdir}/jobscripts";
make_path($jobdir) unless -d $jobdir;

# ---------------------------------------------------------------- job template
my $template = <<'END_TEMPLATE';
#!/bin/bash
#$ -cwd
#$ -N __JOBNAME__
# error = Merged with joblog
#$ -o __OUTDIR__/joblog.analysis.__MODE__.__ID__.$JOB_ID
#$ -j y
#$ -l h_rt=__H_RT__,h_data=__H_DATA__
#$ -pe shared 1
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea

echo "Job $JOB_ID (analysis, halo __ID__, __MODE__ mode) started on: " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

. /u/local/Modules/default/init/modules.sh
module load python

cd __MODEL_DIR__ || { echo "ERROR: cannot cd to __MODEL_DIR__"; exit 1; }
source __VENV__

ID=__ID__
DAT=__DAT__
TRACKDIR=__TRACKDIR__
ANALYSIS_DIR=__ANALYSIS_DIR__

if [ ! -f "$DAT" ]; then
    echo "ERROR: $DAT not found (did the model job for halo $ID finish?)"
    exit 1
fi
mkdir -p "$ANALYSIS_DIR"

CMD="python analysis.py $DAT __ALPHA__ --subhalo-id $ID --output-dir $ANALYSIS_DIR"
if [ -d "$TRACKDIR" ]; then
    CMD="$CMD --track-dir $TRACKDIR"
else
    echo "NOTE: $TRACKDIR not found -- running without radius tracks"
fi
echo $CMD
$CMD
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "WARNING: analysis.py exited with status $STATUS for halo $ID"
fi

echo " "
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
exit $STATUS
END_TEMPLATE

# ---------------------------------------------------------------- submit
my @halos = split ' ', $opt{halos};
die "No halo IDs given\n" unless @halos;
print "Analysis for mode: $opt{mode}" . ($obs ? " ($obs_tag)" : '')
    . ", host-mass-model: $opt{'host-mass-model'}" . ($opt{label} ? ", label: $opt{label}" : '')
    . ", alpha: $opt{alpha}\n";
print "Analysis outputs -> $analysis_dir\n";
print "Job scripts in $jobdir\n";

my $n_submitted = 0;
for my $id (@halos) {
    die "Halo ID '$id' is not an integer\n" unless $id =~ /^\d+$/;

    my $dat      = "$opt{outdir}/cluster_output_${id}_$opt{snap}$suffix.dat";
    my $trackdir = "$opt{outdir}/cluster_tracks_$id$suffix";
    my $model_job = "imbh_${mode_short}_$id";

    # Without --wait-for-model the model output must already exist.
    if (!$opt{'wait-for-model'} && !-f $dat) {
        warn "WARNING: $dat not found -- skipping halo $id "
           . "(use --wait-for-model if its model job is still queued/running)\n";
        next;
    }

    my %fill = (
        JOBNAME      => "ana_${mode_short}_$id",
        OUTDIR       => $opt{outdir},
        MODE         => $mode_short,
        ID           => $id,
        H_RT         => $opt{h_rt},
        H_DATA       => $opt{h_data},
        MODEL_DIR    => $opt{'model-dir'},
        VENV         => $opt{venv},
        DAT          => $dat,
        TRACKDIR     => $trackdir,
        ANALYSIS_DIR => $analysis_dir,
        ALPHA        => $opt{alpha},
    );
    (my $script = $template) =~ s/__([A-Z][A-Z_]*?)__/exists $fill{$1} ? $fill{$1} : "__${1}__"/ge;
    die "Internal error: unfilled placeholder $1 in job script for halo $id\n"
        if $script =~ /__([A-Z][A-Z_]*)__/;

    my $path = "$jobdir/analysis_${mode_short}_$id$suffix.sh";
    open(my $fh, '>', $path) or die "Cannot write $path: $!\n";
    print $fh $script;
    close($fh);
    chmod 0755, $path;

    my @qsub = ('qsub');
    push @qsub, '-hold_jid', $model_job if $opt{'wait-for-model'};
    push @qsub, $path;

    if ($opt{'dry-run'}) {
        print "\n[dry run] @qsub\n";
        next;
    }
    my $cmdline = join(' ', @qsub);
    my $out = `$cmdline 2>&1`;
    if ($? != 0) {
        warn "ERROR submitting analysis for halo $id: $out";
        next;
    }
    print "halo $id -> $out";
    $n_submitted++;
}
print "\nSubmitted $n_submitted job(s).\n" unless $opt{'dry-run'};
