#!/usr/bin/env perl
#
# submit_halos.pl -- submit one SGE job per halo for imbh.py.
#
# For each halo ID this writes a small job script (same environment setup as
# run_everything.sh, but running a single halo) into $OUTDIR/jobscripts/ and
# submits it with qsub, so every halo runs as its own job in parallel.
#
# Usage (from anywhere on the cluster):
#   perl submit_halos.pl                                  # default (simulation) mode
#   perl submit_halos.pl --mode observational             # observational mode, default settings
#   perl submit_halos.pl --mode observational --n-relation bf20_seed --radius-relation marks_kroupa12
#   perl submit_halos.pl --halos "467548 685512"          # only these halos
#   perl submit_halos.pl --label v2                       # tag outputs so older runs aren't overwritten
#   perl submit_halos.pl --dry-run                        # write + print the job scripts, don't submit
#
# Output names (per halo ID, SNAP=--snap):
#   default mode:       $OUTDIR/cluster_output_<ID>_<SNAP>[_<label>].dat
#                       $OUTDIR/cluster_tracks_<ID>[_<label>]/
#   observational mode: $OUTDIR/cluster_output_<ID>_<SNAP>_<obs tag>[_<label>].dat
#                       $OUTDIR/cluster_tracks_<ID>_<obs tag>[_<label>]/
#   with <obs tag> = obs_<n-relation>_<mass-function>_<radius-relation>_<phase-space>_host<model>_boost<n-boost>
# Without --label, default-mode names match what run_analysis.sh expects
# (and will overwrite earlier default-mode outputs in $OUTDIR).

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
    # TNG50-1-Dark subhalo IDs of the selected ellipticals
    # (from tng_download/selected_ellipticals.json); earlier sets:
    #   "685512 697044 588075 467548 665702"
    #   "801308 753345 8 745415 826784"
    'halos'           => '1235585 1117358 1136724 1044309 939095',
    'mode'            => 'default',          # default | observational
    'host-mass-model' => 'own',              # own | group
    'label'           => '',
    'seed'            => '',
    'h_rt'            => '24:00:00',
    'h_data'          => '4G',
    'radius-tracks'   => 1,                  # --no-radius-tracks to turn off
    # observational-mode settings (ignored in default mode)
    'n-relation'      => 'bf20',             # bf20 | bf20_seed | harris17_number | harris17_mass
    'mass-function'   => 'gclf',             # gclf | icmf
    'radius-relation' => 'brown_gnedin21',   # brown_gnedin21 | brown_gnedin21_young | marks_kroupa12 | const_surface_density
    'phase-space'     => 'analytic',         # analytic | simulation
    'n-boost'         => '1.0',
    'n-scatter-dex'   => '',                 # empty = the relation's own default
    'dry-run'         => 0,
);

GetOptions(\%opt,
    'model-dir=s', 'outdir=s', 'venv=s', 'snap=i', 'halos=s', 'mode=s',
    'host-mass-model=s', 'label=s', 'seed=s', 'h_rt=s', 'h_data=s',
    'radius-tracks!', 'n-relation=s', 'mass-function=s', 'radius-relation=s',
    'phase-space=s', 'n-boost=s', 'n-scatter-dex=s', 'dry-run!', 'help',
) or die "Bad options -- see the header of $0\n";
if ($opt{help}) { system('sed', '-n', '2,26p', $0); exit 0; }

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

# ---------------------------------------------------------------- naming
my $obs = $opt{mode} eq 'observational';
my $obs_tag = $obs
    ? "obs_$opt{'n-relation'}_$opt{'mass-function'}_$opt{'radius-relation'}_$opt{'phase-space'}"
      . "_host$opt{'host-mass-model'}_boost$opt{'n-boost'}"
    : '';
my $suffix = join('', map { "_$_" } grep { length } ($obs_tag, $opt{label}));
my $mode_short = $obs ? 'obs' : 'def';

my $jobdir = "$opt{outdir}/jobscripts";
make_path($jobdir) unless -d $jobdir;

# ---------------------------------------------------------------- job template
# Placeholders __LIKE_THIS__ are filled in per halo; everything else is the
# job script verbatim (so $JOB_ID etc. are expanded by SGE/bash at run time).
my $template = <<'END_TEMPLATE';
#!/bin/bash
#$ -cwd
#$ -N __JOBNAME__
# error = Merged with joblog
#$ -o __OUTDIR__/joblog.__MODE__.__ID__.$JOB_ID
#$ -j y
#$ -l h_rt=__H_RT__,h_data=__H_DATA__
#$ -pe shared 1
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea

echo "Job $JOB_ID (halo __ID__, __MODE__ mode) started on: " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

. /u/local/Modules/default/init/modules.sh
module load python

cd __MODEL_DIR__ || { echo "ERROR: cannot cd to __MODEL_DIR__"; exit 1; }
source __VENV__

ID=__ID__
CSV=subhalo_formation_${ID}.csv
if [ ! -f "$CSV" ]; then
    echo "ERROR: $CSV not found in __MODEL_DIR__"
    exit 1
fi

CMD="__CMD__"
echo $CMD
$CMD
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "WARNING: imbh.py exited with status $STATUS for halo __ID__"
fi

echo " "
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
exit $STATUS
END_TEMPLATE

# ---------------------------------------------------------------- submit
my @halos = split ' ', $opt{halos};
die "No halo IDs given\n" unless @halos;
print "Mode: $opt{mode}" . ($obs ? " ($obs_tag)" : '') . ", host-mass-model: $opt{'host-mass-model'}"
    . ($opt{label} ? ", label: $opt{label}" : '') . "\n";
print "Job scripts in $jobdir\n";

my $n_submitted = 0;
for my $id (@halos) {
    die "Halo ID '$id' is not an integer\n" unless $id =~ /^\d+$/;

    unless (-f "$opt{'model-dir'}/subhalo_formation_$id.csv") {
        warn "WARNING: $opt{'model-dir'}/subhalo_formation_$id.csv not found -- skipping halo $id\n";
        next;
    }

    my @args = (
        "python imbh.py subhalo_formation_$id.csv --output-snap $opt{snap}",
        "--save-path $opt{outdir}/cluster_output_${id}_$opt{snap}$suffix.dat",
        "--host-mass-model $opt{'host-mass-model'}",
    );
    push @args, "--save-radius-tracks --track-dir $opt{outdir}/cluster_tracks_$id$suffix"
        if $opt{'radius-tracks'};
    if ($obs) {
        push @args,
            "--cluster-sampler observational",
            "--obs-n-relation $opt{'n-relation'}",
            "--obs-mass-function $opt{'mass-function'}",
            "--obs-radius-relation $opt{'radius-relation'}",
            "--obs-phase-space $opt{'phase-space'}",
            "--obs-n-boost $opt{'n-boost'}";
        push @args, "--obs-n-scatter-dex $opt{'n-scatter-dex'}" if length $opt{'n-scatter-dex'};
    }
    push @args, "--seed $opt{seed}" if length $opt{seed};
    my $cmd = join(' ', @args);

    my $jobname = "imbh_${mode_short}_$id";   # SGE job names can't start with a digit
    my %fill = (
        JOBNAME   => $jobname,
        OUTDIR    => $opt{outdir},
        MODE      => $mode_short,
        ID        => $id,
        H_RT      => $opt{h_rt},
        H_DATA    => $opt{h_data},
        MODEL_DIR => $opt{'model-dir'},
        VENV      => $opt{venv},
        CMD       => $cmd,
    );
    (my $script = $template) =~ s/__([A-Z_]+?)__/exists $fill{$1} ? $fill{$1} : "__${1}__"/ge;

    my $path = "$jobdir/run_${mode_short}_$id$suffix.sh";
    open(my $fh, '>', $path) or die "Cannot write $path: $!\n";
    print $fh $script;
    close($fh);
    chmod 0755, $path;

    if ($opt{'dry-run'}) {
        print "\n[dry run] $path\n$cmd\n";
        next;
    }
    my $out = `qsub $path 2>&1`;
    if ($? != 0) {
        warn "ERROR submitting halo $id: $out";
        next;
    }
    print "halo $id -> $out";
    $n_submitted++;
}
print "\nSubmitted $n_submitted job(s).\n" unless $opt{'dry-run'};
