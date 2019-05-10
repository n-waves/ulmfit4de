for f in /mnt/data/poleval19/smalldata/*/cleaned.csv; do python split.py $(dirname $f); done
