# Application of clustering method to SARS-CoV-2 outbreaks in Boston between 4 March and 9 May 2020

Sequences and metadata from [Lemieux et al., (2021)](https://www.science.org/doi/10.1126/science.abe3261) was downloaded from the project's [Terra Workspace](https://app.terra.bio/#workspaces/pathogen-genomic-surveillance/COVID-19_Broad_Viral_NGS).

## Sequence analysis with [`Nextclade`](https://docs.nextstrain.org/projects/nextclade/en/stable/index.html)

In order to assign clades, Nextclade places sequences on a reference tree that is representative of the global phylogeny (see figure below). The query sequence (dashed) is compared to all sequences (including internal nodes) of the reference tree to identify the nearest neighbor.

[Nextclade Result Table Columns Meaning](https://docs.nextstrain.org/projects/nextclade/en/stable/user/output-files/04-results-tsv.html)

STEP 1
```aiignore
nextclade dataset get --name "nextstrain/sars-cov-2/wuhan-hu-1/orfs" --output-dir "sars-cov-2"
```

STEP 2
```aiignore
nextclade run \
    --input-dataset "sars-cov-2" \
    --output-fasta "MGH_DPH_98percent_772samples_aligned.fasta" \
    --output-tsv "MGH_DPH_98percent_772samples_nextclade.tsv" \
    --include-reference false \
    "MGH_DPH_98percent_772samples.fasta"
```

## Computation of pairwise genetic and temporal distances, and probabilities

Pairwise temporal distances were calculated as the absolute difference between the sampling dates. The aligned sequences outputted by Nextclade were utilised to calculate pairwise genetic distances using a standalone package [tn93](https://github.com/veg/tn93) that implements the [TN93 substitution model](https://academic.oup.com/mbe/article/10/3/512/1016366). Ambiguous nucleotides are skipped during the distance calculation. Genetic distance (substitutions per site) was converted to substitutions per genome (i.e, single-nucleotide polymorphism: SNPs) by multiplying with the alignment length (29903).

```aiignore
tn93 -t 1 -a skip -o "MGH_DPH_98percent_772samples_tn93_distances.csv" \
"MGH_DPH_98percent_772samples_aligned.fasta"
```


