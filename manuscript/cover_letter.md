# Cover Letter — PLOS Computational Biology

---

**To:** The Editors, *PLOS Computational Biology*

**Manuscript title:** EpiLink: a process-based compatibility model for genomic transmission clustering in infectious disease surveillance

**Running title:** EpiLink: process-based genomic transmission clustering

**Corresponding author:** Rowland R. Kao (rowland.kao@ed.ac.uk)

---

Dear Editors,

We submit for your consideration a Research Article describing **EpiLink**, a new computational method for identifying transmission clusters from pathogen genomic surveillance data. We believe this work is a strong fit for *PLOS Computational Biology* under the **Epidemiology & Public Health** section.

**What the paper addresses.** Grouping infectious disease cases into transmission clusters is a routine task in pathogen surveillance, yet most current approaches rely on fixed genetic distance thresholds whose relationship to recent transmission is often unclear—particularly during superspreading events, where rapid, clustered transmission produces many cases with minimal genetic divergence. We developed EpiLink to address this interpretability gap: rather than applying an arbitrary cut-off, it generates graded pairwise compatibility scores by simulating the joint distribution of expected temporal and genetic distances under user-defined recent-transmission scenarios, explicitly accounting for uncertainty in incubation period, testing delay, and molecular evolution.

**Key advances.** First, EpiLink is threshold-free and process-based: every compatibility score decomposes directly into temporal and genetic contributions from named transmission scenarios, making cluster membership auditable rather than opaque. Second, its best-performing variant (ESD) approaches the clustering accuracy of a supervised logistic regression model trained on labelled transmission pairs, without requiring any labelled data—a practically important property for novel or under-resourced outbreak settings. Third, we identify a clear operational trade-off between deterministic and stochastic model formulations, provide systematic sensitivity analyses of natural-history and evolutionary parameters, and demonstrate temporal stability of cluster partitions that makes the method suitable for rolling surveillance. Finally, application to 772 SARS-CoV-2 sequences from the 2020 Boston epidemic recovers clusters strongly enriched for documented superspreading events at a skilled nursing facility and a conference, without any training supervision.

**Suitability for PLOS CB.** The paper combines novel computational method development, rigorous simulation-based benchmarking, and empirical validation on a well-characterised public health dataset—squarely within the scope of *PLOS Computational Biology*. The method, code, and all data required to reproduce the analyses are publicly available, consistent with the journal's open-science principles.

**Suggested Academic Editor.** We suggest **Sarah Cobey** (University of Chicago), whose work on pathogen evolutionary dynamics and genomic surveillance makes her well placed to handle this manuscript. We defer to the editors if a different assignment is more appropriate.

We confirm that this manuscript is not under consideration elsewhere and that all authors have approved the submission. We declare no competing interests.

We look forward to your consideration.

Yours sincerely,

Dominic Arthur, Christopher J. Banks, Rowland R. Kao

The Roslin Institute, University of Edinburgh

---

*Submitted on behalf of all authors by Rowland R. Kao*
