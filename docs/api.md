# API Reference

::: epilink.infectiousness_profile
    options:
      members:
        - InfectiousnessParams
        - TOST
        - TOIT
        - presymptomatic_fraction

::: epilink.transmission_linkage_model
    options:
      members:
        - Epilink
        - linkage_probability
        - linkage_probability_matrix
        - genetic_linkage_probability
        - temporal_linkage_probability

::: epilink.simulate_epidemic_and_genomic
    options:
      members:
        - SequencePacker64
        - PackedGenomicData
        - populate_epidemic_data
        - simulate_genomic_data
        - generate_pairwise_data
