# Cross-Transition Pattern Summary

## Group overview

| transition_group   |   total_frequency |   distinct_mechanisms | dominant_mechanism              |   dominant_mechanism_frequency |   shared_mechanisms_across_all_groups |   unique_mechanisms_in_dataset |
|:-------------------|------------------:|----------------------:|:--------------------------------|-------------------------------:|--------------------------------------:|-------------------------------:|
| SAFE_to_SAFE       |               308 |                     6 | Harmful intent ignored          |                             69 |                                     5 |                             12 |
| SAFE_to_UNSAFE     |                44 |                    10 | Procedural assistance increased |                             11 |                                     5 |                             12 |
| UNSAFE_to_SAFE     |               387 |                     9 | Harmful intent ignored          |                             96 |                                     5 |                             12 |
| UNSAFE_to_UNSAFE   |               340 |                    15 | OCR dependency                  |                             63 |                                     5 |                             12 |

## Pattern family counts

| pattern_family           |   size |
|:-------------------------|-------:|
| unique_to_one_group      |     12 |
| shared_across_all_groups |      5 |
| change_leaning           |      3 |
| stable_leaning           |      1 |

## Transition class counts

| transition_class         |   size |
|:-------------------------|-------:|
| unique_to_one_group      |     12 |
| shared_across_all_groups |      5 |
| transition_dominant      |      3 |
| stability_dominant       |      1 |

## Top mechanisms by total frequency

| Behavioral Mechanism                      |   total_frequency | transition_class         | dominant_side   |
|:------------------------------------------|------------------:|:-------------------------|:----------------|
| Harmful intent ignored                    |               195 | shared_across_all_groups | change          |
| Benign reinterpretation                   |               208 | shared_across_all_groups | stable          |
| Visual grounding                          |                94 | shared_across_all_groups | stable          |
| Image sufficient                          |                74 | shared_across_all_groups | stable          |
| Prompt dependency                         |                35 | shared_across_all_groups | stable          |
| Stable descriptive behavior               |                71 | stability_dominant       | stable          |
| OCR dependency                            |               153 | transition_dominant      | change          |
| Procedural assistance reduced             |                91 | transition_dominant      | change          |
| Procedural assistance increased           |                13 | transition_dominant      | change          |
| Stable benign behavior                    |                27 | unique_to_one_group      | change          |
| Operational interpretation                |                 7 | unique_to_one_group      | change          |
| Concealment-oriented assistance increased |                 1 | unique_to_one_group      | change          |
| Disclaimer with assistance                |                 1 | unique_to_one_group      | change          |
| Stable harmful-intent avoidance           |                 1 | unique_to_one_group      | change          |
| Stable refusal                            |                 1 | unique_to_one_group      | change          |

## Stability-leaning mechanisms

| Behavioral Mechanism        |   SAFE_to_SAFE |   UNSAFE_to_UNSAFE |   SAFE_to_UNSAFE |   UNSAFE_to_SAFE |   total_frequency |
|:----------------------------|---------------:|-------------------:|-----------------:|-----------------:|------------------:|
| Stable descriptive behavior |             69 |                  2 |                0 |                0 |                71 |

## Transition-leaning mechanisms

| Behavioral Mechanism            |   SAFE_to_SAFE |   UNSAFE_to_UNSAFE |   SAFE_to_UNSAFE |   UNSAFE_to_SAFE |   total_frequency |
|:--------------------------------|---------------:|-------------------:|-----------------:|-----------------:|------------------:|
| OCR dependency                  |              0 |                 63 |                0 |               90 |               153 |
| Procedural assistance reduced   |              0 |                 22 |                0 |               69 |                91 |
| Procedural assistance increased |              0 |                  2 |               11 |                0 |                13 |

## Mechanisms shared across all four groups

| Behavioral Mechanism    |   total_frequency |   balance_score |
|:------------------------|------------------:|----------------:|
| Harmful intent ignored  |               195 |      -0.025641  |
| Benign reinterpretation |               208 |       0.0288462 |
| Visual grounding        |                94 |       0.808511  |
| Image sufficient        |                74 |       0.918919  |
| Prompt dependency       |                35 |       0.428571  |

