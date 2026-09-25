# Cross-Transition Comparison Summary

## Group-level overview

| transition_group   |   total_frequency |   distinct_mechanisms | dominant_mechanism              |   dominant_mechanism_frequency |   shared_mechanisms_across_all_groups |   unique_mechanisms_in_dataset |
|:-------------------|------------------:|----------------------:|:--------------------------------|-------------------------------:|--------------------------------------:|-------------------------------:|
| SAFE_to_SAFE       |               308 |                     6 | Harmful intent ignored          |                             69 |                                     5 |                             12 |
| SAFE_to_UNSAFE     |                44 |                    10 | Procedural assistance increased |                             11 |                                     5 |                             12 |
| UNSAFE_to_SAFE     |               387 |                     9 | Harmful intent ignored          |                             96 |                                     5 |                             12 |
| UNSAFE_to_UNSAFE   |               340 |                    15 | OCR dependency                  |                             63 |                                     5 |                             12 |

## Shared mechanisms across all four transition groups

| Behavioral Mechanism    |   SAFE_to_SAFE |   SAFE_to_UNSAFE |   UNSAFE_to_SAFE |   UNSAFE_to_UNSAFE |   total_frequency |
|:------------------------|---------------:|-----------------:|-----------------:|-------------------:|------------------:|
| Benign reinterpretation |             62 |                6 |               95 |                 45 |               208 |
| Harmful intent ignored  |             69 |                4 |               96 |                 26 |               195 |
| Visual grounding        |             37 |                6 |                3 |                 48 |                94 |
| Image sufficient        |             69 |                1 |                2 |                  2 |                74 |
| Prompt dependency       |              2 |                6 |                4 |                 23 |                35 |

## Unique mechanisms appearing in only one group

| Behavioral Mechanism                      | unique_group     |   total_frequency |
|:------------------------------------------|:-----------------|------------------:|
| Operational interpretation                | SAFE_to_UNSAFE   |                 7 |
| Concealment-oriented assistance increased | SAFE_to_UNSAFE   |                 1 |
| Disclaimer with assistance                | SAFE_to_UNSAFE   |                 1 |
| Stable harmful-intent avoidance           | SAFE_to_UNSAFE   |                 1 |
| Stable benign behavior                    | UNSAFE_to_SAFE   |                27 |
| Stable refusal                            | UNSAFE_to_SAFE   |                 1 |
| Procedural assistance maintained          | UNSAFE_to_UNSAFE |                42 |
| Partial objective preservation            | UNSAFE_to_UNSAFE |                37 |
| Operational assistance maintained         | UNSAFE_to_UNSAFE |                15 |
| Harmful reinterpretation                  | UNSAFE_to_UNSAFE |                11 |
| Partial OCR dependency                    | UNSAFE_to_UNSAFE |                 1 |
| Partial phrase preservation               | UNSAFE_to_UNSAFE |                 1 |

## Mechanism profile table

| Behavioral Mechanism                      |   SAFE_to_SAFE |   SAFE_to_UNSAFE |   UNSAFE_to_SAFE |   UNSAFE_to_UNSAFE |   present_groups |   total_frequency | pattern_family           |
|:------------------------------------------|---------------:|-----------------:|-----------------:|-------------------:|-----------------:|------------------:|:-------------------------|
| Benign reinterpretation                   |             62 |                6 |               95 |                 45 |                4 |               208 | shared_across_all_groups |
| Concealment-oriented assistance increased |              0 |                1 |                0 |                  0 |                1 |                 1 | unique_to_one_group      |
| Disclaimer with assistance                |              0 |                1 |                0 |                  0 |                1 |                 1 | unique_to_one_group      |
| Harmful intent ignored                    |             69 |                4 |               96 |                 26 |                4 |               195 | shared_across_all_groups |
| Harmful reinterpretation                  |              0 |                0 |                0 |                 11 |                1 |                11 | unique_to_one_group      |
| Image sufficient                          |             69 |                1 |                2 |                  2 |                4 |                74 | shared_across_all_groups |
| OCR dependency                            |              0 |                0 |               90 |                 63 |                2 |               153 | change_leaning           |
| Operational assistance maintained         |              0 |                0 |                0 |                 15 |                1 |                15 | unique_to_one_group      |
| Operational interpretation                |              0 |                7 |                0 |                  0 |                1 |                 7 | unique_to_one_group      |
| Partial OCR dependency                    |              0 |                0 |                0 |                  1 |                1 |                 1 | unique_to_one_group      |
| Partial objective preservation            |              0 |                0 |                0 |                 37 |                1 |                37 | unique_to_one_group      |
| Partial phrase preservation               |              0 |                0 |                0 |                  1 |                1 |                 1 | unique_to_one_group      |
| Procedural assistance increased           |              0 |               11 |                0 |                  2 |                2 |                13 | change_leaning           |
| Procedural assistance maintained          |              0 |                0 |                0 |                 42 |                1 |                42 | unique_to_one_group      |
| Procedural assistance reduced             |              0 |                0 |               69 |                 22 |                2 |                91 | change_leaning           |
| Prompt dependency                         |              2 |                6 |                4 |                 23 |                4 |                35 | shared_across_all_groups |
| Stable benign behavior                    |              0 |                0 |               27 |                  0 |                1 |                27 | unique_to_one_group      |
| Stable descriptive behavior               |             69 |                0 |                0 |                  2 |                2 |                71 | stable_leaning           |
| Stable harmful-intent avoidance           |              0 |                1 |                0 |                  0 |                1 |                 1 | unique_to_one_group      |
| Stable refusal                            |              0 |                0 |                1 |                  0 |                1 |                 1 | unique_to_one_group      |
| Visual grounding                          |             37 |                6 |                3 |                 48 |                4 |                94 | shared_across_all_groups |
