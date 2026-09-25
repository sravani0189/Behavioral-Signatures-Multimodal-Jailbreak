# Information Source Analysis Summary

## Variant overview

| variant   | transition_group   |   sample_count |
|:----------|:-------------------|---------------:|
| SD        | SAFE_to_SAFE       |            220 |
| SD        | SAFE_to_UNSAFE     |             22 |
| SD        | UNSAFE_to_SAFE     |             26 |
| SD        | UNSAFE_to_UNSAFE   |            112 |
| SD_TYPO   | SAFE_to_SAFE       |             40 |
| SD_TYPO   | SAFE_to_UNSAFE     |             14 |
| SD_TYPO   | UNSAFE_to_SAFE     |            173 |
| SD_TYPO   | UNSAFE_to_UNSAFE   |            120 |
| TYPO      | SAFE_to_SAFE       |             48 |
| TYPO      | SAFE_to_UNSAFE     |              8 |
| TYPO      | UNSAFE_to_SAFE     |            188 |
| TYPO      | UNSAFE_to_UNSAFE   |            108 |

## Source overview

| input_modality   | transition_group   |   sample_count |
|:-----------------|:-------------------|---------------:|
| Image            | SAFE_to_SAFE       |            220 |
| Image            | SAFE_to_UNSAFE     |             22 |
| Image            | UNSAFE_to_SAFE     |             26 |
| Image            | UNSAFE_to_UNSAFE   |            112 |
| Image+Text       | SAFE_to_SAFE       |             40 |
| Image+Text       | SAFE_to_UNSAFE     |             14 |
| Image+Text       | UNSAFE_to_SAFE     |            173 |
| Image+Text       | UNSAFE_to_UNSAFE   |            120 |
| Text             | SAFE_to_SAFE       |             48 |
| Text             | SAFE_to_UNSAFE     |              8 |
| Text             | UNSAFE_to_SAFE     |            188 |
| Text             | UNSAFE_to_UNSAFE   |            108 |

## Variant influence summary

| variant   |   sample_count |   distinct_mechanisms | top_mechanism           |   top_mechanism_count | top_source_candidate                 |
|:----------|---------------:|----------------------:|:------------------------|----------------------:|:-------------------------------------|
| SD        |            380 |                    17 | Harmful intent ignored  |                    75 | Visual scene likely influential      |
| SD_TYPO   |            347 |                    16 | Benign reinterpretation |                    72 | Visual scene likely influential      |
| TYPO      |            352 |                    16 | OCR dependency          |                    80 | Embedded OCR text likely influential |

## Source influence summary

| source_candidate                     |   mechanism_count |   total_occurrences |   shared_across_all_variants |   variant_specific |   mean_dominance_ratio |
|:-------------------------------------|------------------:|--------------------:|-----------------------------:|-------------------:|-----------------------:|
| Multisource / robust across variants |                 3 |                 410 |                            3 |                  0 |                 0.3864 |
| Embedded OCR text likely influential |                 4 |                 308 |                            3 |                  0 |                 0.5999 |
| Visual scene likely influential      |                 6 |                 307 |                            5 |                  0 |                 0.6294 |

## Top mechanisms by total occurrences

| Behavioral Mechanism                      |   total_occurrences |
|:------------------------------------------|--------------------:|
| Benign reinterpretation                   |                 208 |
| Harmful intent ignored                    |                 195 |
| OCR dependency                            |                 153 |
| Visual grounding                          |                  94 |
| Procedural assistance reduced             |                  91 |
| Image sufficient                          |                  74 |
| Stable descriptive behavior               |                  71 |
| Procedural assistance maintained          |                  42 |
| Partial objective preservation            |                  37 |
| Prompt dependency                         |                  35 |
| Stable benign behavior                    |                  27 |
| Operational assistance maintained         |                  15 |
| Procedural assistance increased           |                  13 |
| Harmful reinterpretation                  |                  11 |
| Operational interpretation                |                   7 |
| Concealment-oriented assistance increased |                   1 |
| Disclaimer with assistance                |                   1 |
| Partial OCR dependency                    |                   1 |
| Partial phrase preservation               |                   1 |
| Stable harmful-intent avoidance           |                   1 |

## Shared vs variant-specific mechanisms

| Behavioral Mechanism                      | variants_present   |   variant_support_count | support_class              | dominant_variant   |   dominant_variant_count | source_candidate                                      |   dominance_ratio |
|:------------------------------------------|:-------------------|------------------------:|:---------------------------|:-------------------|-------------------------:|:------------------------------------------------------|------------------:|
| OCR dependency                            | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | TYPO               |                       80 | Embedded OCR text likely influential                  |          0.522876 |
| Harmful intent ignored                    | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD                 |                       75 | Multisource / robust across variants                  |          0.384615 |
| Benign reinterpretation                   | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD_TYPO            |                       72 | Multisource / robust across variants                  |          0.346154 |
| Visual grounding                          | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD                 |                       67 | Visual scene likely influential                       |          0.712766 |
| Image sufficient                          | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD                 |                       51 | Visual scene likely influential                       |          0.689189 |
| Stable descriptive behavior               | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD                 |                       47 | Visual scene likely influential                       |          0.661972 |
| Partial objective preservation            | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | TYPO               |                       25 | Embedded OCR text likely influential                  |          0.675676 |
| Procedural assistance maintained          | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD_TYPO            |                       20 | Visual scene likely influential                       |          0.47619  |
| Prompt dependency                         | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD                 |                       18 | Shared instruction / prompt influence likely          |          0.514286 |
| Stable benign behavior                    | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | TYPO               |                       17 | Embedded OCR text likely influential                  |          0.62963  |
| Operational assistance maintained         | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD                 |                        9 | Visual scene likely influential                       |          0.6      |
| Procedural assistance increased           | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD                 |                        7 | Shared instruction / prompt influence likely          |          0.538462 |
| Operational interpretation                | SD, SD_TYPO, TYPO  |                       3 | shared_across_all_variants | SD                 |                        3 | Multisource / robust across variants                  |          0.428571 |
| Procedural assistance reduced             | SD_TYPO, TYPO      |                       2 | shared_across_two_variants | TYPO               |                       52 | Embedded OCR text likely influential                  |          0.571429 |
| Harmful reinterpretation                  | SD, SD_TYPO        |                       2 | shared_across_two_variants | SD_TYPO            |                        7 | Visual scene likely influential                       |          0.636364 |
| Concealment-oriented assistance increased | TYPO               |                       1 | variant_specific           | TYPO               |                        1 | Embedded OCR text + Rephrased Question                |          1        |
| Disclaimer with assistance                | SD                 |                       1 | variant_specific           | SD                 |                        1 | Visual scene + SD prompt                              |          1        |
| Partial OCR dependency                    | SD                 |                       1 | variant_specific           | SD                 |                        1 | Visual scene + SD prompt                              |          1        |
| Partial phrase preservation               | SD_TYPO            |                       1 | variant_specific           | SD_TYPO            |                        1 | Visual scene + Embedded OCR text + Rephrased Question |          1        |
| Stable harmful-intent avoidance           | SD                 |                       1 | variant_specific           | SD                 |                        1 | Visual scene + SD prompt                              |          1        |
| Stable refusal                            | TYPO               |                       1 | variant_specific           | TYPO               |                        1 | Embedded OCR text + Rephrased Question                |          1        |

## Mechanism by variant

| Behavioral Mechanism                      |   SD |   SD_TYPO |   TYPO |
|:------------------------------------------|-----:|----------:|-------:|
| Benign reinterpretation                   |   71 |        72 |     65 |
| Concealment-oriented assistance increased |    0 |         0 |      1 |
| Disclaimer with assistance                |    1 |         0 |      0 |
| Harmful intent ignored                    |   75 |        56 |     64 |
| Harmful reinterpretation                  |    4 |         7 |      0 |
| Image sufficient                          |   51 |        11 |     12 |
| OCR dependency                            |    1 |        72 |     80 |
| Operational assistance maintained         |    9 |         4 |      2 |
| Operational interpretation                |    3 |         2 |      2 |
| Partial OCR dependency                    |    1 |         0 |      0 |
| Partial objective preservation            |    2 |        10 |     25 |
| Partial phrase preservation               |    0 |         1 |      0 |
| Procedural assistance increased           |    7 |         4 |      2 |
| Procedural assistance maintained          |   16 |        20 |      6 |
| Procedural assistance reduced             |    0 |        39 |     52 |
| Prompt dependency                         |   18 |        11 |      6 |
| Stable benign behavior                    |    6 |         4 |     17 |
| Stable descriptive behavior               |   47 |        11 |     13 |
| Stable harmful-intent avoidance           |    1 |         0 |      0 |
| Stable refusal                            |    0 |         0 |      1 |
| Visual grounding                          |   67 |        23 |      4 |

## Mechanism by transition group

| Behavioral Mechanism                      |   SAFE_to_SAFE |   SAFE_to_UNSAFE |   UNSAFE_to_SAFE |   UNSAFE_to_UNSAFE |
|:------------------------------------------|---------------:|-----------------:|-----------------:|-------------------:|
| Benign reinterpretation                   |             62 |                6 |               95 |                 45 |
| Concealment-oriented assistance increased |              0 |                1 |                0 |                  0 |
| Disclaimer with assistance                |              0 |                1 |                0 |                  0 |
| Harmful intent ignored                    |             69 |                4 |               96 |                 26 |
| Harmful reinterpretation                  |              0 |                0 |                0 |                 11 |
| Image sufficient                          |             69 |                1 |                2 |                  2 |
| OCR dependency                            |              0 |                0 |               90 |                 63 |
| Operational assistance maintained         |              0 |                0 |                0 |                 15 |
| Operational interpretation                |              0 |                7 |                0 |                  0 |
| Partial OCR dependency                    |              0 |                0 |                0 |                  1 |
| Partial objective preservation            |              0 |                0 |                0 |                 37 |
| Partial phrase preservation               |              0 |                0 |                0 |                  1 |
| Procedural assistance increased           |              0 |               11 |                0 |                  2 |
| Procedural assistance maintained          |              0 |                0 |                0 |                 42 |
| Procedural assistance reduced             |              0 |                0 |               69 |                 22 |
| Prompt dependency                         |              2 |                6 |                4 |                 23 |
| Stable benign behavior                    |              0 |                0 |               27 |                  0 |
| Stable descriptive behavior               |             69 |                0 |                0 |                  2 |
| Stable harmful-intent avoidance           |              0 |                1 |                0 |                  0 |
| Stable refusal                            |              0 |                0 |                1 |                  0 |
| Visual grounding                          |             37 |                6 |                3 |                 48 |

## Mechanism by prompt variant

| Behavioral Mechanism                      |   Rephrased Question |   Rephrased Question(SD) |
|:------------------------------------------|---------------------:|-------------------------:|
| Benign reinterpretation                   |                  137 |                       71 |
| Concealment-oriented assistance increased |                    1 |                        0 |
| Disclaimer with assistance                |                    0 |                        1 |
| Harmful intent ignored                    |                  120 |                       75 |
| Harmful reinterpretation                  |                    7 |                        4 |
| Image sufficient                          |                   23 |                       51 |
| OCR dependency                            |                  152 |                        1 |
| Operational assistance maintained         |                    6 |                        9 |
| Operational interpretation                |                    4 |                        3 |
| Partial OCR dependency                    |                    0 |                        1 |
| Partial objective preservation            |                   35 |                        2 |
| Partial phrase preservation               |                    1 |                        0 |
| Procedural assistance increased           |                    6 |                        7 |
| Procedural assistance maintained          |                   26 |                       16 |
| Procedural assistance reduced             |                   91 |                        0 |
| Prompt dependency                         |                   17 |                       18 |
| Stable benign behavior                    |                   21 |                        6 |
| Stable descriptive behavior               |                   24 |                       47 |
| Stable harmful-intent avoidance           |                    0 |                        1 |
| Stable refusal                            |                    1 |                        0 |
| Visual grounding                          |                   27 |                       67 |

## Source signature by mechanism

| Behavioral Mechanism                      |   OCR text + prompt |   Visual scene + OCR text + prompt |   Visual scene + SD prompt |
|:------------------------------------------|--------------------:|-----------------------------------:|---------------------------:|
| Benign reinterpretation                   |                  65 |                                 72 |                         71 |
| Concealment-oriented assistance increased |                   1 |                                  0 |                          0 |
| Disclaimer with assistance                |                   0 |                                  0 |                          1 |
| Harmful intent ignored                    |                  64 |                                 56 |                         75 |
| Harmful reinterpretation                  |                   0 |                                  7 |                          4 |
| Image sufficient                          |                  12 |                                 11 |                         51 |
| OCR dependency                            |                  80 |                                 72 |                          1 |
| Operational assistance maintained         |                   2 |                                  4 |                          9 |
| Operational interpretation                |                   2 |                                  2 |                          3 |
| Partial OCR dependency                    |                   0 |                                  0 |                          1 |
| Partial objective preservation            |                  25 |                                 10 |                          2 |
| Partial phrase preservation               |                   0 |                                  1 |                          0 |
| Procedural assistance increased           |                   2 |                                  4 |                          7 |
| Procedural assistance maintained          |                   6 |                                 20 |                         16 |
| Procedural assistance reduced             |                  52 |                                 39 |                          0 |
| Prompt dependency                         |                   6 |                                 11 |                         18 |
| Stable benign behavior                    |                  17 |                                  4 |                          6 |
| Stable descriptive behavior               |                  13 |                                 11 |                         47 |
| Stable harmful-intent avoidance           |                   0 |                                  0 |                          1 |
| Stable refusal                            |                   1 |                                  0 |                          0 |
| Visual grounding                          |                   4 |                                 23 |                         67 |
