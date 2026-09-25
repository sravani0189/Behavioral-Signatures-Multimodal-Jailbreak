# Information Source Analysis Protocol (ISAP)

## Variant summary

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

## Source summary

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

## Dominant source candidates

| Behavioral Mechanism                      |   SD_count |   SD_TYPO_count |   TYPO_count |   image_bearing_count |   ocr_bearing_count |   image_bearing_ratio |   ocr_bearing_ratio |   SD_prompt_count |   Rephrased_Question_count |   prompt_dominance_ratio | variants_present   |   variant_support_count | dominant_variant   |   dominant_variant_count |   dominance_ratio | source_candidate                                      |   source_confidence_score | source_confidence_level   | source_rationale                                                     | prompt_candidate                          |   prompt_confidence_score | prompt_confidence_level   | prompt_rationale                                                                                |
|:------------------------------------------|-----------:|----------------:|-------------:|----------------------:|--------------------:|----------------------:|--------------------:|------------------:|---------------------------:|-------------------------:|:-------------------|------------------------:|:-------------------|-------------------------:|------------------:|:------------------------------------------------------|--------------------------:|:--------------------------|:---------------------------------------------------------------------|:------------------------------------------|--------------------------:|:--------------------------|:------------------------------------------------------------------------------------------------|
| OCR dependency                            |          1 |              72 |           80 |                    73 |                 152 |                0.4771 |              0.9935 |                 1 |                        152 |                   0.9935 | SD, SD_TYPO, TYPO  |                       3 | TYPO               |                       80 |            0.5229 | Embedded OCR text likely influential                  |                      0.85 | High                      | OCR-bearing variants dominate (152/153); SD is weak (1/153)          | Rephrased Question likely influential     |                    0.85   | High                      | Generic Rephrased Question dominates: 152/153                                                   |
| Harmful intent ignored                    |         75 |              56 |           64 |                   131 |                 120 |                0.6718 |              0.6154 |                75 |                        120 |                   0.6154 | SD, SD_TYPO, TYPO  |                       3 | SD                 |                       75 |            0.3846 | Multisource / robust across variants                  |                      0.6  | Medium                    | Counts are relatively balanced across SD=75, SD_TYPO=56, TYPO=64     | Rephrased Question likely influential     |                    0.6154 | Medium                    | Generic Rephrased Question dominates: 120/195                                                   |
| Benign reinterpretation                   |         71 |              72 |           65 |                   143 |                 137 |                0.6875 |              0.6587 |                71 |                        137 |                   0.6587 | SD, SD_TYPO, TYPO  |                       3 | SD_TYPO            |                       72 |            0.3462 | Multisource / robust across variants                  |                      0.6  | Medium                    | Counts are relatively balanced across SD=71, SD_TYPO=72, TYPO=65     | Rephrased Question likely influential     |                    0.6587 | Medium                    | Generic Rephrased Question dominates: 137/208                                                   |
| Visual grounding                          |         67 |              23 |            4 |                    90 |                  27 |                0.9574 |              0.2872 |                67 |                         27 |                   0.7128 | SD, SD_TYPO, TYPO  |                       3 | SD                 |                       67 |            0.7128 | Visual scene likely influential                       |                      0.85 | High                      | Image-bearing variants dominate (90/94); TYPO is weak (4/94)         | Rephrased Question(SD) likely influential |                    0.7128 | Medium                    | SD prompt dominates: 67/94                                                                      |
| Image sufficient                          |         51 |              11 |           12 |                    62 |                  23 |                0.8378 |              0.3108 |                51 |                         23 |                   0.6892 | SD, SD_TYPO, TYPO  |                       3 | SD                 |                       51 |            0.6892 | Visual scene likely influential                       |                      0.65 | Medium                    | Concentrated in SD (51/74)                                           | Rephrased Question(SD) likely influential |                    0.6892 | Medium                    | SD prompt dominates: 51/74                                                                      |
| Stable descriptive behavior               |         47 |              11 |           13 |                    58 |                  24 |                0.8169 |              0.338  |                47 |                         24 |                   0.662  | SD, SD_TYPO, TYPO  |                       3 | SD                 |                       47 |            0.662  | Visual scene likely influential                       |                      0.65 | Medium                    | Concentrated in SD (47/71)                                           | Rephrased Question(SD) likely influential |                    0.662  | Medium                    | SD prompt dominates: 47/71                                                                      |
| Partial objective preservation            |          2 |              10 |           25 |                    12 |                  35 |                0.3243 |              0.9459 |                 2 |                         35 |                   0.9459 | SD, SD_TYPO, TYPO  |                       3 | TYPO               |                       25 |            0.6757 | Embedded OCR text likely influential                  |                      0.85 | High                      | OCR-bearing variants dominate (35/37); SD is weak (2/37)             | Rephrased Question likely influential     |                    0.85   | High                      | Generic Rephrased Question dominates: 35/37                                                     |
| Procedural assistance maintained          |         16 |              20 |            6 |                    36 |                  26 |                0.8571 |              0.619  |                16 |                         26 |                   0.619  | SD, SD_TYPO, TYPO  |                       3 | SD_TYPO            |                       20 |            0.4762 | Visual scene likely influential                       |                      0.85 | High                      | Image-bearing variants dominate (36/42); TYPO is weak (6/42)         | Rephrased Question likely influential     |                    0.619  | Medium                    | Generic Rephrased Question dominates: 26/42                                                     |
| Prompt dependency                         |         18 |              11 |            6 |                    29 |                  17 |                0.8286 |              0.4857 |                18 |                         17 |                   0.5143 | SD, SD_TYPO, TYPO  |                       3 | SD                 |                       18 |            0.5143 | Shared instruction / prompt influence likely          |                      0.5  | Medium                    | No single source family clearly dominates: SD=18, SD_TYPO=11, TYPO=6 | Prompt wording likely shared influence    |                    0.55   | Medium                    | Prompt variants are relatively balanced: SD prompt=18, Rephrased Question=17                    |
| Stable benign behavior                    |          6 |               4 |           17 |                    10 |                  21 |                0.3704 |              0.7778 |                 6 |                         21 |                   0.7778 | SD, SD_TYPO, TYPO  |                       3 | TYPO               |                       17 |            0.6296 | Embedded OCR text likely influential                  |                      0.65 | Medium                    | Concentrated in TYPO (17/27)                                         | Rephrased Question likely influential     |                    0.7778 | High                      | Generic Rephrased Question dominates: 21/27                                                     |
| Operational assistance maintained         |          9 |               4 |            2 |                    13 |                   6 |                0.8667 |              0.4    |                 9 |                          6 |                   0.6    | SD, SD_TYPO, TYPO  |                       3 | SD                 |                        9 |            0.6    | Visual scene likely influential                       |                      0.85 | High                      | Image-bearing variants dominate (13/15); TYPO is weak (2/15)         | Rephrased Question(SD) likely influential |                    0.6    | Medium                    | SD prompt dominates: 9/15                                                                       |
| Procedural assistance increased           |          7 |               4 |            2 |                    11 |                   6 |                0.8462 |              0.4615 |                 7 |                          6 |                   0.5385 | SD, SD_TYPO, TYPO  |                       3 | SD                 |                        7 |            0.5385 | Shared instruction / prompt influence likely          |                      0.5  | Medium                    | No single source family clearly dominates: SD=7, SD_TYPO=4, TYPO=2   | Prompt wording likely shared influence    |                    0.55   | Medium                    | Prompt variants are relatively balanced: SD prompt=7, Rephrased Question=6                      |
| Operational interpretation                |          3 |               2 |            2 |                     5 |                   4 |                0.7143 |              0.5714 |                 3 |                          4 |                   0.5714 | SD, SD_TYPO, TYPO  |                       3 | SD                 |                        3 |            0.4286 | Multisource / robust across variants                  |                      0.6  | Medium                    | Counts are relatively balanced across SD=3, SD_TYPO=2, TYPO=2        | Prompt wording likely shared influence    |                    0.55   | Medium                    | Prompt variants are relatively balanced: SD prompt=3, Rephrased Question=4                      |
| Procedural assistance reduced             |          0 |              39 |           52 |                    39 |                  91 |                0.4286 |              1      |                 0 |                         91 |                   1      | SD_TYPO, TYPO      |                       2 | TYPO               |                       52 |            0.5714 | Embedded OCR text likely influential                  |                      0.85 | High                      | OCR-bearing variants dominate (91/91); SD is weak (0/91)             | Rephrased Question likely influential     |                    0.7    | Medium                    | No SD-prompt occurrences; all instances are under Rephrased Question (91/91)                    |
| Harmful reinterpretation                  |          4 |               7 |            0 |                    11 |                   7 |                1      |              0.6364 |                 4 |                          7 |                   0.6364 | SD, SD_TYPO        |                       2 | SD_TYPO            |                        7 |            0.6364 | Visual scene likely influential                       |                      0.85 | High                      | Image-bearing variants dominate (11/11); TYPO is weak (0/11)         | Rephrased Question likely influential     |                    0.6364 | Medium                    | Generic Rephrased Question dominates: 7/11                                                      |
| Concealment-oriented assistance increased |          0 |               0 |            1 |                     0 |                   1 |                0      |              1      |                 0 |                          1 |                   1      | TYPO               |                       1 | TYPO               |                        1 |            1      | Embedded OCR text + Rephrased Question                |                      0.35 | Low                       | Observed only in TYPO (1/1)                                          | Rephrased Question likely influential     |                    0.7    | Medium                    | No SD-prompt occurrences; all instances are under Rephrased Question (1/1)                      |
| Disclaimer with assistance                |          1 |               0 |            0 |                     1 |                   0 |                1      |              0      |                 1 |                          0 |                   1      | SD                 |                       1 | SD                 |                        1 |            1      | Visual scene + SD prompt                              |                      0.35 | Low                       | Observed only in SD (1/1)                                            | Rephrased Question(SD) likely influential |                    0.7    | Medium                    | No generic Rephrased Question occurrences; all instances are under Rephrased Question(SD) (1/1) |
| Partial OCR dependency                    |          1 |               0 |            0 |                     1 |                   0 |                1      |              0      |                 1 |                          0 |                   1      | SD                 |                       1 | SD                 |                        1 |            1      | Visual scene + SD prompt                              |                      0.35 | Low                       | Observed only in SD (1/1)                                            | Rephrased Question(SD) likely influential |                    0.7    | Medium                    | No generic Rephrased Question occurrences; all instances are under Rephrased Question(SD) (1/1) |
| Partial phrase preservation               |          0 |               1 |            0 |                     1 |                   1 |                1      |              1      |                 0 |                          1 |                   1      | SD_TYPO            |                       1 | SD_TYPO            |                        1 |            1      | Visual scene + Embedded OCR text + Rephrased Question |                      0.35 | Low                       | Observed only in SD_TYPO (1/1)                                       | Rephrased Question likely influential     |                    0.7    | Medium                    | No SD-prompt occurrences; all instances are under Rephrased Question (1/1)                      |
| Stable harmful-intent avoidance           |          1 |               0 |            0 |                     1 |                   0 |                1      |              0      |                 1 |                          0 |                   1      | SD                 |                       1 | SD                 |                        1 |            1      | Visual scene + SD prompt                              |                      0.35 | Low                       | Observed only in SD (1/1)                                            | Rephrased Question(SD) likely influential |                    0.7    | Medium                    | No generic Rephrased Question occurrences; all instances are under Rephrased Question(SD) (1/1) |
| Stable refusal                            |          0 |               0 |            1 |                     0 |                   1 |                0      |              1      |                 0 |                          1 |                   1      | TYPO               |                       1 | TYPO               |                        1 |            1      | Embedded OCR text + Rephrased Question                |                      0.35 | Low                       | Observed only in TYPO (1/1)                                          | Rephrased Question likely influential     |                    0.7    | Medium                    | No SD-prompt occurrences; all instances are under Rephrased Question (1/1)                      |
