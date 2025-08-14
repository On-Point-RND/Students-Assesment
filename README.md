# LLMs for Educational Assessment

**Scope:** Evaluate the effectiveness and reliability of LLMs
in automating assessment of anonymized summer school
applications through direct comparison with human
evaluations.

**Provide actionable recommendations** on
implementation feasibility and contexts favoring human
versus LLM-based evaluation.

**Importance:** Identify biases and inconsistencies in human
assessment while revealing fundamental differences
between human and LLM evaluation approaches, enabling
systemic improvements in educational evaluation
practices.

## Data overview

The raw data is saved into `/data/raw` folder. It contains anonymized entries (CVs, Motivational letters and Presentation texts) from applicants and their scores. The initial data is spreaded among several subfolders with `.txt` files. The applicants' scores and their final marks are saved in several `.csv` files that have been created at various stages of selection process and during the school.

The preprocessed and compressed dataset is saved as `/data/dataset.parquet` file with all necessary data for each student, who have either submitted CV, Motivational letter or Presentation. The dataset contains 577 rows and 25 columns in total. The table below gives explicit description of the dataset content.

| # | Column                      | Non-Null Count | Dtype   | Range      | Description                                                                 |
|---|-----------------------------|----------------|---------|------------|-----------------------------------------------------------------------------|
| 0 | cv                          | 567 non-null   | object  | N/A        | The anonymized CV in the form of string.                                    |
| 1 | letter                      | 568 non-null   | object  | N/A        | The anonymized Motivational letter in the form of string.                   |
| 2 | presentation                | 564 non-null   | object  | N/A        | The anonymized text from Presentation slides in the form of string.        |
| 3 | cv_phd_1                    | 567 non-null   | float64 | 0 - 5      | The CV score from first PhD student assessor.                               |
| 4 | cv_phd_2                    | 567 non-null   | float64 | 0 - 5      | The CV score from second PhD student assessor.                              |
| 5 | letter_phd_1                | 567 non-null   | float64 | 0 - 5      | The Motivational letter score from first PhD student assessor.              |
| 6 | letter_phd_2                | 567 non-null   | float64 | 0 - 5      | The Motivational letter score from second PhD student assessor.             |
| 7 | pres_phd_1                  | 567 non-null   | float64 | 0 - 20     | The Presentation score from first PhD student assessor.                     |
| 8 | pres_phd_2                  | 567 non-null   | float64 | 0 - 20     | The Presentation score from second PhD student assessor.                    |
| 9 | pres_class                  | 567 non-null   | float64 | 0 - 3      | The type of Presentation (review, article replication of personal project). |
| 10| video_phd_1                 | 383 non-null   | float64 | 0 - 20     | The Presentation video score from first PhD student assessor.               |
| 11| video_phd_2                 | 409 non-null   | float64 | 0 - 20     | The Presentation video score from second PhD student assessor.              |
| 12| all_phd_1                   | 567 non-null   | float64 | 0 - 100    | The total score from first PhD student assessor.                            |
| 13| all_phd_2                   | 567 non-null   | float64 | 0 - 100    | The total score from first PhD student assessor.                            |
| 14| final_human_score           | 98 non-null    | float64 | 0 - 100    | The final score (weighted mean of PhD students and Professor scores).       |
| 15| prof_score                  | 98 non-null    | float64 | 0 - 10     | The Professor's overall score.                                              |
| 16| offline_test_1              | 48 non-null    | float64 | 0 - 1      | The results of the first offline test.                                      |
| 17| offline_test_2              | 48 non-null    | float64 | 0 - 1      | The results of the second offline test.                                     |
| 18| offline_test_3              | 48 non-null    | float64 | 0 - 1      | The results of the third offline test.                                      |
| 19| offline_test_4              | 48 non-null    | float64 | 0 - 1      | The results of the fourth offline test.                                     |
| 20| offline_test_total          | 48 non-null    | float64 | 0 - 1      | The sum of all offline tests.                                               |
| 21| online_test_score           | 68 non-null    | float64 | 0 - 67     | The results of the online test                                              |
| 22| project_participation_flag  | 577 non-null   | bool    | T/F        | The flag of student's participation in the project activity.                |
| 23| project_ta_score            | 118 non-null   | float64 | 0 - 5      | The project score from the teacher assistants.                              |
| 24| project_peer_score          | 118 non-null   | float64 | 0 - 10     | The project score from the peer-review stage.                               |

The detailed data description is provided in `/preprocessing.ipynb`.

