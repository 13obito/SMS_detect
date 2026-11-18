# SMS Spam Detection Under Character Obfuscation

This project tests how small changes to text affect SMS spam detection. It uses TF-IDF and Logistic Regression to classify messages as spam or ham (non-spam).

The experiments compare word and character features, with and without edited training examples. Text changes include replacing letters with numbers, adding spaces inside words, and swapping nearby letters.

## Setup and quick start

Use Python 3.11 or later. Run these commands from the project folder:

```bash
python -m pip install -r requirements.txt
python run_experiments.py midsemester --download
```

This downloads the UCI SMS Spam Collection, prepares the data, and runs the word-based baseline on clean and edited validation messages. A GPU is not needed.

The dataset comes from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). If the download fails, extract the dataset and place `SMSSpamCollection` in `data/raw/`.

## Experiments

The data is split into about 70% training, 15% validation, and 15% testing. Similar messages are kept in the same split to reduce data leakage. The midsemester experiment uses only the validation set.

Compare word and character TF-IDF using the same baseline settings:

```bash
python run_experiments.py compare
```

Tune models with and without edited training examples, then save the selected settings:

```bash
python run_experiments.py tune
```

Run the final test only after the model settings are fixed:

```bash
python run_experiments.py final --frozen-run frozen_development
```

Default settings are in `config.json`. For more options, run `python run_experiments.py --help`.

## Results

Results are saved in:

- `results/midsemester/` for the word baseline.
- `results/representation_comparison/` for the word and character comparison.
- `results/frozen_development/` for tuned models, with final test results in `final_test/`.

Outputs include precision, recall, F1, the false positive rate for ham, plots, predictions, and error examples. For the midsemester report, start with `report_fill_in.txt` and the three `table*.csv` files. Tables with `_percent` in the name use percentages; other metric tables use values from 0 to 1.

The files in `manual_review/` need to be checked by hand for readability and whether the edited messages keep their original meaning.

## Project files

- `run_experiments.py`: command-line entry point.
- `sms_spam/`: data preparation, text edits, models, evaluation, and reports.
- `config.json`: experiment settings and random seeds.
- `requirements.txt`: required packages.
- `requirements-tested.txt`: package versions used for the tested setup.
- `tests/`: automated tests.

Run the tests with:

```bash
python -m pytest -q
```
