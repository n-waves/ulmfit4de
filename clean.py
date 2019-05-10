from dataclasses import dataclass
from pathlib import Path
import fire


def fmtbl(condition, true_str, false_str):
    return true_str if condition else false_str

@dataclass
class LMExperiment:
    dataset: str
    keep_ogonki: bool = True
    lowercase: bool = False
    escape_emoji: bool = True

    def to_options(self):
        return f"--keep-ogonki {self.keep_ogonki} --lowercase {self.lowercase} --escape-emoji {self.escape_emoji}"

    def to_dirname(self):
        return f"{self.dataset}{fmtbl(self.keep_ogonki, '', '-noogonki')}{fmtbl(self.lowercase, '-lowercase', '')}{fmtbl(self.escape_emoji, '', '-emoji')}"


lm_experiments = [
        LMExperiment('reddit-pl'),
        LMExperiment('wiki'),
        LMExperiment('reddit-mix'),
        LMExperiment('reddit-pl', keep_ogonki=False),
        LMExperiment('reddit-pl', lowercase=True),
        LMExperiment('reddit-pl', keep_ogonki=False, lowercase=True),
        LMExperiment('wiki', keep_ogonki=False),
        LMExperiment('wiki', keep_ogonki=False, lowercase=True)
]


def create_script(DATA_DIR):
    DATA_DIR = Path(DATA_DIR)
    RAW_DIR=DATA_DIR / 'rawtexts'
    for e in lm_experiments:
        print(f'mkdir -p "{DATA_DIR / e.to_dirname()}"')
        lm = f'python preprocessing/reddit-preprocess.py --filename "{ RAW_DIR / e.dataset }.csv" --format csv --keep-numbers False --remove-asciiart False --output-format csv ' + e.to_options() + f' > "{ DATA_DIR / e.to_dirname()/ "cleaned.csv" }" &'
        trn = f'python preprocessing/reddit-preprocess.py --filename "{ RAW_DIR / "train_text.txt" }" --format lines --keep-numbers False --remove-asciiart False --output-format csv --twitter --keep-order ' + e.to_options() + f' > "{ DATA_DIR / e.to_dirname() / "cls_train.csv" }" &'
        tst = f'python preprocessing/reddit-preprocess.py --filename "{ RAW_DIR / "test_text.txt" }" --format lines --keep-numbers False --remove-asciiart False --output-format csv --twitter --keep-order' + e.to_options() + f' > "{ DATA_DIR / e.to_dirname() / "cls_test.csv" }" &'
        print(lm)
        print(trn)
        print(tst)
        print(f'cp "{RAW_DIR}/train_tags.txt" "{DATA_DIR}/{e.to_dirname()}"')
        print(f'cp "{RAW_DIR}/test_tags.txt" "{DATA_DIR}/{e.to_dirname()}"')

if __name__ == "__main__": fire.Fire(create_script)
