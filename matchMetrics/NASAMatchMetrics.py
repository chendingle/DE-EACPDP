import os

import pandas as pd


def NASAMatchMetrics():
    readPath = '.\\data\\NASA_unmatched\\'
    writePath = '.\\data\\NASA\\'
    files = os.listdir(readPath)
    for file in files:
        df = pd.read_csv(readPath + file)
        df = df[['LOC_EXECUTABLE', 'BRANCH_COUNT', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS', 'CYCLOMATIC_COMPLEXITY',
                 'DESIGN_COMPLEXITY', 'ESSENTIAL_COMPLEXITY', 'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY',
                 'HALSTEAD_EFFORT', 'HALSTEAD_ERROR_EST', 'HALSTEAD_LENGTH', 'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME',
                 'NUM_OPERANDS', 'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS', 'LOC_TOTAL', 'bug']]
        df.to_csv(writePath + file, index=False, header=True)
