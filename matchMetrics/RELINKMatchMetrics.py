import os

import pandas as pd


def RELINKMatchMetrics():
    readPath = '.\\data\\RELINK_unmatched\\'
    writePath = '.\\data\\RELINK\\'
    files = os.listdir(readPath)
    for file in files:
        df = pd.read_csv(readPath + file)
        df = df[['CountLineCode', 'AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential',
                 'AvgLine', 'AvgLineBlank', 'AvgLineCode', 'AvgLineComment', 'CountClassBase', 'CountClassCoupled',
                 'CountClassDerived', 'CountDeclClassMethod', 'CountDeclClassVariable', 'CountDeclInstanceMethod',
                 'CountDeclInstanceVariable', 'CountDeclMethod', 'CountDeclMethodAll', 'CountDeclMethodPrivate',
                 'CountDeclMethodProtected', 'CountDeclMethodPublic', 'CountLine', 'CountLineBlank',
                 'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment', 'CountSemicolon', 'CountStmt',
                 'CountStmtDecl', 'CountStmtExe', 'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict',
                 'MaxInheritanceTree', 'PercentLackOfCohesion', 'RatioCommentToCode', 'SumCyclomatic',
                 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential', 'isDefective']]
        df.to_csv(writePath + file, index=False, header=True)
