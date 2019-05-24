import pandas as pd

# 所有标签
csv_path = "../dataset/Labels/AllLabels.csv"

# 训练集和测试集
train_txt = "../dataset/DataSet/Train.txt"
validation_txt = "../dataset/DataSet/Validation.txt"
test_txt = "../dataset/DataSet/Test.txt"


# 读取所有标签
csv = pd.read_csv(csv_path)
engagement = csv[["ClipID", "Engagement"]]

# 测试集
train = pd.read_table(train_txt, header=None)
train.columns = ["ClipID"]
# 验证集
validation = pd.read_table(validation_txt, header=None)
validation.columns = ["ClipID"]
# 测试集
test = pd.read_table(test_txt, header=None)
test.columns = ["ClipID"]

# 合并训练集
train = pd.merge(train, engagement, how="left", on="ClipID")
print(len(train))
# 合并验证集
validation = pd.merge(validation, engagement, how="left", on="ClipID")
print(len(validation))
# 合并测试集
test = pd.merge(test, engagement, how="left", on="ClipID")
print(len(test))

# 进一步融合
test2 = pd.read_csv("../dataset/Labels/TestLabels.csv")[["ClipID", "Engagement"]]
train2 = pd.read_csv("../dataset/Labels/TrainLabels.csv")[["ClipID", "Engagement"]]
validation2 = pd.read_csv("../dataset/Labels/ValidationLabels.csv")[["ClipID", "Engagement"]]

test = pd.merge(test, test2, how="outer", on=["ClipID", "Engagement"])
train = pd.merge(train, train2, how="outer", on=["ClipID", "Engagement"])
validation = pd.merge(validation, validation2, how="outer", on=["ClipID", "Engagement"])
test.dropna(axis=0, how="any", inplace=True)
train.dropna(axis=0, how="any", inplace=True)
validation.dropna(axis=0, how="any", inplace=True)

test.to_csv("../dataset/Labels/mysTest.csv", index=None)
train.to_csv("../dataset/Labels/mysTrain.csv", index=None)
validation.to_csv("../dataset/Labels/mysValidation.csv", index=None)
print(len(test)+len(train)+len(validation))
