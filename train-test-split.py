import pandas as pd

dataframe = pd.read_csv("./lesion-csv.csv")

# These testing images were manually selected to give a diverse range of testing examples
# Many examples are part of a sequence of images, so we try to include all images in a sequence
# in the test set, so we can fairly evaluate our model. 
# (eg. does our model generalize well, or is it just recognizing previously seen examples)
testing_image_ids = [
    "0001",
    "0002",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
    "0011",
    "0055",
    "0056",
    "0057",
    "0076",
    "0077",
    "0078",
    "0081",
    "0082",
    "0103",
    "0104",
    "0105",
    "0110",
    "0111",
    "0112",
    "0113",
    "0121",
    "0122",
    "0123",
    "0124",
    "0128",
    "0129",
    "0137",
    "0135",
    "0136",
    "0138",
    "0139",
    "0143",
    "0144",
    "0156",
    "0158",
    "0159",
    "0160",
    "0165",
    "0166",
    "0205",
    "0220",
    "0227",
    "0228",
    "0223",
    "0224",
    "0236",
    "0237",
    "0238",
    "0252",
    "0268",
    "0269",
    "0282",
    "0321",
    "0322",
    "0323",
    "0368",
    "0360",
    "0361",
    "0362",
    "0363",
    "0364",
    "0365",
    "0366",
    "0367",
    "0335",
    "0336",
    "0337",
    "0338",
    "0339",
    "0368",
    "0310",
    "0311",
    "0312",
    "0313",
    "0314",
    "0315",
    "0316",
    "0317",
    "0318",
    "0319",
    "0320",
]

print("Test Set Length: " + str(len(testing_image_ids)))

imageids_with_rotations = []

for imageid in testing_image_ids:
    imageids_with_rotations.append(imageid + "-0")
    imageids_with_rotations.append(imageid + "-1")
    imageids_with_rotations.append(imageid + "-2")
    imageids_with_rotations.append(imageid + "-3")

test_df = dataframe.loc[dataframe.imageId.isin(imageids_with_rotations)]
train_df = dataframe.loc[~dataframe.imageId.isin(imageids_with_rotations)]

print("---")
print("Testing DF Preview: ")
print(test_df.head())

print("---")
print("length of both train and test should equal length of orig DF")

length_matches = False
length_of_both = len(test_df) + len(train_df)
if length_of_both == len(dataframe):
    length_matches = True

print("Does it? :" + str(length_matches))

if length_matches:
    test_df.to_csv("test.csv")
    train_df.to_csv("train.csv")
