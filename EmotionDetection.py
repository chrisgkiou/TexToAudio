from deepface import DeepFace
import pandas as pd

# analyze the image
face_analysis = DeepFace.analyze(img_path=r'C:\Users\User\Desktop\happy_face_woman.png')
print(face_analysis)

df = pd.DataFrame(face_analysis)
print(df)

from pandas.io.json import json_normalize

# normalize the JSON output into a Pandas DataFrame
df = json_normalize(face_analysis)

# print the DataFrame
print(df)

