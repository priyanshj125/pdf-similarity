import PyPDF2 # type: ignore
import re   # type: ignore   //import regular expression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
 

for x in os.listdir("."):
    print(x)