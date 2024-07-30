import PyPDF2 # type: ignore
import re   # type: ignore   //import regular expression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def Pagetextextract(pdf_path):
  
   str = ""
   with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        numberofpage=len(reader.pages)

        for page_num in range(numberofpage):
            page = reader.pages[page_num]
            str += page.extract_text() or ''
   return str

    

def featuresextraction(str):
   
   invoice_number = re.search(r'Rechnung Nr\.\s*(\d+)', str)
   customer_number = re.search(r'Kundennummer\s*(\d+)', str)
   debitor_account = re.search(r'Debitorenkonto\s*(\d+)', str)
   vat_id = re.search(r'Ihre USt.-ID\s*([A-Z0-9]+)', str)
   invoice_date = re.search(r'Datum\s*(\d{2}\.\d{2}\.\d{4})', str)
   delivery_date = re.search(r'Lieferdatum vom\s*(\d{2}\.\d{2}\.\d{4})', str)
   total_amount = re.search(r'Zwischensumme EUR\s*([\d,]+)', str)
   vat_amount = re.search(r'MwSt Betrag EUR\s*([\d,]+)', str)
   total_invoice_amount = re.search(r'Rechnungsbetrag EUR\s*([\d,]+)', str)
   stop_words = set(["the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "at", "by", "an", "from"])
   keywords = set(re.findall(r"\b\w+\b", str.lower())) - stop_words
   features = {
        'str': str,
        'invoice_number': invoice_number.group(1) if invoice_number else None,
        'customer_number': customer_number.group(1) if customer_number else None,
        'debitor_account': debitor_account.group(1) if debitor_account else None,
        'vat_id': vat_id.group(1) if vat_id else None,
        'invoice_date': invoice_date.group(1) if invoice_date else None,
        'delivery_date': delivery_date.group(1) if delivery_date else None,
        'total_amount': float(total_amount.group(1).replace(',', '.')) if total_amount else None,
        'vat_amount': float(vat_amount.group(1).replace(',', '.')) if vat_amount else None,
        'total_invoice_amount': float(total_invoice_amount.group(1).replace(',', '.')) if total_invoice_amount else None,
        'keywords': keywords  #  Extracted keywords

    }
   return features
def cosineSimilarity(feature1,feature2):  #applying cosine similarity

   vectorizer = TfidfVectorizer()
   tfidf_matrix = vectorizer.fit_transform([feature1['str'], feature2['str']])
   cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
   return cosine_sim[0][0]
def jaccardSimilarity(feature1,feature2):   #applying the jaccard similarity
    set1 = feature1['keywords']
    set2 = feature2['keywords']
    s1_intersection_2 = set1.intersection(set2)
    s1_union_s2 = set1.union(set2)
    if len(s1_union_s2) == 0:
        return 0.0
    return len(s1_intersection_2) / len(s1_union_s2)
def MostSimilarInvoice(new_invoice_path, existing_invoices):   #taking average of both simliratity
    # Extract text and features from the new invoice
    new_invoice_text = Pagetextextract(new_invoice_path)
    new_invoice_features = featuresextraction(new_invoice_text)
    
    max_similarity = 0
    most_similar_invoice = None
    
    for invoice in existing_invoices:
        # Compute similarity scores
        cosine_sim = cosineSimilarity(new_invoice_features, invoice)
        jaccard_sim = jaccardSimilarity(new_invoice_features, invoice)
        
        # Average the two similarity scores
        similarity = (cosine_sim + jaccard_sim) / 2
        
        if similarity > max_similarity:
            max_similarity = similarity
            print("this is changing")
            most_similar_invoice = invoice
    
    return most_similar_invoice, max_similarity

train_folder = "train"                           #read train folder
training_invoices = []
for filename in os.listdir(train_folder):
    if filename.endswith(".pdf"):
        
        file_path = os.path.join(train_folder, filename)
        text = Pagetextextract(file_path)
        features = featuresextraction(text)
        features['file_name'] = filename

        filename_dis=[{'file_name': filename}]
        training_invoices.append(features)


test = "test"                                     #read test folder
results = []
for filename in os.listdir(test):
    if filename.endswith(".pdf"):
        file_path = os.path.join(test, filename)
        text = Pagetextextract(file_path)
        features = featuresextraction(text)                        
        most_similar_invoice, similarity_score = MostSimilarInvoice(file_path, training_invoices)
        results.append((filename, most_similar_invoice, similarity_score))        


# Output the results
for result in results:
    test_invoice, most_similar_invoice, similarity_score = result

    print(f"Test Invoice: {test_invoice}")
    print(f"Most Similar Training Invoice: {most_similar_invoice['file_name']}")
    print(f"Similarity Score: {similarity_score}")
    print("\n")

      

    

