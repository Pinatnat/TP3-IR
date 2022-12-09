from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
BSBI_instance.load()

with open("res.txt","w") as wr :
    for query in queries:
        tfidf = Letor()
        docs = []
        wr.write("Query  : "+ query)
        wr.write("\nResults  : ")
        count = 0
        for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 100):
            if count <= 9:
                wr.write(f"\n{doc:30} {score:>.3f}")
            count+=1
            with open(doc) as file :
                data = file.read().replace('\n'," ")
                docs.append((doc,data))
        wr.write("\n\n")

        wr.write("LETOR")    
        tfidf.process_rank(query,docs,wr)
        wr.write("\n\n")

        wr.write("-----------------------------------\n\n")
    file.close()
