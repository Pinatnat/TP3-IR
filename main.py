from base64 import decode
from distutils.log import error
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor
import os
import json
from fastapi.middleware.cors import CORSMiddleware

security = HTTPBasic()
app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
        )

BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

BSBI_instance.load()
tfidf = Letor()

res = {"success" : None,"error" : None,"data" : []}

@app.get('/search')
async def search (query : str) :
    data_fin = {}
    docs = []
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 100):
        with open(os.path.join(os.getcwd(),doc.replace("\\","/"))) as file :
            data = file.read().replace('\n'," ")
            docs.append((doc,data))

    list_a = tfidf.calc(query, docs)
    res["data"] = [] 
    for doc in list_a:
        with open(os.path.join(os.getcwd(),doc.replace("\\","/")), "r") as read:
            res["data"].append({"doc": doc.replace("\\","/"), "content" : read.read().replace("\n"," ")})
    res["success"]=True,
    return res

@app.get('/a')
def test():
    return "test"

