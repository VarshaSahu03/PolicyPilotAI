import os
from policypilotai.graph import stream_user_question
from policypilotai.ingestion import ingestion_of_pdfs

if __name__ == "__main__":
    print("==PolicyPilotAI==")
    if not os.path.exists("./data/chroma_db"):
        print("DB not found, starting ingestion")
        ingestion_of_pdfs("data/policies",batch_size=100)
    else:
        print("db already exist")
        
    while True:
        #terminal take que
        qu=input("\nWrite your Question: ")
        if qu.lower() in ["exit", "quit"]:
            break
        if qu.strip()=="":
            print("Write any question plz!")
            continue
        print("\nAnswer: ", end="", flush=True) 
        for token in stream_user_question(qu):
            print(token, end="", flush=True)
        print("\n")  