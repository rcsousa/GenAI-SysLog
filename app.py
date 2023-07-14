from modules.utils import carregar_credenciais
from modules.utils import definir_embedder
from modules.utils import separar_texto
from langchain.vectorstores import FAISS

from kafka import KafkaConsumer, TopicPartition
import requests
import numpy as np
import faiss
import os


def main():

    carregar_credenciais()
    embeddings = definir_embedder()

    # Read the content from the Linux_2k.log into a variable
    with open('Linux_2k.log', 'r') as f:
        text = f.read()

    # Split the text into chunks of 1000 characters
    chunk = separar_texto(text)

    # Create a FAISS index with a dimensionality compatible with OpenAI embeddings
    db = FAISS.from_texts(chunk, embeddings)


#     Initialize a Kafka consumer
#    consumer = KafkaConsumer(
#        'syslog-ng',  # Kafka topic
#        bootstrap_servers=['localhost:9092'],  # Replace with your Kafka server
#        auto_offset_reset='earliest',  # Start reading from the beginning of the topic
#        enable_auto_commit=True,
#        group_id='syslog-ng-indexer',  # Consumer group ID
#        #consumer_timeout_ms=1000  # Stop iteration if no message is received after 1 second
#    )
#
    # If there is a message in the topic, create a FAISS index with a dimensionality compatible with OpenAI embeddings
  
#    for message in consumer:
#        text = message.value.decode('utf-8')
#        chunk = separar_texto(text)
#        db = FAISS.from_texts(chunk, embeddings)
    db.save_local("faiss_syslog_index_file")
#        query = "root"
#        docs = db.similarity_search(query)
#        print(docs[0].page_content)


if __name__ == '__main__':
    main()