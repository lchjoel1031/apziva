Potential Talent
In this project, I used different models -- bag of words, tf-idf, word2vec, glove, Bert -- to convert text into vector embedding, such that I can use cosine similarity to find similar texts for recommedation. I select Bert as the production model. The rerank can be done by giving starred candidate id, and the job_title of the starred candidate will be used for reranking.

A notebook walking through all steps is saved in notebook/, and the source code to run the training is saved in src/train.py, as well as the source code to apply the bert model to new dataset in src/app.py
