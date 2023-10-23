Potential Talent
In this project, I used different models -- bag of words, tf-idf, word2vec, glove, Bert -- to convert text into vector embedding, such that I can use cosine similarity to find similar texts for recommedation. I select Bert as the production model. The rerank can be done by giving starred candidate id, and the job_title of the starred candidate will be used for reranking.

A notebook walking through all steps is saved in notebook/, and the source code to run the training is saved in src/train.py, as well as the source code to apply the bert model to new dataset in src/app.py

For Bonuses questions:
To rerank after a candidate is starred, simply replace the keyword with the starred candidate's job_title.
To filter out candidates that should not be in the list in the first place, simply remove candidates that do not have any word of the keywords in their job_title.
