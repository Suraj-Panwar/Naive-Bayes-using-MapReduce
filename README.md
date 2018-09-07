# Naive-Bayes-using-MapReduce

This project undertakes the task of Naive Bayes implementation using MapReduce and Local Machine implementation, the data set is DBPedia. The following are brief descriptions of the project files:

1. Local_Naive_Bayes.py
   The python script implements the Naive Bayes algorithm on the train, test and dev dataset and records the corresponding accuracies along    with time taken to train the algorithm on local machine. 

2. Mapreduce_Naive_Bayes.py
   The python script implements the Naive Bayes algorithm on the train, test and dev dataset and records the corresponding accuracies along    with time taken to train the algorithm. The dictionary is prepared on hadoop mapreduce platform.
   
3. mapper.py
   The mapper python script is used to mapping using hadoop streaimng for generating the (label, word, 1) stream output.
   
4. reducer.py
   The reducer python script is used to mapping using hadoop streaimng for generating the (label, word, count) stream output.

5. log.txt
   Log file containing hadoop log records.
   
6. main_dictionary.pickle/ dictionary.pickle
   Pickle files for dictinary storage so that model dosent have to be trained every time.

