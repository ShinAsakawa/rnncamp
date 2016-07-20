# rnncamp

# Acknowledgment

[RNN camp](http://connpass.com/event/35055/) was intended to show you how RNNs works.
I deployed the directories "01cpp" and "01python".  The "01cpp" directory includes RNNLM originally developed Tomas Mikolov.
There are 9 files in the directory.  You need install [srilm-toolkit](http://www.speech.sri.com/projects/srilm/download.html) to process the rnnlm.  You must register you ID before you use the srilm-toolkit.  Two files, **ngram** and **ngram-count** are required to deal with the rnnlm.  

You can see two files, "min-char-rnn.py" and "elman.py" in 01python directory.  "min-char-rnn.py" was written by Andrej Karpathy (@karpathy) Stanford university.  I wrote "elman.py" w.r.t. on Karpathy's python code.

N.B.  You must install 'termcolor' in your python distribution through pip.
```
pip install --upgrade termcolor
```
Numpy and matplotlib are also required for you to run "elman.py"

Have a fun with these files. lol

Shin
