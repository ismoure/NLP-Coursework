This code implements the Cocke–Kasami–Younger (CKY) algorithm in python. I wrote all the code except for the evaluate_parser.py
and the format checking functions in cky.py. This was for my Natural Language Processing class that I took last Fall, 2022. grammar.py
contains the class representation of the probabalistic context free grammar (PCFG) and can check if a grammar is valid. The atis3 PCFG 
is given here as an example. cky.py implements the CKY algorithm given a valid PCFG and a string input. It can return the table or a 
tree representation of the most likely parse for the sentence. 