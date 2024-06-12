"""
COMS W4705 - Natural Language Processing - Fall 2022
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum, isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.lhs_set = set()
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                    self.lhs_set.add(lhs)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF. LHS -> XP XP | terminal and RHS prob sum to one
        Otherwise return False. 
        """

        for lhs in self.lhs_set:
            sum = 0
            rules = self.lhs_to_rules[lhs]

            for triplet in rules: 
                sum += triplet[-1]
            
                if len(triplet[1]) > 1 and triplet[1][0].islower(): # check if non terminal has binary branching
                    return False
                
                elif len(triplet[1]) == 1 and triplet[1][0].isupper(): # check that terminal has one option
                    return False
            
            if not isclose(sum, 1): # check if probs sum to 1
                return False
 
        return True 


if __name__ == "__main__":
    with open("atis3.pcfg",'r') as grammar_file:
        grammar = Pcfg(grammar_file)

        if grammar.verify_grammar():
            print("Grammar is valid PCFG")
        else:
            print("Grammar is invalid")
        
        print(grammar.rhs_to_rules[('PP', 'PP')])
        
