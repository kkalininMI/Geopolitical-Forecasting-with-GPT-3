# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:08:38 2022

@author: kkalinin
"""
import os
import openai
import pandas as pd
import numpy as np
import re
import statistics
from itertools import permutations
import pickle
import re

openai.api_key = os.environ["OPENAI_API_KEY"]

########################################
#       Territorial Dimension          #
########################################

os.chdir("C:/Users/Kirill/Desktop/Predictioneer's game")
qdat = pd.read_csv('territorial_dimension.csv')

#Russia
insertname="Russia"
insertphrase=""
comput_results_Russia = run_GPT3 (qdat, insertname, insertphrase)

#Ukraine
insertname="Ukraine"
insertphrase=""
comput_results_Ukraine = run_GPT3 (qdat, insertname, insertphrase)

#Belarus
insertname="Belarus"
insertphrase=""
comput_results_Belarus = run_GPT3 (qdat, insertname, insertphrase)

#Kazakhstan
insertname="Kazakhstan"
insertphrase=""
comput_results_Kazakhstan = run_GPT3 (qdat, insertname, insertphrase)

#Turkey
insertname="Turkey"
insertphrase=""
comput_results_Turkey = run_GPT3 (qdat, insertname, insertphrase)

#United States
insertname="United States"
insertphrase=""
comput_results_US = run_GPT3 (qdat, insertname, insertphrase)

#China
insertname="China"
insertphrase=""
comput_results_China = run_GPT3 (qdat, insertname, insertphrase)

#EU
insertname="European Union"
insertphrase=""
comput_results_EU = run_GPT3 (qdat, insertname, insertphrase)

#NATO
insertname="NATO"
insertphrase=""
comput_results_NATO = run_GPT3 (qdat, insertname, insertphrase)

#Merge data together
variables = ["player", "chosen", "score1", "scorew"]
comput_results_Russia["player"] = "Russia"; dat_Russia = comput_results_Russia[variables]
comput_results_Ukraine["player"] = "Ukraine"; dat_Ukraine = comput_results_Ukraine[variables]
comput_results_Belarus["player"] = "Belarus"; dat_Belarus = comput_results_Belarus[variables]
comput_results_Kazakhstan["player"] = "Kazakhstan"; dat_Kazakhstan = comput_results_Kazakhstan[variables]
comput_results_Turkey["player"] = "Turkey"; dat_Turkey = comput_results_Turkey[variables]
comput_results_US["player"] = "United States"; dat_US = comput_results_US[variables]
comput_results_China["player"] = "China"; dat_China = comput_results_China[variables]
comput_results_EU["player"] = "European Union"; dat_EU = comput_results_EU[variables]
comput_results_NATO["player"] = "NATO"; dat_NATO = comput_results_NATO[variables]


territorial_combined = pd.concat([dat_Russia, 
                                    dat_Ukraine,
                                    dat_Belarus,
                                    dat_Kazakhstan,
                                    dat_Turkey,
                                    dat_US,
                                    dat_China,
                                    dat_EU,
                                    dat_NATO
                                    ])

#territorial_combined.to_csv("territorial_processed.csv")

########################################
#              Domestic                #
########################################

qdat = pd.read_csv('domestic_dimension.csv')

#Vladimir Putin
insertname="Vladimir Putin"
insertphrase=""
comput_results_Putin = run_GPT3 (qdat, insertname, insertphrase)

#Security services (e.g., FSB, FSO, GRU)
insertname="Russian security services (e.g., FSB, FSO, GRU)"
insertphrase=""
comput_results_security = run_GPT3 (qdat, insertname, insertphrase)

#Russian military (Ministry of Defense)
insertname="Russian military (ministry of defense)"
insertphrase=""
comput_results_military = run_GPT3 (qdat, insertname, insertphrase)

#Federal bureaucracy
insertname="Russian federal bureaucracy"
insertphrase=""
comput_results_federal_bureaucracy = run_GPT3 (qdat, insertname, insertphrase)

#Turkey
insertname="Russian regional bureaucracy"
insertphrase=""
comput_results_regional_bureaucracy = run_GPT3 (qdat, insertname, insertphrase)

#Russian nationalists
insertname="Russian nationalists (e.g., Igor Strelkov, Alexander Dugin)"
insertphrase=""
comput_results_nationalists = run_GPT3 (qdat, insertname, insertphrase)

#Systemic liberals (e.g., Alexei Kudrin, Herman Gref)
insertname="Russian systemic liberals (e.g., Alexei Kudrin, Herman Gref)"
insertphrase=""
comput_results_liberals = run_GPT3 (qdat, insertname, insertphrase)

#Private armies (e.g., Yevgeny Prigozhin, Ramzan Kadyrov)
insertname="Russian private armies (e.g., Yevgeny Prigozhin, Ramzan Kadyrov)"
insertphrase=""
comput_results_domestic_private_armies = run_GPT3 (qdat, insertname, insertphrase)

#Business elites (Private Business)
insertname="Russian business elites (private business)"
insertphrase=""
comput_results_private_business = run_GPT3 (qdat, insertname, insertphrase)

#Business elites (State-Owned Enterprises)
insertname="Russian business elites (state-owned enterprises)"
insertphrase=""
comput_results_state_business = run_GPT3 (qdat, insertname, insertphrase)

#Merge data together
variables = ["player", "chosen", "score1", "scorew"]
comput_results_Putin["player"] = "Putin"; dat_Putin = comput_results_Putin[variables]
comput_results_security["player"] = "Security"; dat_Security = comput_results_security[variables]
comput_results_military["player"] = "Military"; dat_Military = comput_results_military[variables]
comput_results_federal_bureaucracy["player"] = "Federal"; dat_federal_bureaucracy = comput_results_federal_bureaucracy[variables]
comput_results_regional_bureaucracy["player"] = "Regional"; dat_regional_bureaucracy = comput_results_regional_bureaucracy[variables]
comput_results_nationalists["player"] = "Nationalists"; dat_nationalists = comput_results_nationalists[variables]
comput_results_liberals["player"] = "Liberals"; dat_liberals = comput_results_liberals[variables]
comput_results_domestic_private_armies["player"] = "Private Armies"; dat_domestic_private_armies = comput_results_domestic_private_armies[variables]
comput_results_private_business["player"] = "Business"; dat_private_business = comput_results_private_business[variables]
comput_results_state_business["player"] = "State"; dat_state_business = comput_results_state_business[variables]

domestic_combined = pd.concat([dat_Putin, 
                                    dat_Security,
                                    dat_Military,
                                    dat_federal_bureaucracy,
                                    dat_regional_bureaucracy,
                                    dat_nationalists,
                                    dat_liberals,
                                    dat_domestic_private_armies,
                                    dat_private_business,
                                    dat_state_business
                                    ])

#domestic_combined.to_csv("domestic_processed.csv")


########################################
#          Nuclear dimension           #
########################################
qdat = pd.read_csv('nuclear_dimension.csv')

#Russia
insertname="Russia"
insertphrase=""
comput_results_Russia_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#Ukraine
insertname="Ukraine"
insertphrase=""
comput_results_Ukraine_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#Belarus
insertname="Belarus"
insertphrase=""
comput_results_Belarus_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#Kazakhstan
insertname="Kazakhstan"
insertphrase=""
comput_results_Kazakhstan_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#Turkey
insertname="Turkey"
insertphrase=""
comput_results_Turkey_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#United States
insertname="United States"
insertphrase=""
comput_results_US_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#China
insertname="China"
insertphrase=""
comput_results_China_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#EU
insertname="European Union"
insertphrase=""
comput_results_EU_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#NATO
insertname="NATO"
insertphrase=""
comput_results_NATO_nuclear = run_GPT3 (qdat, insertname, insertphrase)

#Merge data together
variables = ["player", "chosen", "score1", "scorew"]

comput_results_Russia_nuclear["player"] = "Russia"; dat_Russia_nuclear = comput_results_Russia_nuclear[variables]
comput_results_Ukraine_nuclear["player"] = "Ukraine"; dat_Ukraine_nuclear = comput_results_Ukraine_nuclear[variables]
comput_results_Belarus_nuclear["player"] = "Belarus"; dat_Belarus_nuclear = comput_results_Belarus_nuclear[variables]
comput_results_Kazakhstan_nuclear["player"] = "Kazakhstan"; dat_Kazakhstan_nuclear = comput_results_Kazakhstan_nuclear[variables]
comput_results_Turkey_nuclear["player"] = "Turkey"; dat_Turkey_nuclear = comput_results_Turkey_nuclear[variables]
comput_results_US_nuclear["player"] = "United States"; dat_US_nuclear = comput_results_US_nuclear[variables]
comput_results_China_nuclear["player"] = "China"; dat_China_nuclear = comput_results_China_nuclear[variables]
comput_results_EU_nuclear["player"] = "European Union"; dat_EU_nuclear = comput_results_EU_nuclear[variables]
comput_results_NATO_nuclear["player"] = "NATO"; dat_NATO_nuclear = comput_results_NATO_nuclear[variables]


nuclear_combined = pd.concat([dat_Russia, 
                                    dat_Ukraine,
                                    dat_Belarus,
                                    dat_Kazakhstan,
                                    dat_Turkey,
                                    dat_US,
                                    dat_China,
                                    dat_EU,
                                    dat_NATO
                                    ])
#nuclear_combined.to_csv("nuclear_processed.csv")




########################################
#               FUNCTIONS              #
########################################
def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def call_GPT3(prompt):
    kwargs = {"engine":"text-davinci-003", 
              "temperature":0, 
              "max_tokens":1,
              "top_p":1,
              "frequency_penalty":0, 
              "presence_penalty":0, 
              "logprobs":10}
     
    response = openai.Completion.create(prompt=prompt, **kwargs)
    scores = pd.DataFrame([response["choices"][0]["logprobs"]["top_logprobs"][0]]).T
    scores.columns = ["logprob"]
    scores["%"] = scores["logprob"].apply(lambda x: 100*np.e**x)
    scores = scores.sort_values(by ='%', ascending=False)
    score_names = [re.sub(r"^\s+|\s+$", "", i).strip()  for i in scores.index]
    chosen = [letter for letter in score_names][0]

    chosen_score = scores[['%']].iloc[0]
    
    res = []; res.append(prompt); res.append(score_names); res.append(list(scores["%"])); res.append(chosen);  res.append(chosen_score);  
    res.append(list(pd.DataFrame([response["id"]])[0])); res.append(list(pd.DataFrame([response["model"]])[0]));
    res.append(list(pd.DataFrame([response["object"]])[0]))
    res = flatten(res)
    
    return(res)


def create_list_questions(qdat, insertname, insertphrase):
    qdat.set_index("Index", inplace=True, drop=False)
    list_of_questions = []
    for question in range(1, qdat.shape[0]+1, 1):
        
        newlist = [x for x in qdat.loc[[question]].values.tolist()[0] if pd.isnull(x) == False]
 
        questionW = newlist[3]

        if insertname!="" or insertphrase!="":
            questionW = re.sub(r"\[PLACEHOLDER\]", insertname, questionW)
            questionW = insertphrase + " " + questionW                

        output = str(questionW) + '\n\nAnswer:'
        list_of_questions.append(output)
        
    return(list_of_questions)

def run_GPT3(qdat, insertname, insertphrase):
    list_questions = create_list_questions(qdat=qdat, 
                                           insertname=insertname, 
                                           insertphrase=insertphrase)
    data = []

    for question in list_questions:
        print (question)
        gpt3_response = call_GPT3(question)
        data.append(gpt3_response) 
    
        
    res_dat = pd.DataFrame(data, columns = ['question', 
                                            'answer1','answer2','answer3','answer4', 'answer5', 
                                            'score1','score2','score3','score4', 'score5', 
                                            "chosen", "chosen_score", "id", "model", "object"])
    
    res_m = res_dat[['answer1','answer2','answer3','answer4', 
                     'answer5', 'score1','score2','score3','score4', 'score5',]]
    res_m = res_m.apply(pd.to_numeric, errors='coerce')
    res_m.fillna(0, inplace=True)
    
    res_mn = res_m.to_numpy().astype(float)
    
    res_dat['scorew'] = list((res_mn[:, [0,1,2,3,4]] * res_mn[:, [5,6,7,8,9]]).sum(axis=1)/100)
  
    return(res_dat)