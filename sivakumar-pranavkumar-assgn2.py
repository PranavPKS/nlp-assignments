#import numpy as np
#from sklearn.cross_validation import KFold
#import os

file = 'berp-POS-training.txt'
the_list=[]

#Saving the training data in a 2D list
with open(file) as f:
    for line in f:
        if line not in ['\n']:
            inner_list = [ele.strip() for ele in line.split('\t')]
            the_list.append(inner_list)

#The Viterbi function
def viterbi(observations):
    '''
    Input: list of words in a sentence
    Output: list of predicted tags for the corresponding words
    '''
    vt={}
    bp={}
    if observations[0] not in the_dict:
        observations[0] = 'UNK'
    for tag in the_dict[observations[0]]:
        vt[tag,0] = start_prob[tag]*emission_prob[observations[0]][tag]

    i=1
    while i<len(observations):
        if observations[i] not in the_dict:
            observations[i] = 'UNK'
        for tag in the_dict[observations[i]]:
            vt[tag,i] = 0
            for key in the_dict[observations[i-1]]:
                x = transition_prob[key][tag]*emission_prob[observations[i]][tag]*vt[key,i-1]
                if x > vt[tag,i]:
                    vt[tag,i] = x
                    bp[tag,i] = key
        i=i+1
    vtf=0
    i=i-1
    bt=[]
    for tag in the_dict[observations[-1]]:
        if vt[tag,i] > vtf:
            vtf = vt[tag,i]
            bp['end',i] = tag
    bt.append(bp['end',i])
    j=0
    
    while i>0:
        bt.append(bp[bt[j],i])
        j=j+1
        i=i-1
    bt.reverse()
    
    return bt            


#the_list_full = np.load('the_list.npy')
#acc=[]
#kf = KFold(len(the_list_full), n_folds=10)
#for train_index, test_index in kf:
#    the_list_train, the_list_test = the_list_full[train_index], the_list_full[test_index]
#    the_list = the_list_train

i=0
the_dict={} #The main dictionary that has counts of each word with respect to its tags e.g. the_dict['i']['PRP'] returns the count of 'i' being a 'PRP'
tag_dict={} #counts of each tag in the document
word_freq={} #counts of each word
bi_dict={}  #counts of each bigram
start_count={} #counts of each tag being a start tag
sent_count=0 #total number of sentences
while i< len(the_list):
    try:
        if the_list[i][2] in the_dict[the_list[i][1]]: 
            the_dict[the_list[i][1]][the_list[i][2]] = the_dict[the_list[i][1]][the_list[i][2]] + 1
        else:
            the_dict[the_list[i][1]][the_list[i][2]] = 1
    except:
        the_dict[the_list[i][1]]={}
        the_dict[the_list[i][1]][the_list[i][2]] = 1

    try:
        tag_dict[the_list[i][2]] = tag_dict[the_list[i][2]] + 1
    except:
        tag_dict[the_list[i][2]] = 1

    try:
        word_freq[the_list[i][1]] = word_freq[the_list[i][1]] + 1
    except:
        word_freq[the_list[i][1]] = 1

    if the_list[i-1][1] == '.':
            sent_count = sent_count+1
            try:
                start_count[the_list[i][2]] = start_count[the_list[i][2]] + 1
            except:
                start_count[the_list[i][2]] = 1
    #Bigram counts    
    if the_list[i][1] != '.':
        try:
            if the_list[i+1][2] in bi_dict[the_list[i][2]]: 
                bi_dict[the_list[i][2]][the_list[i+1][2]] = bi_dict[the_list[i][2]][the_list[i+1][2]] + 1
            else:
                bi_dict[the_list[i][2]][the_list[i+1][2]] = 1
        except:
            if i+1 != len(the_list):
                bi_dict[the_list[i][2]]={}
                bi_dict[the_list[i][2]][the_list[i+1][2]] = 1    
    i=i+1

#Finding low frequent words
low_freq_words=[]

for word in word_freq:
    if word_freq[word] == 1:
        low_freq_words.append(word)

#update the dictionary, the list and the frequency set with UNK
the_dict['UNK'] = {}
for word in low_freq_words:
    for tag in the_dict[word]:
        try:
            the_dict['UNK'][tag] = the_dict['UNK'][tag] + the_dict[word][tag]
        except:
            the_dict['UNK'][tag] = the_dict[word][tag]
        del the_dict[word]
i=0
while i< len(the_list):
    if the_list[i][1] in low_freq_words:
        the_list[i][1] ='UNK'
    i=i+1

unk_count=0
for word in low_freq_words:
    unk_count = unk_count + word_freq[word]    
    del word_freq[word]
word_freq['UNK'] = unk_count


#Emission probabilities
emission_prob = {}
for word in word_freq:
    for tag in the_dict[word]:
        try:
            emission_prob[word][tag] = float(the_dict[word][tag]) / float(tag_dict[tag])
        except:
            emission_prob[word] = {}
            emission_prob[word][tag] = float(the_dict[word][tag]) / float(tag_dict[tag])

#Calculating the counts required for kneser-ney
tot_bi_count=0 #Total number of available bigram tag combinations
end_bi_count={} #count of each tag being a second tag in a bigram
for tag1 in tag_dict:
    for tag2 in tag_dict:
        if tag1 != '.':
            try:
                if tag2 in bi_dict[tag1]:
                    end_bi_count[tag2] = end_bi_count[tag2] + 1
            except:
                if tag2 in bi_dict[tag1]:
                    end_bi_count[tag2] = 1
            if tag2 in bi_dict[tag1]:
                tot_bi_count = tot_bi_count + 1

for tag in tag_dict:
    if tag not in end_bi_count:
        end_bi_count[tag]=1

#Transition probablities
#Kneser-Ney smoothing
transition_prob={}
for tag1 in bi_dict:
    for tag2 in end_bi_count:
        if tag1 != '.':
            try:
                if tag2 in bi_dict[tag1]:
                    lmda = (0.75,0.5)[bi_dict[tag1][tag2] == 1]
                    transition_prob[tag1][tag2] = (max(float(bi_dict[tag1][tag2])-lmda,0)/float(tag_dict[tag1])) + ((lmda/(float(tag_dict[tag1])))*float(len(bi_dict[tag1]))*(float(end_bi_count[tag2])/float(tot_bi_count)))
                else:
                    lmda = 0.75
                    transition_prob[tag1][tag2] = ((lmda/(float(tag_dict[tag1])))*float(len(bi_dict[tag1]))*(float(end_bi_count[tag2])/float(tot_bi_count)))
            except:
                transition_prob[tag1] ={}
                if tag2 in bi_dict[tag1]:
                    lmda = (0.75,0.5)[bi_dict[tag1][tag2] == 1]
                    transition_prob[tag1][tag2] = (max(float(bi_dict[tag1][tag2])-lmda,0)/float(tag_dict[tag1])) + ((lmda/(float(tag_dict[tag1])))*float(len(bi_dict[tag1]))*(float(end_bi_count[tag2])/float(tot_bi_count)))
                else:
                    lmda = 0.75
                    transition_prob[tag1][tag2] = ((lmda/(float(tag_dict[tag1])))*float(len(bi_dict[tag1]))*(float(end_bi_count[tag2])/float(tot_bi_count)))
#Transition probablities
#Laplace smoothing
#transition_prob={}
#for tag1 in tag_dict:
#    for tag2 in tag_dict:
#        if tag1 != '.':
#            try:
#                if tag2 in bi_dict[tag1]:
#                    transition_prob[tag1][tag2] = (float(bi_dict[tag1][tag2])+float(1)) / (float(tag_dict[tag1]) + float(len(tag_dict)))
#                else:
#                    transition_prob[tag1][tag2] = float(1) / (float(tag_dict[tag1]) + float(len(tag_dict)))
#            except:
#                transition_prob[tag1] ={}
#                if tag2 in bi_dict[tag1]:
#                    transition_prob[tag1][tag2] = (float(bi_dict[tag1][tag2])+float(1)) / (float(tag_dict[tag1]) + float(len(tag_dict)))
#                else:
#                    transition_prob[tag1][tag2] = float(1) / (float(tag_dict[tag1]) + float(len(tag_dict)))

#start probabilities smoothed by Laplace smoothing
start_prob={}
for key in tag_dict:
    if key in start_count:
        start_prob[key] = (float(start_count[key])+float(1)) / (float(sent_count)+float(len(tag_dict)))  
    else:
        start_prob[key] = float(1) / (float(sent_count)+float(len(tag_dict)))

#Testing the model on the given test-set data
the_list_test=[]
file = 'assgn2-test-set.txt'
with open(file) as f:
    for line in f:
        if line not in ['\n']:
            inner_list = [ele.strip() for ele in line.split('\t')]
            the_list_test.append(inner_list)

i=0
pass_text=[]
obt_tags=[]
#cor_tags=[]


while i<len(the_list_test):
    pass_text.append(the_list_test[i][1])
    if the_list_test[i][1] == '.':
        obt_tags.extend(viterbi(pass_text))
        pass_text=[]
    #cor_tags.append(the_list_test[i][2])
    i=i+1
if pass_text:
    obt_tags.extend(viterbi(pass_text))
#    count=0
#    for i in range(len(cor_tags)):
#        if obt_tags[i]==cor_tags[i]:
#            count=count+1
#    accy = float(count) / float(len(the_list_test))
#    print(accy)
#    acc.append(accy)
            
#print 'The accuracy is ', np.mean(acc)    

#Printing the given data with the obtained tags into a text file
thefile = open('sivakumar-pranavkumar-assgn2-test-output.txt', 'w')

i=0
while i<len(the_list_test):
    thefile.write("%s\t%s\t%s\n" % (the_list_test[i][0],the_list_test[i][1],obt_tags[i]))
    if the_list_test[i][1] == '.':
        thefile.write("\n")
    i=i+1
thefile.close()
