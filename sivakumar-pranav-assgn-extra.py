import string
import re

file1 = 'hotelT-train.txt'
file2 = 'hotelF-train.txt'
vocab=[]
#using regex to detect question marks and exclamation marks
question_mark_RE_str = '\?'
exclamation_point_RE_str = '\!'
# any combination of multiple exclamation points and question marks
interrobang_RE_str = '[\?\!]{2,}'

def get_data(file):
    '''
    Input: the text file with the reviews and their IDs
    Output: A list with ID of the review as the first element and
    a modified scanned word list as the second element
    '''
    with open(file) as f:
        rev_list = []
        for line in f:
            id, rev = line.split('\t')[0] , line.translate(None, string.punctuation).split('\t')[1].replace('\n','').split(' ')

            if len(re.findall(r'%s' % exclamation_point_RE_str, line)) > 0:
                rev.append("PUNCxEXCLAMATION_POINT")
            if len(re.findall(r'%s' % question_mark_RE_str, line)) > 0:
                rev.append("PUNCxQUESTION_MARK")
            if len(re.findall(r'%s' % interrobang_RE_str, line)) > 0:
                rev.append("PUNCxINTERROBANG")

            rev = [x for x in rev if x != '']
            inner_list = [id, rev]
            rev_list.append(inner_list)
    return rev_list

def get_count(rev):
    '''
    Input: output list from get_data
    Output: Dictionary with the counts of each word in the reviews of
    each class 
    '''
    i=0
    count={}
    while(i<len(rev)):
        for word in rev[i][1]:
            if word not in vocab:
                vocab.append(word)
            try:
                count[word] = count[word] + 1
            except:
                count[word] = 1
        i = i + 1
    return count


rev_T = get_data(file1)
rev_F = get_data(file2)
T_count = get_count(rev_T)
F_count = get_count(rev_F)
#Calculating the prior probabilities
prior_T = float(len(rev_T)) / float(len(rev_T) + len(rev_F))
prior_F = float(len(rev_F)) / float(len(rev_T) + len(rev_F))

#Removing stop-words
stop_words = ['and', 'a', 'we', 'room', 'for', 'that', 'I', 'of', 'hotel', 'had', 'it', 'to', 'were', 'at', 'in', 'my', 'the', 'was', 'The', 'on', 'with','is']
for sw in stop_words:
        vocab.remove(sw)
        del T_count[sw]
        del F_count[sw]
        
T_likelihood = {}
F_likelihood = {}
#Calculating the likelihood probabilities
for word in vocab:
    if word in T_count.keys():
        T_likelihood[word] = float(T_count[word] + 1) / float(sum(T_count.values()) + len(vocab))
    else:
        T_likelihood[word] = 1.0 / float(sum(T_count.values()) + len(vocab))

    if word in F_count.keys():
        F_likelihood[word] = float(F_count[word] + 1) / float(sum(F_count.values()) + len(vocab))
    else:
        F_likelihood[word] = 1.0 / float(sum(F_count.values()) + len(vocab))


test_rev = get_data('hotelDeceptionTest.txt')
output_file = open('sivakumar-pranavkumar-extra-out.txt', 'w')
i=0
while(i<len(test_rev)):
    prodp,prodn = 1.0,1.0
    for word in test_rev[i][1]:
        if word in vocab:
            prodp = prodp * T_likelihood[word]
            prodn = prodn * F_likelihood[word]
    if(prodp*prior_T > prodn*prior_F):
        output_file.write("%s\tT\n" % test_rev[i][0])
    else:
        output_file.write("%s\tF\n" % test_rev[i][0])
    i = i+1
output_file.close()
