# coding: utf-8
# !pip install scikit-image

import os
from PIL import Image
import numpy as np
import pylab as plt
import re
from IPython.display import display
import math
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import load_model
import itertools
import distance

from enchant import *
from enchant.tokenize import *

'''
depth_first_search finds the connected component of pixels below a certain threshold
value that includes the input pixel index (i,j) on the numpy array - array.
'''

def depth_first_search(empty, array, i, j, dark, thin, white, threshold):
   m = i
   n = j
   p = 0;
   ls,ws = empty.shape
   empty[m,n] = white

   if array[max(m-thin,0),n] <= threshold and array[min(m+thin,ls-1),n] <= threshold and array[m,max(n-thin,0)] <= threshold and array[m,min(n+thin,ws-1)] <= threshold:
       return_array = np.array([[m],[n]])
   elif thin > 1 and array[max(m-thin+1,0),min(n+thin-1,ws-1)] <= threshold and array[min(m+thin-1,ls-1),max(n-thin+1,0)] <= threshold and array[max(m-thin+1,0),max(n-thin+1,0)] <= threshold and array[min(m+thin-1,ls-1),min(n+thin-1,ws-1)] <= threshold:
       return_array = np.array([[m],[n]])
   else:
       return_array = None

   if m < ls-1 and array[m+1,n] <= threshold and empty[m+1,n] == dark:
       (empty, dark_pixels) = depth_first_search(empty, array, m+1, n, dark, thin, white, threshold)
       if return_array is not None and dark_pixels is not None:
           return_array = np.concatenate([return_array, dark_pixels], axis = 1)
       elif dark_pixels is not None:
           return_array = dark_pixels

   if m > 0  and array[m-1,n] <= threshold and empty[m-1,n] == dark:
       (empty, dark_pixels) = depth_first_search(empty, array, m-1, n, dark, thin, white, threshold)
       if return_array is not None and dark_pixels is not None:
           return_array = np.concatenate([return_array, dark_pixels], axis = 1)
       elif arr is not None:
           return_array = dark_pixels

   if n < ws-1 and array[m,n+1] <= threshold and empty[m,n+1] == dark:
       (empty, dark_pixels) = depth_first_search(empty, array, m, n+1, dark, thin, white, threshold)
       if return_array is not None and dark_pixels is not None:
           return_array = np.concatenate([return_array, dark_pixels], axis = 1)
       elif arr is not None:
           return_array = dark_pixels

   if n > 0 and array[m,n-1] <= threshold and empty[m,n-1] == dark:
       (empty, dark_pixels) = depth_first_search(empty, array, m, n-1, dark, thin, white, threshold)
       if return_array is not None and dark_pixels is not None:
           return_array = np.concatenate([return_array, dark_pixels], axis = 1)
       elif arr is not None:
           return_array = dark_pixels

   return (empty, return_array)

'''
letter_extract segments the input text image into character images and returns these images in the order (LTR) they appear on the text,
adding newline and blank space characters when necessary.
Parameters:
threshold: The number that determines which pixels should be classified as 'dark' and included in a connected component
pixel_count: The number that sets the minimum number of pixels in connected component for that component to be considered as a character
min_words: The minimum number of characters a line can have. Lines with less number of characters are discarded and these characters are not returned. (except the last line)
min_seperation_coeff: This parameter is used while calculating min_seperation which is the minimum number of pixels between two different words
h_ratio: 1/ratio where ratio is the maximum ratio of the widths of the bounding boxes for two connected components belonging to the same character
w_ratio: 1/ratio where ratio is the maximum ratio of the heights of the bounding boxes for two connected components belonging to the same character 

'''

def letter_extract(imname, dark = 0, white = 255, thin = 0, lettercolor = 0, threshold = 0, pixel_count = 5, min_words = 3, min_seperation_coeff = 2, h_ratio = 5, w_ratio = 2):

   array = np.array(Image.open(imname).convert("L"))
   length, width = array.shape
   empty = np.zeros((length, width))
   images = []
   letternum = 0
   averagew = 0
   averagel = 0
   lines = []
   ws = []
   retarr = []
   moving_waverage = 0
   count_moving_waverage = 0
   min_seperation = 0

   for i in range(length):
       for j in range(width):

           if empty[i,j] == 0 and array[i,j] <= threshold:

               #Find the connected component
               (empty, arr) = depth_first_search(empty, array, i, j, dark, thin, white, threshold)
               (_, lettercount) = arr.shape

               if arr is not None and lettercount > pixel_count:
                   maxh = np.amax(arr[0,:])
                   minh = np.amin(arr[0,:])
                   maxw = np.amax(arr[1,:])
                   minw = np.amin(arr[1,:])

                   #Combine connected components corresponding to the same character
                   if len(images) > 0:

                       delindex = []
                       for idx in range(len(images)-1,-1,-1):
                           image = images[idx]
                           omaxh, ominh, omaxw, ominw = image[0]
                           oarr = image[2]
                           (l,w) = oarr.shape
                           if ((maxw >= omaxw or omaxw - maxw < (omaxw - ominw)/w_ratio) and (minw <= ominw or minw - ominw < (omaxw - ominw)/w_ratio)) and ((minh - omaxh > 0 and minh - omaxh < (maxh - minh)/h_ratio) or (minh <= ominh and omaxh <= maxh)):
                               cond = False
                               minh = ominh
                               arr = np.concatenate([arr,oarr], axis = 1)
                               delindex.append(idx)
                           elif ((maxw >= omaxw or omaxw - maxw < (omaxw - ominw)/w_ratio) and (minw <= ominw or minw - ominw < (omaxw - ominw)/w_ratio)) and ((ominh - maxh > 0 and ominh - maxh < (maxh - minh)/h_ratio) or (minh <= ominh and omaxh <= maxh)):
                               cond = False
                               maxh = omaxh
                               arr = np.concatenate([arr,oarr], axis = 1)
                               delindex.append(idx)

                       for deli in delindex:
                           del images[deli]
                           letternum -= 1

                   s = arr.shape;
                   letternum = letternum + 1
                   newimarr = np.ones((maxh-minh+1,maxw-minw+1))*white;
                   for k in range(s[1]):
                       newimarr[arr[0,k]-minh,arr[1,k]-minw] = lettercolor;
                   images.append(((maxh, minh, maxw, minw),newimarr,arr))
                   averagew += maxw - minw
                   averagel += maxh - minh

   averagew = math.ceil(averagew/letternum)
   averagel = math.ceil(averagel/letternum)

   #Group character images into arrays corresponding to their line on the text
   for i in range(len(images)):
       image = images[i]
       omaxh, ominh, omaxw, ominw = image[0]
       oarr = image[1]
       if i == 0:
           linnum = 1;
           lines.append([image])
           ws.append([ominw])
       elif abs(prevomaxh-omaxh) > averagel:
           linnum += 1;
           lines.append([image])
           ws.append([ominw])
       else:
           lines[-1].append(image)
           ws[-1].append(ominw)
       prevomaxh = omaxh


   #Add newline and newword markers to identify lines and words on the text
   for i in range(len(lines)):
       sindex = np.argsort(np.array(ws[i]))
       if len(sindex) > min_words or i == len(lines)-1:
           if i != 0:
               retarr.append("newline")
           for idxx in range(len(sindex)):
               j = sindex[idxx]
               image1 = lines[i][j]
               maxh1, minh1, maxw1, minw1 = image1[0]
               retarr.append(image1[1])
               if idxx != len(sindex)-1:
                   jp1 = sindex[idxx+1]
                   image2 = lines[i][jp1] 
                   maxh2, minh2, maxw2, minw2 = image2[0]                    
                   if moving_waverage == 0 and minw2 - maxw1 > averagew:
                       retarr.append("newword")
                   elif moving_waverage > 0 and minw2 - maxw1 >= min_seperation:
                       retarr.append("newword")
                   else:
                       count_moving_waverage += 1
                       moving_waverage += min_seperation_coeff*max(0,minw2 - maxw1)
                       min_seperation = moving_waverage / count_moving_waverage

   return retarr


def handle_resize(extracted, focus_factor):
   if focus_factor % 2 == 1: 
       raise ValueError("Focus factor should be an odd integer!")
   return_list = []
   for element in extracted:
       if type(element) != str:
           downsampled_element = misc.imresize(element,[focus_factor,focus_factor],'bilinear')
           tuples = [((128-focus_factor)//2,(128-focus_factor)//2),((128-focus_factor)//2,(128-focus_factor)//2)]
           resized_element = np.pad(downsampled_element, tuples, mode='constant', constant_values=255)
           return_list.append(resized_element)
       else:
           return_list.append(element)
   return return_list
def handle_resize(extracted, focus_factor):
   if focus_factor % 2 == 1: 
       raise ValueError("Focus factor should be an odd integer!")
   return_list = []
   for element in extracted:
       if type(element) != str:
           downsampled_element = misc.imresize(element,[focus_factor,focus_factor],'bilinear')
           tuples = [((128-focus_factor)//2,(128-focus_factor)//2),((128-focus_factor)//2,(128-focus_factor)//2)]
           resized_element = np.pad(downsampled_element, tuples, mode='constant', constant_values=255)
           return_list.append(resized_element)
       else:
           return_list.append(element)
   return return_list

def predict_stuff(model,sample_data):
   '''
   Apply forward propagation and return 
   probability distribution of predictions
   Requires: sample_data to be of shape:
   (num_samples,128,128), where 128 is 
   width and height of images
   Returns: probability distribution of 
   predictions in descending order. Its shape:
   (num_samples,num_classes)
   '''
   m = sample_data.shape[1]
   sample_data = sample_data.reshape((-1, m, m, 1))
   output = model.predict(sample_data)
   prob_dist = np.argsort(output,axis=1)
   sorted_output = np.flip(np.sort(output,axis=1),axis=1)
   reversed_prob_dist = np.flip(prob_dist,axis=1) #convert to decreasing order (high to low probs)
   return reversed_prob_dist

def category_to_char_old(number):
   labeldict = category_to_char = {
           0:"5",1:"J",2:"K",3:"L",4:"m",5:"N",
           6:"o",7:"Z",8:"j",9:"k",10:"l",11: "O",
           12:"z",13: "0",14: "1",15: "2",16: "3",
           17: "4",18: "6",19: "7",20: "8",21: "9",
           22: "A",23:"B",24:"C",25:"D",26:"E",27:"F",
           28: "G",29:"H",30:"I",31:"P",32:"Q",33:"R",
           34: "S", 35: "T", 36: "U", 37: "V", 38: "Y",
           39: "a", 40: "b", 41: "d", 42: "e", 43: "f",
           44: "g", 45: "i", 46: "p", 47: "q", 48: "s",
           49: "t", 50: "u", 51: "y", 52: "W", 53: "c", 
           54: "r", 55: "v", 56: "w", 57: "x"
   }
   return labeldict[number]

def return_raw_array(extracted,model):
   '''
   Takes in the output of extract_letters.
   Returns a final string consisting of NNs prediction
   '''
   word_array = []
   temp_word = ""
   temp_word_list = []
   all_possibilities = []
   for element_index in range(len(extracted)):
       element = extracted[element_index]
       if type(element) != str:            
           eleman = element[np.newaxis,:,:] #(128,128)-->(1,128,128)
           results = predict_stuff(model,eleman)
           predicted_category = int(results[0][0]) #find most likely category label
           predicted_char = category_to_char_old(predicted_category)
           possibles = (results[0][:5])
           possible_chars = [category_to_char_old(possible) for possible in possibles]
           temp_word += predicted_char
           temp_word_list.append(possible_chars)
           if element_index == len(extracted) - 1:
               word_array.append(temp_word)
               all_possibilities.append(temp_word_list)
       elif element == "newword":
           word_array.append(temp_word)
           word_array.append("")
           temp_word = ""
           all_possibilities.append(temp_word_list)
           temp_word_list = []
       elif element == "newline":
           word_array.append(temp_word)
           word_array.append("\n")
           temp_word = ""
           all_possibilities.append(temp_word_list)
           temp_word_list = []
   return (word_array, all_possibilities)

#[["H","A","h","b","j"],["e","y","4","3","w"]] --> He, Hy, H4, H3 etc...
def all_permutations(char_list):
   '''
   Please give small words ~len(5) to 
   this function. Grows very fast!'''
   permutations = [p for p in itertools.product(*char_list)] #tuple permutations
   string_list = ["".join(p) for p in permutations]
   return string_list


def edits1(word):
   alphabet = 'abcdefghijklmnopqrstuvwxyz'
   s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in s if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
   inserts    = [a + c + b     for a, b in s for c in alphabet]
   return set(deletes + transposes + replaces + inserts)


def handle_suggests(word):
   english_dict = enchant.Dict("en_US") #consider modularizing en_US
   if len(word) == 0: 
       return []
   edit1s = edits1(word)
   suggests = list(english_dict.suggest(word))
   suggests = sorted(suggests,key=lambda sth: distance.levenshtein(sth, word),reverse=False)
   suggestions = []
   for suggest in suggests:
       suggestions.append(suggest)
   for edit in edit1s:
       suggestions.append(edit)
   return suggestions

def filter_correct_words(word_permutes):
   english_dict = enchant.Dict("en_US") #consider modularizing en_US
   #first try all the Top-5 combinations
   for word in word_permutes:
       try:
           correctly_spelled = english_dict.check(word)
           if correctly_spelled:
               return word
       except:
           print("Shouldn't have reached here. Word is: ",word)
           continue
   print ("Started looking at English vocab and edit-distance...")
   for word in word_permutes:
       suggested_words = handle_suggests(word)
       for suggested in suggested_words:
           correctly_spelled = english_dict.check(suggested)
           if correctly_spelled and not(suggested.isdigit()):
               return suggested
   print ("Wasn't able to find relevant words! Returning the Top-1 prediction...")
   return word_permutes[0] #if nothing works out, just return whatever found by Top-1

def compute_final_outcome(raw_array,possible_permutations):
   results = []
   for i in range(len(possible_permutations)):
       lis = possible_permutations[i]
       word_permutes = all_permutations(lis)
       results.append(filter_correct_words(word_permutes))
   counter =0
   final = []
   for i in range(len(raw_array)):
       element = raw_array[i]
       if len(element)==0 or element=='\n':
           final.append(element)
           continue
       else:
           final.append(results[counter])
           counter += 1 
   return final

def ocr(pic_name,neural_net_path):
   print("Loading the Neural Network Model...")
   model = load_model(neural_net_path) #Load Neural Net
   # get_ipython().run_line_magic('matplotlib', 'inline')
   im = np.array(Image.open(pic_name).convert("L"))
   plt.figure()
   plt.imshow(im)
   print("Extracting letters from the image...")
   extracted = letter_extract(pic_name, threshold = 190, pixel_count=30, min_seperation_coeff = 2)
   resized_extracted = handle_resize(extracted,64)
   print("Getting initial predictions from the Neural Network...")
   raw_array, possible_permutations = return_raw_array(resized_extracted,model)
   print ("Raw outcome is: ", raw_array)
   print("Polishing and self-correcting Neural Network output...")
   FINAL_OUTCOME = (compute_final_outcome(raw_array,possible_permutations))
   FINAL_STRING = ""
   for s in FINAL_OUTCOME:
       if len(s) == 0: 
           FINAL_STRING += " "
       else:
           FINAL_STRING += s
   return FINAL_STRING
pic_name = "phrase.jpg"
neural_net_path = "my_deep_model.h5"
print (ocr(pic_name,neural_net_path))

#UNCOMMENT WHEN YOU WANT TO TUNE HYPERPARAMETERS
# english_dict = enchant.Dict("en_US")
# print(set(english_dict.suggest("Gome")))
# print(distance.levenshtein("Game", "Gome"))
# DEBUGGING AND HYPERPARAMETER TUNING. UNCOMMENT WHEN NEEDED.
# %matplotlib inline
# picname = "handwritten3.jpg"
# a = np.array(Image.open(picname).convert("L"))
# print(a.shape)

# extracted = letter_extract(picname, th = 190, pixel_count=30)
# resized_extracted = handle_resize(extracted,64)
# plt.figure()
# countw = 0
# countl = 0
# for element in resized_extracted:
#     if type(element) != str:
#         eleman = element[np.newaxis,:,:]

#         print("Extracted letter's shape: ", eleman.shape)
#         probs = predict_stuff(model,eleman)
#         display(Image.fromarray(element,mode="L").resize((128,128)))
#         print ("Top 5 predictions are:")
#         print (category_to_char_old(probs[0][0]),category_to_char_old(probs[0][1]),
#                category_to_char_old(probs[0][2]),category_to_char_old(probs[0][3]),
#                category_to_char_old(probs[0][4]))
#         countw += 1
#     elif element == "newword":
# #         print("cw", countw)
#         countw = 0
#         countl += 1
#     elif element == "newline":
#         countl += 1
# #         print("cw", countw)
# #         print("cl", countl)
#         countl = 0
#         countw = 0