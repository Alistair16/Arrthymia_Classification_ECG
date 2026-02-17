#!/usr/bin/env python
# coding: utf-8

# In[5]:


## Mapping via SNOMED codes

def extract_group(header_path):
    group_map = {
        # --- SR group ---
        "426783006": "SR",   # SR
        "427084000": "SR",   # ST
        "427393009": "SR",   # SA

        #----AFIB group---
        "164889003": "AFIB", # AFIB
        "164890007": "AFIB", # AFL

        #----GSTV group ---
        "426761007": "GSTV", # SVT
        "713422000": "GSTV", # AT
        "233896004": "GSTV", # AVNRT
        "233897008": "GSTV", # AVRT
        "195101003": "GSTV", # SAAWR/WAP

        # --- SB group ---
        "426177001": "SB"    # SB
    }

    # Open the file at the path header_path, use 'r' means read mode
    # Returns a file object (like a handle to the file)
    #Assign the file object to the variable f
    # with statement ensures the file is properly closed after the block finsishes, even if an error occurs
    #Don't need to call f.close() manually
    with open(header_path,'r') as f:
        for line in f:
            if line.startswith("#") and "Dx" in line:
                codes = line.split(":")[1] #Split the diagnosis using :, the seond part will contain the codes

                # Splits the string at every comma. 
                # The results is a list of sbustring
                # c.strip() removes the leading and trailing whitespace from each substring
                codes = [c.strip() for c in codes.split(",")] 


                primary_code = codes[0] # use first code only

                #group_map.get(key, default) looks up key in the dictionary group_map
                #IF the key exists, it returns the corresponding value
                #If the key does not exist, it returns the default value instead of throwing an error
                return group_map.get(primary_code, "UNKNOWN")

    #return unknow if the header has no Dx line or the line is formatted differently
    return "UNKNOWN"


# In[ ]:




