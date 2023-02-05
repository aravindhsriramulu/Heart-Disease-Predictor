#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st

def load_html(file_path):
    with open(file_path, 'r') as f:
        html_content = f.read()
        st.write(html_content, unsafe_allow_html=True)

if __name__ == '__main__':
    load_html(file_path)


# In[ ]:




