
import nltk
import validators
import streamlit as st
from transformers import AutoTokenizer, pipeline

# local modules
import torch
from summarizer import Summarizer
model = Summarizer()
from utilis import (
    clean_text,
    fetch_article_text,
    preprocess_text_for_abstractive_summarization,
    read_text_from_file,
)

if __name__ == "__main__":
     
        st.markdown("<h1 style='text-align: center;'>Text Summarization Tool 📝</h1>", unsafe_allow_html=True)
        st.markdown("<p color: grey;'>As name indicates it is an online tool for text summary. But the special thing about this app is that it gives summary of research papers.It allows you to summarize your important reserch papers or any other documents by taking up the important concepts.</p>", unsafe_allow_html=True)

    
        
         # SETUP & Constants
        nltk.download("punkt")
        abs_tokenizer_name = "facebook/bart-large-cnn"
        abs_model_name = "facebook/bart-large-cnn"
        abs_tokenizer = AutoTokenizer.from_pretrained(abs_tokenizer_name)
        abs_max_length = 90
        abs_min_length = 30

    
        inp_text = st.text_input("Enter text or a url here")
        st.markdown(
            "<h5 style='text-align: center; color: Red;'>OR</h5>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload file for summarization"
        )


        is_url = validators.url(inp_text)
        
        if is_url:
            # complete text, chunks to summarize (list of sentences for long docs)
            text, clean_txt = fetch_article_text(url=inp_text)
        elif uploaded_file:
            clean_txt = read_text_from_file(uploaded_file)
            clean_txt = clean_text(clean_txt)
        else:
            clean_txt = clean_text(inp_text)
          # view summarized text (expander)
        with st.expander("View input text"):
              if is_url:
                  st.write(clean_txt[0])
              else:
                  st.write(clean_txt)
        summarize = st.button("Summarize")


         # view summarized text (expander)
        if is_url:
             text_to_summarize = " ".join([txt for txt in clean_txt])
        else:
             text_to_summarize = clean_txt
        with st.spinner(
            text=" This might take a few seconds ..."
        ):
            ext_model = Summarizer()
            summarized_text = ext_model(text_to_summarize, num_sentences=5)
            
      
        # final summarized output
        st.subheader("Summarized text")
        st.info(summarized_text)