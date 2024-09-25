from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import os

import pandas as pd
import numpy as np
from typing import List, Dict

DATA_DIR_NAME = 'datasets'
PAGE_LINK = "https://timeseriesclassification.com/results/PublishedResults/"

def get_html_page_and_prepare_soup(page_link: str) -> BeautifulSoup:
    """
    Fetches an HTML page from a given URL and parses it into a BeautifulSoup object.

    Args:
        - page_link (str): URL of the page to be fetched.

    Returns:
        - BeautifulSoup: Parsed HTML content of the page.
    """
    response = requests.get(page_link)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    return soup

def get_content_list_from_html(soup: BeautifulSoup, tag_name: str) -> List[str]:
    """
    Extracts text content from all HTML elements with the given tag and returns it as a list of strings.

    Args:
        - soup (BeautifulSoup): Parsed HTML content (soup object).
        - tag_name (str): The name of the HTML tag to search for.

    Returns:
        - List[str]: A list of text content from each found HTML element.
    """
    return [elem.get_text().strip() for elem in soup.find_all(tag_name)]

def load_model_results(paper_list: List[str], need_download: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    '''
    Takes paper_list, downloads it from the 'timeseriesclassification.com' website if need_download is True
    to DATA_DIR_NAME/paper_name directories. Then opens all model's results csv files as pd.DataFrames.

    Args:
        - paper_list: List[str] - The list with titles of papers with "name/" format
        - need_download - The bool specifies whether the models should be loaded from the website or only loaded into Python.

    Returns:
        - Dictionary[str, Dict[str, pd.DataFrame]]:
            * keys: string of paper name
            * values: a dictionary of models in pd.DataFrame format 
    '''
    paper_models_dict = {}
    
    for paper in paper_list:
        if not os.path.isdir(DATA_DIR_NAME + "/" + paper[:-1]):
            os.mkdir(DATA_DIR_NAME + "/" + paper[:-1])
            
        print(f"Parsing {paper[:-1]} models...\n")
        page_link_paper = PAGE_LINK + paper
    
        soup_i = get_html_page_and_prepare_soup(page_link_paper)
        models_list = get_content_list_from_html(soup_i, 'a')[1:]
    
        pd_models_dict = {}
        for model_name in models_list:
            if need_download:
                file_response = requests.get(page_link_paper + model_name, stream=True)
        
                with open(DATA_DIR_NAME + '/' + paper + model_name, "wb") as handle:
                    for data in tqdm(file_response.iter_content()):
                        handle.write(data)
            pd_models_dict.update({model_name.split('_')[0]:pd.read_csv(DATA_DIR_NAME + '/' + paper + model_name)})
        paper_models_dict.update({paper[:-1]:pd_models_dict})
    return paper_models_dict