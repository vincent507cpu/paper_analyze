class Config:
    hf_token = "..."
    target_year = '23'
    target_month = '05'
    skip = 0
    show = 2000
    download_papers = False
    save_dir = '../../arXiv_paper_collection'
    abstract_vector_store_path = 'abstract_vector_store'
    full_text_vector_store_path = 'full_text_vector_store'
    target_category = 'cs.LG'
    embedding_name = "BAAI/bge-base-en-v1.5"
    llm_name = 'togethercomputer/Llama-2-7B-32K-Instruct'
    retrieve_number = 50