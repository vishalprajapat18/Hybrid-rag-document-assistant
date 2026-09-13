from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter= RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=50,
    separators=["\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

def create_chunks(text:str)-> list[str]:
    return splitter.split_text(text)