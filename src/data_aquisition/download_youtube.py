import pytube
import pandas as pd 
import os as os 
from tqdm import tqdm 
from argparse import ArgumentParser
import sys
os.getcwd()

def download(URL, titles, output_dir, sub_dir):
#Utility to download a series of youtube videos given as a list of URLs and a list of titles 
    full_path = os.path.join(output_dir, sub_dir)
    pbar = tqdm(zip(URL, titles))

    for url, title in pbar:  
        title = title.replace(" ", "_")
        title = "".join((title, ".mp4"))
        print(title)
        youtube = pytube.YouTube(url)
        video = youtube.streams.filter(res='720p').first()
        video.download(output_path=full_path, filename=title)   

def parse_file(excel_file):
    #Utility to parse an excel file from http://www.williamsportwebdeveloper.com/FavBackUp.aspx
    video_metadata = pd.read_excel(excel_file)
    video_metadata = video_metadata.rename(columns=video_metadata.iloc[0]).drop(video_metadata.index[0])
    videos = video_metadata['Video URL'].to_list()
    titles = video_metadata['Title'].to_list()
    return videos, titles



if __name__ == "__main__":
    parser = ArgumentParser(description="Download Videos From Excel Binary")
    parser.add_argument("-f", type=str, required=True, default=None)
    parser.add_argument("-o", type=str, required=True,default=None)
    parser.add_argument("-s", type=str, required=False, default = "")
    args = parser.parse_args()
    file = args.f
    output=args.o
    sub_dir = args.s
    URLs, Titles = parse_file(file)
    download(URLs, Titles, output, sub_dir)

