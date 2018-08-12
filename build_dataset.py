'''
use flickr api to retrive images of top ingredients
use google image scraper to to retrive images as well
'''

'''
png to jpg only
leave jpg, jpeg, and jfif files
go through after and remove all files that do not match correct form 
TO DO
    support for other file types
    conversion to jpg for all file types
    api's for other search engines
    
    
looks like outliers are being grouped together
but need a stronger preloaded model to further differentiate
    and create feature vectors that are more distinct
    
    after that do more clustering and try to find a way to ID outliers
'''

DEBUG = True

import urllib.request
from urllib.request import FancyURLopener
#from urllib.request import Request, urlopen
from apiclient.discovery import build
from flickrapi import FlickrAPI
from pathlib import Path
from PIL import Image
import shutil
from glob import glob                                                           
import cv2 
import os

FLICKR_API_KEY = '*****'
FLICKR_API_SECRET = '*****'
CSE_API_KEY = '*****'
CSE_SEARCH_ID = '*****'

PATH = './test_scrape/'
URL_TYPE = ['url_o', 'url_c', 'url_q', 'url_s', 'url_n', 'url_m', 'url_sq', 'url_t']
EXTRAS = 'url_c,url_m,url_n,url_o,url_q,url_s,url_sq,url_t'
FILE_FORMATS = ['jpeg','jpg','jfif']
#FILE_FORMATS = ['jpeg','jfif','jpg','png','exif']

class RepeatSearchException(Exception):
    pass

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

class FixFancyURLOpener(FancyURLopener):
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        if errcode == 403:
            raise ValueError("403")
        return super(FixFancyURLOpener, self).http_error_default(
            url, fp, errcode, errmsg, headers)


def convert_to_jpg(path_to_image):
    new_path = '.'+''.join(path_to_image.split('.')[:-1]) + '.jpg'
    img = cv2.imread(path_to_image)
    cv2.imwrite(new_path,img)
    return path_to_image


def square_up(path_to_image,side=256):
    '''
    takes an image via path and crops it so it is square then saves it at same location
    default dimension is 256x256
    '''
    image = Image.open(path_to_image)
    width,height = image.size[0], image.size[1]
    aspect = width / float(height)
    
    ideal_width, ideal_height = side,side
    ideal_aspect = ideal_width / float(ideal_height)
     
    if aspect > ideal_aspect:
        # Then crop the left and right edges:
        new_width = int(ideal_aspect * height)
        offset = (width - new_width) / 2
        resize = (offset, 0, width - offset, height)
    else:
        # ... crop the top and bottom:
        new_height = int(width / ideal_aspect)
        offset = (height - new_height) / 2
        resize = (0, offset, width, height - offset)
     
    square = image.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)
    square.save(path_to_image)


def search_flickr(keyword):
    '''
    return [url]
    url is link to image from flickr
    '''
    if DEBUG: print('\t...searching flickr...')
    images_per_page = 100
    
    url_list = []
    failed = 0
    for page_num in range(1,6):
        response = flickr.photos.search(text=keyword,
                                        tags=keyword,
                                        tag_mode='all',
                                        per_page=images_per_page,
                                        page=page_num,
                                        sort='relevance',
                                        extras=EXTRAS)
        photos = response['photos']['photo']
        for photo in photos:
            for url_type in URL_TYPE:
                try:
                    url_list.append(photo[url_type])
                    failed -= 1
                    break
                except:
                    pass
            failed += 1
    if DEBUG: print('\t...retreived {} images ({} failed)'.format(len(url_list), failed))
    return url_list


def search_google(keyword):
    '''
    return [url]
    url is link to image from google image search
    '''
    if DEBUG: print('\t...searching google...')
    images_per_page = 10
    last_page = 101
    
    url_list = []
    failed = 0
    for file in FILE_FORMATS:
        for page in range(1,last_page,10):
            response = service.cse().list(
                q=keyword,
                cx=CSE_SEARCH_ID,
                searchType='image',
                num=images_per_page,
                imgType='photo',
                fileType=file,
                start=page,
                safe='off'
            ).execute()
            
            if 'items' in response:
                photos = response['items']
                for photo in photos:
                    try:
                        url_list.append(photo['link'])
                    except:
                        failed += 1
    if DEBUG: print('\t...retreived {} images ({} failed)'.format(len(url_list), failed))
    return url_list
        
        
def create_directory(keyword):
    '''
    creates directory if doesn't exist
    returns number of files in target directory
    '''
    path_to_dir = PATH + keyword + '/'
    path_to_file = path_to_dir + keyword
    if os.path.exists(path_to_dir):
        shutil.rmtree(path_to_dir)
    os.makedirs(path_to_dir)
    return path_to_file
        
        
def save_images(list_of_urls,path):
    '''
    downloads all images
    '''
    if DEBUG: print('\t{} images to save...'.format(len(list_of_urls)))
    global FILE_NUM
    failed = 0
    
    for url in list_of_urls:
        extension = url.split('.')[-1].split('?')[0]
        file_name = '{}{}.{}'.format(path,FILE_NUM,extension)

        try:
            opener = AppURLopener()
            response = opener.open(url)
            file_path = Path(file_name)
            file_path.touch()
            out = open(file_name, 'wb')
            out.write(response.read())
            out.close()
            square_up(file_name)
        except:
            failed += 1
        else:
            FILE_NUM += 1
    print('\t...saved {} images ({} failed)'.format(FILE_NUM, failed))


def download_images_by_keyword(keyword):
    path = create_directory(keyword)
    urls = search_flickr(keyword) + search_google(keyword)
    save_images(urls, path)
    

if __name__ == "__main__":

    flickr=FlickrAPI(FLICKR_API_KEY,FLICKR_API_SECRET,format='parsed-json')
    service = build("customsearch", "v1", developerKey=CSE_API_KEY)
    searches = ['puppies']
    
    for item in searches:
        FILE_NUM = 0
        if DEBUG: print('scraping images for {}...'.format(item))
        
        download_images_by_keyword(item.lower())
        
        FILE_NUM = 0
        if DEBUG: print('\t...finished scraping for {}\n'.format(item))
        
    if DEBUG: print('done')
