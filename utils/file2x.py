#coding=utf8
import fileutils
import gzip
from logger import get_logger
import pandas as pd 


#获取日志对象
logger = get_logger('file2x')


'''
@descpritor:gzip文件转换成user-item-rating文件
@param:path
@return:uir_list
'''
def gzip2uir(path):
    with gzip.open(path,'rt') as f:
        reviews = f.read()
    reviews_list = reviews.split('\n\n')

    uir_list = []
    for i in range(len(reviews_list) - 1):
        uir = [reviews_list[i].split('\n')[3].split(': ')[1],
               reviews_list[i].split('\n')[0].split(': ')[1],
               reviews_list[i].split('\n')[6].split(': ')[1]]
        uir_list.append(uir)
    uir_df = pd.DataFrame(uir_list)
    return uir_df

'''
@descriptor:gzip文件转化成user-item-rating-review文件
@param:path
@return:uirr_list
'''
def gzip2uirr(path):
    with gzip.open(path,'rt') as f:
        reviews = f.read()
    reviews_list = reviews.split('\n\n')
    
    uirr_list = []
    for i in range(len(reviews_list) - 1):
        uirr = [reviews_list[i].split('\n')[3].split(': ')[1],
                  reviews_list[i].split('\n')[0].split(': ')[1],
                  reviews_list[i].split('\n')[6].split(': ')[1],
                  reviews_list[i].split('\n')[9].split(': ')[1]]
        uirr_list.append(uirr)
    uirr_df = pd.DataFrame(uirr_list)
    return uirr_df

'''
@desc:获取指定格式的文件，传进一个文件列表参数(gzip格式的文件)，输出n个uirr类型的文件
@param:file_list,mode(uir or uirr),output_path
@return:写入成功后，返回写入的路径
'''
def get_format_file(file_list,mode,output_path):
	content = ''
	exist = fileutils.read_dir(output_path)
	for file in file_list:
		#构造输出路径文件
		out = output_path+file.split('/')[-1].split('.')[0]+'@'+mode+'.csv'
		if(out in exist):
			logger.info('already exist: '+out)
			continue
		if(mode == 'uir'):
				logger.info('start convert gzip file to uir...'+file)
				content = gzip2uir(file)
				logger.info('convert gzip file to uir finished...'+file)
				filename = output_path+file.split('/')[-1].split('.')[0]+'@'+mode+'.csv'
				content.to_csv(filename)
				logger.info('write uir to file finished: '+filename)
		if(mode == 'uirr'):
				logger.info('start convert gzip file to uirr...'+file)
				content = gzip2uirr(file)
				logger.info('convert gzip file to uirr finished...'+file)
				filename = output_path+file.split('/')[-1].split('.')[0]+'@'+mode+'.csv'
				content.to_csv(filename)
				logger.info('write uirr to file finished: '+filename)
    
	return

file_list = fileutils.read_dir_by_filter('j:/amazon/input/','.gz')
get_format_file(file_list,'uirr','j:/amazon/output2/')