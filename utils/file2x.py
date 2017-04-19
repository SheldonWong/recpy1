#coding=utf8
import sys
import gzip
import os
import fileutils

'''
file2x模块作用
借助常用file API，将原始文件转换成相应的格式，并存储
API结构如下：
file2dict
file2iur
file2iurr
file2uir
file2uirr
gzip2uir
gzip2uirr
get_format_file(file_list,mode,separator,outputPath)
'''
#获取日志对象
logger = get_logger('file2x')





'''
@desc:将原始文件读成字典列表，每个user-item对应一个字典
形如：
{
	user:
	item:
	rating:
	review:
}
@param:path(路径)
@return:dict_list(字典列表)
'''
def file2dict(path):
	l = []
	d = {}
	with open(path) as f:
		for line in f:
			if(line != '\n'):
				(key,value) = line.split(':')
				d[key] = value
			else:
				l.append(d)
	return l

'''
@desc:将原始文件转换成形如
item:user:rating\n
的格式，其中冒号代表separator，可指定该参数
@param:path,separator
@return:str(item:user:rating\n组合成的字符串)
'''
def file2iur(path,separator):
	str = ''
	with open('path') as f:
		for line in f:
			if('product/productId' in line):
				str = str + line.strip('\n').split(': ')[1] + separator
			if('review/userId' in line):
				str = str + line.strip('\n').split(': ')[1] + separator
			if('review/score' in line):
				str = str + line.strip('\n').split(': ')[1]
				str = str + '\n'
	return str	


'''
@desc:将原始文件转换成
item:user:rating:review\n
的格式
@param:path,separator
@return:str
'''
def file2iurr(path,separator):
	str = ''
	with open(path) as f:
		for line in f:
			if('product/productId' in line):
				str = str + line.strip('\n').split(': ')[1] + separator
			if('review/userId' in line):
				str = str + line.strip('\n').split(': ')[1] + separator
			if('review/score' in line):
				str = str + line.strip('\n').split(': ')[1] + separator
			if('review/text' in line):
				str = str + line.strip('\n').split(': ')[1]
				str = str + '\n'
	return str			

'''
@desc:将原始文件转换成
user:item:rating\n
的格式，ItemID在前，UserID在后，如何读取成uir格式？（用一个临时变量先存储起来，在拼接）
@param:path,separator
@return:str
'''

def file2uir(path,separator):
	str = ''
	with open(path) as f:
		for line in f:
			if('product/productId' in line):
				temp = line.strip('\n').split(': ')[1] + separator
			if('review/userId' in line):
				str = str + line.strip('\n').split(': ')[1] + separator + temp 
			if('review/score' in line):
				str = str + line.strip('\n').split(': ')[1]
				str = str + '\n'
	return str	

'''
@desc:将原始文件转换成
user:item:rating:review\n
的格式
@param:path,separator
@return:str
'''
def file2uirr(path,separator):
	str = ''
	with open(path) as f:
		for line in f:
			if('product/productId' in line):
				temp = line.strip('\n').split(': ')[1] + separator
			if('review/userId' in line):
				str = str + line.strip('\n').split(': ')[1] + separator + temp 
			if('review/score' in line):
				str = str + line.strip('\n').split(': ')[1] + separator
			if('review/text' in line):
				str = str + line.strip('\n').split(': ')[1]
				str = str + '\n'
	return str
'''
@desc:直接读取gzip文件,转换成uir文件
@param:path,separator
@return:str
'''
def gzip2uir(path,separator):
	str = ''
	with gzip.open(path,'rt') as f:
		for line in f:
			if('product/productId' in line):
				temp = line.strip('\n').split(': ')[1] + separator
			if('review/userId' in line):
				str = str + line.strip('\n').split(': ')[1] + separator + temp 
			if('review/score' in line):
				str = str + line.strip('\n').split(': ')[1]
				str = str + '\n'
	return str


'''
@desc:直接读取gzip文件,转换成uirr文件
@param:path,separator
@return:str
'''
def gzip2uirr(path,separator):
	str = ''
	with gzip.open(path,'rt') as f:
		for line in f:
			if('product/productId' in line):
				temp = line.strip('\n').split(': ')[1] + separator
			if('review/userId' in line):
				str = str + line.strip('\n').split(': ')[1] + separator + temp 
			if('review/score' in line):
				str = str + line.strip('\n').split(': ')[1] + separator
			if('review/text' in line):
				str = str + line.strip('\n').split(': ')[1]
				str = str + '\n'
	return str



'''
@desc:获取指定格式的文件，传进一个文件列表参数(gzip格式的文件)，输出n个uirr类型的文件
@param:file_list,mode(uir or uirr),separator(自己指定),output_path
@return:写入成功后，返回写入的路径
'''
def get_format_file(file_list,mode,separator,output_path):
	content = ''
	exist = fileutils.read_dir(output_path)
	for file in file_list:
		#构造输出路径文件
		out = output_path+file.split('/')[-1].split('.')[0]+'@'+mode+'.csv'
		if(out in exist):
			logger.info('already exist: '+out)
			continue
		if(mode == 'uir' and 'Amazon_Instant_Video' in file):
				logger.info('start convert gzip file to uir...'+file)
				content = gzip2uir(file,separator)
				logger.info('convert gzip file to uir finished...'+file)
				filename = output_path+file.split('/')[-1].split('.')[0]+'@'+mode+'.csv'
				with open(filename,'w') as f:
					f.write(content)
				logger.info('write uir to file finished: '+filename)
		if(mode == 'uirr'):
				logger.info('start convert gzip file to uirr...'+file)
				content = gzip2uirr(file,separator)
				logger.info('convert gzip file to uir finished...'+file)
				filename = output_path+file.split('/')[-1].split('.')[0]+'@'+mode+'.csv'
				with open(filename,'w') as f:
					f.write(content)
				logger.info('write uirr to file finished: '+filename)
	return


file_list = fileutils.read_dir_by_filter('j:/amazon/input/','.gz')
get_format_file(file_list,'uir',',','j:/amazon/output/')